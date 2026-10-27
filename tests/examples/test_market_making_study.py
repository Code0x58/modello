"""Independent numerical checks and reproducibility of the complete study."""

import json

import numpy as np
import pytest
from sympy import S, assuming, exp

from examples._market_making_models import CARAMarketMaker, LogisticIntensity, study_case
from examples._market_making_numerics import QuoteEvaluator, exponential_reference, run_quotes
from examples.market_making import CARAUtility, QuadraticTerminalPenalty
from examples.market_making_study import _provenance, compare, write_study
from modello import Coordinate
from modello.contrib.research import ResearchError


@pytest.fixture(scope="module")
def comparison():
    return compare(capacity=2, points=21)


@pytest.mark.parametrize("family", ["exponential", "logistic"])
def test_terminal_boundary_symmetry_and_convergence(comparison, family):
    record = comparison[family]
    data = record["result"].data
    q = np.array(data.inventory)
    assert np.max(np.abs(np.array(data.theta[-1]) + 0.05 * q * q)) < 1e-12
    assert all(row[-1] is None for row in data.bid)
    assert all(row[0] is None for row in data.ask)
    np.testing.assert_allclose(data.theta, np.array(data.theta)[:, ::-1], atol=1e-10)
    for bid, ask in zip(data.bid, data.ask, strict=True):
        np.testing.assert_allclose(bid[:-1], ask[:0:-1], atol=1e-9)
    assert record["convergence"]["max_theta_difference"] < 1e-6
    assert record["convergence"]["max_quote_difference"] < 1e-6
    diagnostics = data.diagnostics
    assert diagnostics["terminal_error"] == 0
    assert diagnostics["max_root_residual"] < 1e-9
    coarse, fine = diagnostics["time_difference_defects"]
    assert fine < coarse / 2 and fine < 1e-3
    assert all(item.verdict == "strictly true" for item in record["result"].report.checks)


def test_exponential_ode_agrees_with_the_independent_linear_transformation(comparison):
    record = comparison["exponential"]
    data = record["result"].data
    reference = exponential_reference(record["case"], data.times)
    np.testing.assert_allclose(data.theta, reference, rtol=1e-7, atol=1e-8)
    assert record["convergence"]["matrix_exponential_theta_error"] < 1e-6


@pytest.mark.parametrize("cost", [-3.0, 0.0, 2.0])
def test_general_quote_root_agrees_with_the_exponential_formula(cost):
    evaluator = QuoteEvaluator("exponential", 2, 1, 0.1, 1)
    closed = evaluator.evaluate(cost)
    rooted = evaluator.evaluate(cost, force_root=True)
    np.testing.assert_allclose(closed, rooted, atol=1e-10)


@pytest.mark.parametrize("cost", [-3.0, 0.0, 2.0])
def test_logistic_quote_is_a_global_maximum_under_the_checked_premises(cost):
    evaluator = QuoteEvaluator("logistic", 4, 2, 0.1, 1)
    quote, value, derivative = evaluator.evaluate(cost)
    assert quote > cost and derivative < 0
    # Compare the objective directly rather than its rearranged root equation.
    alternatives = np.linspace(quote - 3, quote + 3, 301)
    objective = 4 / (1 + np.exp(2 * alternatives)) * (-np.expm1(-0.1 * (alternatives - cost))) / 0.1
    assert value >= max(objective) - 1e-12
    h = 1e-5

    def objective_at(x):
        return 4 / (1 + np.exp(2 * x)) * (-np.expm1(-0.1 * (x - cost))) / 0.1

    assert abs((objective_at(quote + h) - objective_at(quote - h)) / (2 * h)) < 1e-8


def test_changing_utility_or_losing_uniform_evidence_blocks_the_method():
    class ConvexUtility(CARAUtility):
        value = exp(CARAUtility.gamma * CARAUtility.wealth)

    case = study_case(capacity=2)
    with pytest.raises(ResearchError) as error:
        run_quotes(case.replace("utility", ConvexUtility))
    assert error.value.report.explain("cara_form").verdict == "strictly false"
    logistic = study_case(logistic=True, capacity=2)
    from dataclasses import replace

    with pytest.raises(ResearchError):
        run_quotes(replace(logistic, hints=()))
    with pytest.raises(ResearchError) as error:
        exponential_reference(logistic, (0, 1))
    assert error.value.report.explain("exponential_bid").verdict == "strictly false"


def test_invalid_lattice_and_numerical_failure_retain_case_and_report():
    case = study_case(capacity=S(3) / 2)
    with pytest.raises(ResearchError, match="integral number") as error:
        run_quotes(case)
    assert error.value.case is case
    assert error.value.report.checks
    with pytest.raises(ResearchError, match="max_step"):
        run_quotes(study_case(capacity=1), max_step=0)


def test_asymmetric_intensities_use_independent_component_scopes():
    case = study_case(capacity=2).replace("ask.intensity", LogisticIntensity)
    # Replacement preserves the recorded scale=2, decay=1 on the ask side.
    case = case.with_hint("ask.intensity.rate/curvature_bound", upper=1)
    data = run_quotes(case, points=5).data
    assert data.diagnostics["bid_family"] == "exponential"
    assert data.diagnostics["ask_family"] == "logistic"
    assert data.bid[0][2] != pytest.approx(data.ask[0][2])


def test_report_artifacts_are_complete_and_replayable(tmp_path):
    import matplotlib

    original_font = matplotlib.rcParams["font.size"]
    original_salt = matplotlib.rcParams["svg.hashsalt"]
    payload = write_study(tmp_path, capacity=1, points=5)
    assert matplotlib.rcParams["font.size"] == original_font
    assert matplotlib.rcParams["svg.hashsalt"] == original_salt
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["cases"]["logistic"]["bid"][0][-1] is None
    assert (tmp_path / "report.html").read_text().startswith("<!doctype html>")
    assert (tmp_path / "evidence-logistic.html").is_file()
    assert "sup &lt;=" in (tmp_path / "evidence-logistic.html").read_text()
    assert (tmp_path / "theta.csv").read_text().startswith("family,time,inventory,theta")
    assert (tmp_path / "quotes.csv").read_text().startswith("family,time,inventory,bid_distance,ask_distance")
    assert len(list((tmp_path / "figures").glob("*.svg"))) == 4
    assert saved["provenance"]["versions"]["scipy"]
    assert saved["provenance"]["source_sha256"]
    assert "examples/_research_checks.py" in saved["provenance"]["source_sha256"]
    replayed = compare(capacity=1, points=5)
    for name, record in replayed.items():
        np.testing.assert_allclose(payload["cases"][name]["theta"], record["result"].data.theta, atol=1e-10)
        assert (
            payload["cases"][name]["evidence"]["evidence_sha256"]
            == record["result"].report.to_dict()["evidence_sha256"]
        )


def test_study_is_a_general_hjb_without_a_retained_exponential_optimizer():
    assert ("bid", "optimal_distance") not in CARAMarketMaker._modello_definition.fields
    assert ("ask", "optimal_distance") not in CARAMarketMaker._modello_definition.fields


def test_reference_derives_terminal_inputs_and_validates_its_domain():
    class Quartic(QuadraticTerminalPenalty):
        cost = QuadraticTerminalPenalty.eta * QuadraticTerminalPenalty.inventory**4

    case = study_case(capacity=2).replace("terminal", Quartic)
    reference = exponential_reference(case, (0, 1))
    np.testing.assert_allclose(reference[-1], [-0.8, -0.05, 0, -0.05, -0.8], atol=1e-12)
    actual = run_quotes(case, points=3).data
    np.testing.assert_allclose(actual.theta[0], reference[0], atol=1e-6)
    with pytest.raises(ResearchError, match="integral number"):
        exponential_reference(study_case(capacity=S(3) / 2), (0, 1))
    with pytest.raises(ResearchError, match="within the model horizon"):
        exponential_reference(case, (-1, 1))


def test_coordinate_renaming_preserves_the_declared_gain_and_numerical_model():
    from examples._market_making_models import StudiedExponential

    class Renamed(StudiedExponential):
        distance = Coordinate("new_delta", real=True)
        rate = StudiedExponential.scale * exp(-StudiedExponential.decay * distance)

    case = study_case(capacity=1)
    changed = case.replace("ask.intensity", Renamed)
    gain = changed.expression("ask.gain", wrt="ask.intensity.distance")
    assert StudiedExponential.distance not in gain.free_symbols
    np.testing.assert_allclose(
        run_quotes(changed, points=3).data.theta, run_quotes(case, points=3).data.theta, atol=1e-9
    )


def test_git_is_optional_provenance(monkeypatch):
    import subprocess

    original = subprocess.run

    def without_git(command, *args, **kwargs):
        if command[0] == "git":
            raise FileNotFoundError("No Git executable")
        return original(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", without_git)
    provenance = _provenance()
    assert provenance["git_revision"] is provenance["working_tree_dirty"] is None
    assert provenance["source_sha256"]


def test_numerical_adapter_rechecks_the_declared_neighbor_and_gain_laws():
    from examples._market_making_models import QuoteHamiltonian
    from examples.market_making import ContinuationValue

    class WrongNeighbor(ContinuationValue):
        bid_cost = (ContinuationValue.current - ContinuationValue.after_ask) / ContinuationValue.step

    class WrongGain(QuoteHamiltonian):
        gain = -QuoteHamiltonian.gain

    case = study_case(capacity=1)
    for path, replacement, claim in (
        ("continuation", WrongNeighbor, "bid_neighbor_cost"),
        ("bid", WrongGain, "bid_cara_gain"),
    ):
        with pytest.raises(ResearchError) as error:
            run_quotes(case.replace(path, replacement), points=3)
        assert error.value.report.explain(claim).verdict != "strictly true"


def test_full_ode_rejects_fixed_state_observations_and_assumptions():
    case = study_case(capacity=1)
    for changed, message in (
        (case.observe(inventory=0), "fixed state observations"),
        (case.observe(bid_cost=0), "fixed state observations"),
        (case.assume(case.symbols.inventory >= 0), "state coordinates vary"),
        (case.assume(case.symbols.bid_cost >= 0), "state coordinates vary"),
        (case.assume(case._reference("terminal.inventory") > 0), "state coordinates vary"),
        (case.assume(case._reference("bid.intensity.distance") < 0), "state coordinates vary"),
        (case.assume(case._reference("ask.intensity.distance") < 0), "state coordinates vary"),
    ):
        with pytest.raises(ResearchError, match=message):
            run_quotes(changed, points=3)


@pytest.mark.parametrize(
    "subject", ["inventory", "bid_cost", "terminal.inventory", "bid.intensity.distance", "ask.intensity.distance"]
)
def test_full_ode_rejects_global_state_restrictions(subject):
    case = study_case(capacity=1)
    coordinate = case._reference(subject)
    condition = coordinate < 0 if subject.endswith("distance") else coordinate >= 0
    with assuming(condition):
        with pytest.raises(ResearchError, match="state coordinates vary") as failure:
            run_quotes(case, points=3)
        assert failure.value.report.assumptions
