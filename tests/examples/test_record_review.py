"""One record-review study: exact calculus, reusable premises, and exported evidence."""

import json
from pathlib import Path

import pytest
from sympy import FiniteSet, Rational, Symbol, exp, log, simplify, sqrt

from examples.record_review import (
    CappedReview,
    CheckingStep,
    EntryBudget,
    ExponentialReview,
    RationalReview,
    RecordWork,
    ReviewPlanning,
    derive_checking_response,
    review_context,
    review_plan,
    write_reports,
)
from modello.contrib.verification import VerificationContext, VerificationError, predicate


@pytest.fixture(scope="module")
def measured() -> VerificationContext[ReviewPlanning]:
    return review_context()


@pytest.fixture(scope="module")
def proposed(measured: VerificationContext[ReviewPlanning]) -> VerificationContext[ReviewPlanning]:
    return measured.observe(**{"candidate.review.effort": 2 * log(3)})


def test_short_checks_derive_the_continuous_response():
    short = CheckingStep(
        "short",
        record=RecordWork("record", entry_minutes=4, error_fraction=Rational(1, 10), repair_minutes=60),
        checking_minutes=Rational(1, 10),
        rate=Rational(1, 2),
    )
    assert short.caught_share == Rational(1, 20)
    assert 1000 * short.remaining_error_fraction == 95
    assert short.record.avoidable_minutes == 6
    assert short.expected_repair_minutes == Rational(57, 10)

    # Use the same name as the helper's bound variable to exercise its scope.
    effort = Symbol("N", integer=True, positive=True)
    rate = Symbol("k", real=True)
    response = derive_checking_response(effort, rate)
    assert response == 1 - exp(-rate * effort)
    assert response.subs(effort, 0) == 0
    assert response.subs(rate, 0) == 0
    context = VerificationContext.for_model(ExponentialReview)
    assert context.expression("caught_share", wrt="effort") == derive_checking_response(
        context.symbols.effort, context.symbols.rate
    )
    review = ExponentialReview("observed", effort=2, rate=Rational(1, 2))
    assert review.caught_share == 1 - exp(-1)


def test_definition_reports_distinguish_parameter_conditions_and_regularity():
    definition = ExponentialReview.definition_report()
    assert definition.explain("caught_share/strictly_increasing").verdict == "conditionally true"
    assert definition.explain("caught_share/concave").verdict == "strictly true"
    assert definition.explain("caught_share/continuously_differentiable").verdict == "strictly true"
    context = VerificationContext.for_model(ExponentialReview)
    positive_rate = context.assume(context.symbols.rate > 0)
    assert positive_rate.explain("caught_share/strictly_increasing").verdict == "strictly true"
    assert context.observe(rate=0).explain("caught_share/strictly_increasing").verdict == "strictly false"
    assert context.explain("caught_share/strictly_increasing").verdict == "conditionally true"


def test_stationary_candidate_has_a_global_certificate_and_local_curvature(proposed):
    result = proposed.run(review_plan)
    decision = result.data
    assert simplify(decision.effort - 2 * log(3)) == 0
    assert decision.net_saving == 4 - 2 * log(3)
    assert decision.minutes_per_record == 6 + 2 * log(3)
    assert decision.workload_curvature == Rational(1, 2)
    maximum = result.report.explain("global_maximum")
    assert maximum.verdict == "strictly true"
    premises = {item.obligation.name: item for item in maximum.premises}
    assert set(premises) == {
        "concavity",
        "differentiability",
        "stationarity",
        "feasibility",
    }
    assert all(item.verdict == "strictly true" for item in premises.values())
    assert simplify(maximum.expression - decision.net_saving) == 0
    # The rule checks the whole curve despite the observed checking time;
    # stationarity is checked at that observed point instead.
    axis = proposed.symbols.candidate.review.effort
    for name in ("concavity", "differentiability"):
        assert premises[name].expression.free_symbols == {axis}
        assert premises[name].coordinate == axis
    assert premises["stationarity"].expression.doit() == 0
    # Independent values of the workload on either side of the proposed point.
    workload = proposed.expression("candidate.expected_minutes", wrt="candidate.review.effort").doit()
    assert workload.subs(axis, 2) > decision.minutes_per_record
    assert workload.subs(axis, Rational(5, 2)) > decision.minutes_per_record


def test_capacity_and_staff_budget_can_be_supplied_in_either_direction(proposed):
    day = proposed.observe(budget=EntryBudget("day", staff_minutes=480))
    target = proposed.observe(budget=EntryBudget("target", records=60))
    assert day.run(review_plan).report.explain("global_maximum").verdict == "strictly true"
    assert target.run(review_plan).data == proposed.run(review_plan).data
    assert day.family.values(day.symbols.budget.records) == FiniteSet(simplify(480 / (6 + 2 * log(3))))
    assert target.family.values(target.symbols.budget.staff_minutes) == FiniteSet(360 + 120 * log(3))


def test_revised_inputs_recheck_the_optimum_and_its_derivative_domain(measured):
    record = RecordWork("revised", entry_minutes=3, error_fraction=Rational(1, 5), avoidable_minutes=8)
    revised = review_context(record, rate=Rational(1, 4))
    result = revised.observe(**{"candidate.review.effort": 4 * log(2)}).run(review_plan)
    assert result.report.explain("global_maximum").verdict == "strictly true"
    assert simplify(result.data.effort - 4 * log(2)) == 0
    assert simplify(result.data.minutes_per_record - 7 - 4 * log(2)) == 0
    assert result.data.workload_curvature == Rational(1, 4)
    assert measured.expression("candidate.record.avoidable_minutes") == 6
    assert revised.expression("candidate.record.repair_minutes") == 40

    # Formal substitution at zero does not establish a two-sided derivative
    # within the coordinate's nonnegative domain.
    endpoint_record = RecordWork("endpoint", entry_minutes=3, error_fraction=Rational(1, 5), avoidable_minutes=4)
    endpoint = review_context(endpoint_record, rate=Rational(1, 4)).observe(**{"candidate.review.effort": 0})
    assert endpoint.expression("candidate.marginal_saving").doit() == 0
    with pytest.raises(VerificationError) as error:
        endpoint.run(review_plan)
    assert error.value.report.explain("global_maximum/stationarity").verdict == "unresolved"


def test_a_wrong_candidate_and_an_unregistered_rule_do_not_establish_an_optimum(measured):
    wrong = measured.observe(**{"candidate.review.effort": 1})
    with pytest.raises(VerificationError) as error:
        wrong.run(review_plan)
    assert error.value.context is wrong
    assert error.value.report.explain("global_maximum/stationarity").verdict == "strictly false"
    assert error.value.report.explain("global_maximum").verdict == "unresolved"
    unregistered = VerificationContext.for_model(ReviewPlanning).observe(**{"candidate.review.effort": 1})
    with pytest.raises(VerificationError) as error:
        unregistered.run(review_plan)
    assert error.value.report.explain("global_maximum").verdict == "unresolved"


def test_replacement_rechecks_the_candidate_and_preserves_the_baseline(measured, proposed):
    stale = proposed.replace("candidate.review", RationalReview)
    with pytest.raises(VerificationError) as error:
        stale.run(review_plan)
    assert error.value.report.explain("global_maximum/stationarity").verdict == "strictly false"
    changed = measured.replace("candidate.review", RationalReview).observe(
        **{"candidate.review.effort": 2 * (sqrt(3) - 1)}
    )
    decision = changed.run(review_plan).data
    assert decision.net_saving == 8 - 4 * sqrt(3)
    assert decision.minutes_per_record == 2 + 4 * sqrt(3)
    assert decision.workload_curvature == sqrt(3) / 3
    assert measured.expression("candidate.review.rate") == Rational(1, 2)
    assert proposed.run(review_plan).data.minutes_per_record == 6 + 2 * log(3)


def test_the_capped_curve_retains_its_claims_and_exposes_its_kink(measured):
    capped = measured.replace("candidate.review", CappedReview).observe(**{"candidate.review.effort": 2})
    with pytest.raises(VerificationError) as error:
        capped.run(review_plan)
    regularity = error.value.report.explain("candidate.review.caught_share/continuously_differentiable")
    assert regularity.verdict == "strictly false"
    assert dict(regularity.witness)[capped.symbols.candidate.review.effort] == 2
    assert error.value.report.explain("global_maximum").verdict == "unresolved"


def test_retiring_an_exploratory_saving_target_does_not_waive_the_method_contract(proposed):
    target = proposed.with_obligations(
        predicate(proposed.symbols.candidate.net_saving >= 2, name="saving_target", consumer="review_plan")
    )
    with pytest.raises(VerificationError) as error:
        target.run(review_plan)
    assert error.value.report.explain("saving_target").verdict == "strictly false"
    baseline = target.without_obligations("saving_target", reason="Keep the measured baseline for comparison.")
    assert baseline.run(review_plan).report.explain("global_maximum").verdict == "strictly true"
    assert target.explain("saving_target").verdict == "strictly false"
    assert "Keep the measured baseline" in baseline.changes[-1]


def test_reports_include_exact_results_and_the_blocked_replacement(tmp_path: Path):
    write_reports(tmp_path, plot=True)
    summary = json.loads((tmp_path / "results.json").read_text())
    assert summary["inputs"] == {
        "candidate.record.entry_minutes": "4",
        "candidate.record.error_fraction": "1/10",
        "candidate.record.repair_minutes": "60",
        "candidate.review.rate": "1/2",
    }
    assert summary["plans"]["exponential"]["minutes_per_record"]["exact"] == "2*log(3) + 6"
    assert summary["plans"]["rational"]["minutes_per_record"]["decimal"] > 8.9
    assert "strictly false" in summary["plans"]["capped"]["blocked"]
    for name in ("exponential", "rational", "capped"):
        assert json.loads((tmp_path / f"{name}.json").read_text())["evidence"]
        assert "<!doctype html>" in (tmp_path / f"{name}.html").read_text()
    assert "Why spend time checking?" in (tmp_path / "benefit.svg").read_text()
    assert "Changing the response curve" in (tmp_path / "responses.svg").read_text()
