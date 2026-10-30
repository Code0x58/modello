"""Exact POV certificates and independent numerical execution checks."""

import subprocess
import sys
from dataclasses import replace

import pytest
from sympy import Eq, S, assuming, diff

from examples.research_models.participation_execution import (
    POSITIVE_RATES,
    CappedParticipation,
    ParticipationExecution,
    SurchargedPowerImpact,
    analytic_method,
    numerical_method,
    participation_case,
    symbolic_case,
    uniqueness_theorem,
    with_limits,
)
from modello.contrib.research import ResearchCase, ResearchError, convex


@pytest.fixture(scope="module")
def free_case():
    return participation_case()


@pytest.fixture(scope="module")
def free_result(free_case):
    return free_case.run(analytic_method)


def test_exact_solution_checks_live_identities_uniqueness_and_kkt(free_result):
    result = free_result
    solution = result.data
    assert (solution.status, solution.rate, solution.duration, solution.cost) == ("minimum", S.One / 4, 3, S(9) / 4)
    assert solution.active_bounds == ()
    assert (solution.lower_multiplier, solution.upper_multiplier) == (0, 0)
    for name in (
        "pov_objective_identity",
        "completion_time_identity",
        "constant_pov_inventory_identity",
        "inventory_risk_integral",
        "log_coordinate_identity",
        "pure_power_law",
    ):
        assert result.report.explain(name).condition is S.true
    for name in (
        "scaled_first_order_identity",
        "scaled_slope_strictly_increasing",
        "completion_boundary",
        "pov_kkt_certificate",
    ):
        assert solution.certificate.explain(name).condition is S.true
    uniqueness = solution.certificate.explain("at_most_one_pov_minimum")
    assert uniqueness.kind == "theorem"
    assert uniqueness.condition is S.true
    assert len(uniqueness.premises) == 4
    assert all(premise.condition is S.true for premise in uniqueness.premises)


def test_sublinear_per_share_cost_does_not_require_convexity_in_rate(free_case, free_result):
    rate = free_case.expression("rate", wrt="rate")
    objective = free_case.expression("objective", wrt="rate")
    assert diff(objective, rate, 2).subs(rate, 1) == -S(3) / 8
    claim = convex("objective", wrt="rate", domain=POSITIVE_RATES, name="convex_in_rate")
    report = free_case.require(replace(claim, consumer="test", reason="This objective is not convex in rho.")).check()
    assert report.explain("convex_in_rate").condition is S.false
    assert free_result.data.rate == S.One / 4


@pytest.mark.parametrize(
    ("cap", "deadline", "expected_rate", "expected_duration", "active"),
    [
        (S.One / 8, None, S.One / 8, 6, ("cap",)),
        (None, 2, S(3) / 8, 2, ("deadline",)),
        (S.Half, 2, S(3) / 8, 2, ("deadline",)),
        (S(3) / 8, 2, S(3) / 8, 2, ("deadline", "cap")),
        (S.Half, 4, S.One / 4, 3, ()),
    ],
)
def test_endpoints_and_singleton_feasibility_are_certified(
    free_case, cap, deadline, expected_rate, expected_duration, active
):
    case = with_limits(free_case, cap=cap, deadline=deadline)
    solution = case.run(analytic_method(case)).data
    assert (solution.rate, solution.duration, solution.active_bounds) == (expected_rate, expected_duration, active)
    assert solution.certificate.explain("pov_kkt_certificate").condition is S.true
    assert solution.certificate.explain("completion_boundary").condition is S.true
    if active == ("cap",):
        assert solution.upper_multiplier > 0
    if "deadline" in active:
        assert solution.lower_multiplier > 0


def test_constraints_can_be_removed_without_changing_old_branches(free_case):
    bounded = with_limits(free_case, cap=S.One / 8, deadline=8)
    cap_removed = with_limits(bounded, deadline=8)
    deadline_removed = with_limits(bounded, cap=S.One / 8)
    free_again = with_limits(bounded)
    rates = [
        case.run(analytic_method(case)).data.rate
        for case in (bounded, cap_removed, deadline_removed, free_again, free_case)
    ]
    assert rates == [S.One / 8, S.One / 4, S.One / 8, S.One / 4, S.One / 4]
    assert cap_removed.expression("limits.cap") == S.One / 8
    assert cap_removed.expression("limits.cap_active") == 0
    assert deadline_removed.expression("limits.deadline") == 8
    assert deadline_removed.expression("limits.deadline_active") == 0


@pytest.mark.parametrize("method_factory", [analytic_method, numerical_method])
def test_infeasible_deadline_and_cap_block_both_methods(free_case, method_factory):
    case = with_limits(free_case, cap=S.One / 8, deadline=2)
    with pytest.raises(ResearchError) as failure:
        case.run(method_factory(case))
    assert failure.value.case is case
    assert failure.value.report.explain("feasible_limits").condition is S.false


@pytest.mark.parametrize("cap", [None, S.One / 8])
def test_risk_neutral_free_horizon_has_only_an_unattained_infimum(cap):
    case = participation_case(gamma=0, cap=cap)
    solution = case.run(analytic_method(case)).data
    assert solution.status == "infimum_not_attained"
    assert (solution.rate, solution.duration, solution.cost) == (None, None, 0)
    assert solution.certificate.explain("cost_above_infimum").condition is S.true
    assert solution.certificate.explain("unattained_zero_limit").condition is S.true
    numerical = case.run(numerical_method(case)).data
    assert (numerical.status, numerical.rate, numerical.duration, numerical.cost) == (
        "infimum_not_attained",
        None,
        None,
        0.0,
    )


def test_risk_neutral_deadline_attains_its_lower_endpoint():
    case = participation_case(gamma=0, deadline=2)
    solution = case.run(analytic_method(case)).data
    assert (solution.rate, solution.duration, solution.active_bounds) == (S(3) / 8, 2, ("deadline",))
    assert solution.lower_multiplier > 0
    assert solution.certificate.explain("pov_kkt_certificate").condition is S.true


def test_parameter_assumption_branches_keep_their_evidence():
    template = symbolic_case().observe(quantity=3, volume=4, eta=1, phi=S.Half, sigma=1)
    positive_risk = template.assume(template.symbols.gamma > 0)
    exact = positive_risk.run(analytic_method(positive_risk)).data
    assert exact.rate is not None and template.symbols.gamma in exact.rate.free_symbols
    assert exact.certificate.explain("pov_kkt_certificate").condition is S.true
    neutral = template.assume(Eq(template.symbols.gamma, 0))
    solution = neutral.run(analytic_method(neutral)).data
    assert solution.status == "infimum_not_attained"
    with pytest.raises(ResearchError):
        template.run(analytic_method(template))


@pytest.mark.parametrize("cap,deadline", [(None, None), (S.One / 8, None), (None, 2), (S(3) / 8, 2)])
def test_numerical_minimization_and_quadrature_agree_with_exact_results(free_case, cap, deadline):
    case = with_limits(free_case, cap=cap, deadline=deadline)
    exact = case.run(analytic_method(case)).data
    numerical = case.run(numerical_method(case)).data
    assert numerical.rate == pytest.approx(float(exact.rate), rel=1e-7, abs=1e-10)
    assert numerical.duration == pytest.approx(float(exact.duration), rel=1e-7)
    assert numerical.cost == pytest.approx(float(exact.cost), rel=1e-12)
    assert numerical.diagnostics["quadrature_objective_difference"] < 1e-12
    assert numerical.diagnostics["log_kkt_violation"] < 1e-7
    assert numerical.log_bracket[0] <= numerical.log_bracket[1]


def test_impact_replacement_invalidates_shortcut_but_retains_general_method(free_case):
    changed = free_case.replace("impact", SurchargedPowerImpact).observe(
        impact=SurchargedPowerImpact("surcharge", surcharge=S.One / 4)
    )
    with pytest.raises(ResearchError) as failure:
        changed.run(analytic_method(changed))
    assert failure.value.report.explain("pure_power_law").condition is S.false
    assert failure.value.report.explain("at_most_one_pov_minimum").condition is S.true
    result = changed.run(numerical_method(changed))
    assert result.data.rate == pytest.approx(0.21740361757157867, rel=1e-7)
    assert result.data.rate < S.One / 4
    assert free_case.run(analytic_method(free_case)).data.rate == S.One / 4


@pytest.mark.parametrize("subject", ["rate", "log_rate", "time"])
def test_optimizers_reject_hidden_restrictions_on_the_search_argument(free_case, subject):
    coordinate = free_case._reference(subject)
    restricted = free_case.assume(coordinate < S.One / 8)
    with pytest.raises(ResearchError, match="with_limits"):
        restricted.run(analytic_method(restricted))


@pytest.mark.parametrize("method", [analytic_method, numerical_method])
@pytest.mark.parametrize("subject", ["rate", "log_rate", "time"])
def test_optimizers_reject_global_rate_restrictions(free_case, method, subject):
    coordinate = free_case._reference(subject)
    condition = coordinate >= S.Half
    with assuming(condition):
        with pytest.raises(ResearchError, match="with_limits") as failure:
            free_case.run(method)
        assert condition in failure.value.report.assumptions


def test_observing_completion_time_cannot_silently_fix_the_rate(free_case):
    restricted = free_case.observe(duration=3)
    with pytest.raises(ResearchError, match="derived outputs"):
        restricted.run(analytic_method(restricted))


def test_a_redefined_curvature_cannot_supply_a_uniqueness_certificate():
    class WrongCurvature(ParticipationExecution):
        log_curvature = S.One

    case = (
        ResearchCase.for_model(WrongCurvature)
        .with_theorem(uniqueness_theorem())
        .observe(
            quantity=3,
            volume=4,
            eta=1,
            phi=S.Half,
            sigma=1,
            gamma=S.One / 6,
        )
    )
    with pytest.raises(ResearchError) as failure:
        case.run(analytic_method(case))
    evidence = failure.value.report.explain("at_most_one_pov_minimum/actual_log_curvature")
    assert evidence.condition is S.false


def test_numerical_parameter_and_setting_failures_retain_research_context(free_case):
    unknown_cap = free_case.replace("limits", CappedParticipation)
    with pytest.raises(ResearchError, match="Observe the active") as failure:
        unknown_cap.run(numerical_method(unknown_cap))
    assert failure.value.case is unknown_cap
    with pytest.raises(ResearchError, match="xatol") as failure:
        free_case.run(numerical_method(free_case), xatol=float("nan"))
    assert failure.value.case is free_case


def test_base_import_does_not_load_optional_numerical_or_plotting_packages():
    code = (
        "import sys; import examples.research_models.participation_execution; "
        "assert not {'numpy', 'scipy', 'matplotlib'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
