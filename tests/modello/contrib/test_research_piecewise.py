"""Finite-join regularity checks keep domains, guards, and refutations distinct."""

from dataclasses import replace

import pytest
from sympy import Derivative, Eq, FiniteSet, Ge, Gt, Interval, Le, Lt, Ne, Piecewise, S, Subs, Symbol, diff, limit, sin

from modello import Coordinate, InstanceDummy
from modello.contrib.research import (
    Method,
    ResearchCase,
    ResearchError,
    ResearchModel,
    Theorem,
    differentiable,
    exact_form,
    smooth,
)
from modello.contrib.research._piecewise import piecewise_regular
from modello.contrib.research._proof import _differentiable, _regular

x = Symbol("x", real=True)


def certificate(expression, order, domain=S.Reals):
    result = piecewise_regular(expression, x, domain, order, _regular)
    assert result is not None
    return result


def test_matching_values_and_slopes_do_not_hide_a_second_derivative_jump():
    expression = Piecewise((3 - 2 * x, x <= 1), (x**-2, True))
    for order in (0, 1):
        assert certificate(expression, order).condition is S.true
    result = certificate(expression, 2)
    assert result.condition is S.false
    assert result.failure_order == 2
    assert result.witness == ((x, S.One),)


def test_a_value_jump_has_failure_order_zero():
    expression = Piecewise((0, x < 0), (1, True))
    result = certificate(expression, 0)
    assert result.condition is S.false
    assert result.failure_order == 0


def test_a_first_derivative_cusp_is_continuous_but_not_c1():
    expression = Piecewise((-x, x <= 0), (x, True))
    assert certificate(expression, 0).condition is S.true
    result = certificate(expression, 1)
    assert result.condition is S.false and result.failure_order == 1


@pytest.mark.parametrize("order", [0, 1, 2])
def test_distinct_smooth_branches_join_through_order_two(order):
    expression = Piecewise((x**2, x <= 0), (x**2 + x**3, True))
    assert certificate(expression, order).condition is S.true


def test_the_actual_assigned_endpoint_value_must_match_the_active_side():
    expression = Piecewise((99, x <= 1), (x, True))
    result = certificate(expression, 1, Interval(1, 2))
    assert result.condition is S.false and result.failure_order == 0


def test_an_inactive_branch_at_a_domain_endpoint_can_only_certify_continuity():
    expression = Piecewise((1, x <= 1), (x**-2, True))
    assert certificate(expression, 0, Interval(1, 2)).condition is S.true
    result = certificate(expression, 2, Interval(1, 2))
    assert result.condition is None and result.failure_order is None


def test_an_included_endpoint_does_not_inspect_a_remote_inactive_pole():
    expression = Piecewise((1 / (x - S.Half), x <= 1), (2 * x**-2, True))
    assert certificate(expression, 0, Interval(1, 2)).condition is S.true
    for order in (1, 2):
        result = certificate(expression, order, Interval(1, 2))
        assert result.condition is None and result.failure_order is None


def test_an_excluded_join_and_its_inactive_pole_do_not_block_regularity():
    expression = Piecewise((1 / (x - 1), x <= 1), (x**2, True))
    calls = []

    def record(value, variable, domain, order):
        calls.append(value)
        return _regular(value, variable, domain, order)

    result = piecewise_regular(expression, x, Interval.open(1, S.Infinity), 2, record)
    assert result.condition is S.true
    assert calls == [x**2]


def test_an_undefined_assigned_join_value_is_not_a_finite_mismatch_certificate():
    expression = Piecewise((1 / (x - 1), x <= 1), (x**2, True))
    result = certificate(expression, 0, Interval(1, 2))
    assert result.condition is S.false and result.failure_order is None
    assert result.witness == ((x, S.One),)


def test_equal_finite_side_limits_cannot_hide_an_undefined_assigned_value():
    expression = Piecewise((sin(x) / x, x <= 0), (1 + x**2, True))
    result = certificate(expression, 1)
    assert result.condition is S.false and result.failure_order is None
    assert result.witness == ((x, S.Zero),)


@pytest.mark.parametrize("relation", [Le, Lt, Ge, Gt])
@pytest.mark.parametrize("swapped", [False, True])
def test_strict_reversed_and_swapped_comparisons_preserve_side_orientation(relation, swapped):
    condition = relation(1, x, evaluate=False) if swapped else relation(x, 1, evaluate=False)
    expression = Piecewise((3 - 2 * x, condition), (x**-2, True))
    assert certificate(expression, 1, Interval.open(0, S.Infinity)).condition is S.true
    result = certificate(expression, 2, Interval.open(0, S.Infinity))
    assert result.condition is S.false and result.failure_order == 2


def test_parameter_equalities_are_retained_without_guessing_a_verdict():
    a, b = Symbol("a", real=True), Symbol("b", real=True)
    expression = Piecewise((a * x, x <= 1), (b * x, True))
    result = certificate(expression, 2)
    assert result.condition == Eq(a - b, 0)
    assert result.failure_order is None


def test_parameter_poles_keep_the_source_guard_when_the_join_matches():
    a = Symbol("a", real=True)
    expression = Piecewise((x / a, x <= 1), (x / a + (x - 1) ** 3, True))
    result = certificate(expression, 2)
    assert result.condition == Ne(a, 0)
    assert Ne(a, 0) in result.guards


def test_a_finite_mismatch_under_an_unestablished_source_guard_is_unresolved():
    a = Symbol("a", real=True)
    expression = Piecewise((x / a, x <= 1), (2 * x / a, True))
    result = certificate(expression, 0)
    assert result.condition is None and result.failure_order is None
    assert Ne(a, 0) in result.guards
    assert str(Ne(a, 0)) in " ".join(result.steps)


def test_a_nonfinite_one_sided_limit_refutes_regularity_without_derivative_mismatch_metadata():
    expression = Piecewise((1 / x, x < 0), (0, True))
    result = certificate(expression, 1)
    assert result.condition is S.false and result.failure_order is None


def test_an_oscillatory_limit_is_left_unresolved():
    expression = Piecewise((sin(1 / x), x < 0), (0, True))
    result = certificate(expression, 0)
    assert result.condition is None and result.failure_order is None


def test_a_failed_branch_c2_certificate_does_not_become_a_join_derivative_refutation():
    expression = Piecewise((1 / (x + 1), x <= 0), (x, True))
    result = certificate(expression, 2)
    assert result.condition is S.false
    assert result.failure_order is None


def test_singleton_domains_only_assert_continuity_of_the_actual_finite_value():
    expression = Piecewise((99, x <= 1), (x**2, True))
    assert certificate(expression, 0, FiniteSet(1)).condition is S.true
    assert piecewise_regular(expression, x, FiniteSet(1), 1, _regular) is None


@pytest.mark.parametrize("order", [-1, 3, True])
def test_unsupported_orders_remain_unsupported(order):
    expression = Piecewise((x, x <= 0), (x**2, True))
    assert piecewise_regular(expression, x, S.Reals, order, _regular) is None


def test_multiple_and_parameter_dependent_joins_remain_unsupported():
    a = Symbol("a", real=True)
    multiple = Piecewise((x, x < 0), (x**2, x < 1), (x**3, True))
    parameter = Piecewise((x, x <= a), (x**2, True))
    singleton_branch = Piecewise((0, Eq(x, 1)), (x, True))
    for expression in (multiple, parameter, singleton_branch):
        assert piecewise_regular(expression, x, S.Reals, 1, _regular) is None


def test_the_requested_domain_cannot_exceed_the_variables_intrinsic_domain():
    positive_x = Symbol("positive_x", positive=True)
    expression = Piecewise((positive_x, positive_x <= 1), (positive_x**2, True))
    assert piecewise_regular(expression, positive_x, S.Reals, 0, _regular) is None
    result = piecewise_regular(expression, positive_x, Interval.open(0, S.Infinity), 0, _regular)
    assert result.condition is S.true


def public_check(case, claim):
    return case.check(
        requirements=(replace(claim, consumer="integration", reason="Check the public finite-join contract."),)
    ).explain(claim.name)


def test_public_c1_and_c2_verdicts_and_native_derivative_guards_use_the_actual_join():
    class Joined(ResearchModel):
        x = Coordinate("x", positive=True)
        value = Piecewise((3 - 2 * x, x <= 1), (x**-2, True))

    case = ResearchCase.for_model(Joined)
    domain = Interval.open(0, S.Infinity)
    assert public_check(case, smooth("value", wrt="x", domain=domain, order=1)).condition is S.true
    c2 = public_check(case, smooth("value", wrt="x", domain=domain, order=2))
    assert c2.condition is S.false and c2.witness == ((Joined.x, S.One),)
    assert public_check(case, differentiable("value", wrt="x", domain=domain, order=1)).condition is S.true
    at_join = public_check(case, differentiable("value", wrt="x", domain=domain, order=2, at=1))
    assert at_join.condition is S.false and at_join.witness == ((Joined.x, S.One),)
    first = exact_form(Subs(Derivative(case.symbols.value, Joined.x), Joined.x, 1), -2, name="first_derivative")
    assert public_check(case, first).condition is S.true
    second = exact_form(Subs(Derivative(case.symbols.value, Joined.x, 2), Joined.x, 1), 0, name="second_derivative")
    with pytest.raises(ResearchError) as failure:
        case.run(
            Method(
                "evaluate second derivative",
                lambda current: "must remain blocked",
                (
                    replace(
                        second, consumer="integration", reason="A native derivative needs actual differentiability."
                    ),
                ),
            )
        )
    assert failure.value.report.explain("second_derivative").condition is S.false


def test_public_included_join_endpoint_cannot_discharge_ordinary_derivative_premises():
    class Endpoint(ResearchModel):
        x = Coordinate("x", positive=True)
        value = Piecewise((1, x <= 1), (x**-2, True))

    case = ResearchCase.for_model(Endpoint)
    domain = Interval(1, 2)
    for claim in (
        smooth("value", wrt="x", domain=domain, order=1),
        smooth("value", wrt="x", domain=domain, order=2),
        differentiable("value", wrt="x", domain=domain, order=1),
        differentiable("value", wrt="x", domain=domain, order=1, at=1),
    ):
        assert public_check(case, claim).condition is None
    ordinary = exact_form(
        Derivative(case.symbols.value, Endpoint.x),
        Piecewise((0, Endpoint.x <= 1), (-2 * Endpoint.x**-3, True)),
        wrt="x",
        domain=domain,
        name="ordinary_derivative",
    )
    with pytest.raises(ResearchError) as failure:
        case.run(
            Method(
                "ordinary derivative on a closed interval",
                lambda current: "must remain blocked",
                (replace(ordinary, consumer="integration", reason="The endpoint requires an ordinary derivative."),),
            )
        )
    assert failure.value.report.explain("ordinary_derivative").condition is None


@pytest.mark.parametrize("domain", [Interval(1, 2), FiniteSet(1)])
@pytest.mark.parametrize("order", [1, 2])
def test_guarded_smoothness_requires_the_same_neighborhood_as_differentiability(domain, order):
    class Guarded(ResearchModel):
        x = Coordinate("x", real=True)
        value = Piecewise((x**2, x >= 1), (S.NaN, True))

    case = ResearchCase.for_model(Guarded)
    claim = smooth("value", wrt="x", domain=domain, order=order)
    evidence = public_check(case, claim)
    assert evidence.verdict == "unresolved"
    assert not evidence.witness
    calls = []
    with pytest.raises(ResearchError):
        case.run(Method("guarded derivatives", lambda current: calls.append(current), (claim,)))
    assert not calls
    continuity = _regular(Piecewise((x**2, x >= 1), (S.NaN, True)), x, domain, 0)
    assert continuity.condition is S.true
    interior = smooth("value", wrt="x", domain=Interval.open(1, 2), order=order)
    assert public_check(case, interior).verdict == "strictly true"


def test_public_local_refutations_cannot_be_carried_to_a_different_point():
    class Joined(ResearchModel):
        x = Coordinate("x", positive=True)
        value = Piecewise((3 - 2 * x, x <= 1), (x**-2, True))

    case = ResearchCase.for_model(Joined)
    result = public_check(case, differentiable("value", wrt="x", domain=Interval.open(0, S.Infinity), order=2, at=2))
    assert result.condition is not S.false
    assert not result.witness


@pytest.mark.parametrize("point", (-1, 1))
@pytest.mark.parametrize("order", (1, 2))
@pytest.mark.parametrize("observed", (False, True))
def test_local_piecewise_requirements_do_not_require_matching_inactive_branches(point, order, observed):
    class Joined(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = Piecewise((a, x < 0), (b, True))

    case = ResearchCase.for_model(Joined)
    case = case.observe(a=0, b=1) if observed else case.assume(Ne(case.symbols.a, case.symbols.b))
    local = differentiable("value", wrt="x", order=order, at=point, name="local")
    native = exact_form(Subs(Derivative(case.symbols.value, Joined.x, order), Joined.x, point), 0, name="native")
    conclusion = replace(local, name="reused")
    case = case.with_theorem(
        Theorem("reuse local requirement", conclusion, (local,), "The same mathematical proposition is retained.")
    )
    result = case.run(Method("local derivative", lambda current: point, (local, native, conclusion)))
    assert result.data == point
    assert all(item.verdict == "strictly true" for item in result.report.checks)
    assert result.report.explain("reused/local").verdict == "strictly true"
    at_join = replace(local, at=S.Zero)
    assert case.check(requirements=(at_join,)).explain("local").verdict == "strictly false"
    global_claim = smooth("value", wrt="x", order=order)
    assert case.check(requirements=(global_claim,)).explain(global_claim.name).verdict == "strictly false"


@pytest.mark.parametrize("point", (-1, 1))
def test_local_piecewise_checks_retain_only_active_branch_source_guards(point):
    class Guarded(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = Piecewise((1 / a, x < 0), (1, True))

    case = ResearchCase.for_model(Guarded)
    claim = differentiable("value", wrt="x", at=point)
    expression = Piecewise((1 / case.symbols.a, Guarded.x < 0), (1, True))
    evidence = _differentiable(expression, Guarded.x, S.Reals, 1, S(point))
    expected = Ne(case.symbols.a, 0) if point < 0 else S.true
    assert evidence.condition == expected
    evidence = case.observe(a=0).check(requirements=(claim,)).explain(claim.name)
    assert evidence.condition is (S.false if point < 0 else S.true)


def test_public_failure_of_c2_does_not_refute_existing_second_differentiability():
    class Oscillating(ResearchModel):
        x = Coordinate("x", real=True)
        value = Piecewise((0, x <= 0), (x**4 * sin(1 / x), True))

    # Both second difference-quotient limits are zero, although the right
    # second-derivative curve oscillates and supplies no C2 join certificate.
    assert limit(diff(x**4 * sin(1 / x), x) / x, x, 0, dir="+") == 0
    case = ResearchCase.for_model(Oscillating)
    result = public_check(case, differentiable("value", wrt="x", order=2, at=0))
    assert result.condition is not S.false


def test_public_unsupported_multiple_and_parameter_joins_remain_unresolved():
    class Multiple(ResearchModel):
        x = Coordinate("x", positive=True)
        value = Piecewise((x, x < 1), (x**2, x < 2), (x**3, True))

    class ParameterJoin(ResearchModel):
        x = Coordinate("x", positive=True)
        a = InstanceDummy("a", real=True)
        value = Piecewise((x, x <= a), (x**2, True))

    for model in (Multiple, ParameterJoin):
        result = public_check(
            ResearchCase.for_model(model), smooth("value", wrt="x", domain=Interval.open(0, S.Infinity))
        )
        assert result.condition is None and not result.witness
