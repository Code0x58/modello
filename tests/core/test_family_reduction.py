"""Family reductions retain joint equations, domains, singularities, and fallback conditions."""

import pytest
from sympy import (
    ConditionSet,
    Eq,
    FiniteSet,
    Function,
    ImageSet,
    Interval,
    Rational,
    S,
    Symbol,
    Tuple,
    pi,
    sin,
)

import modello.solutions as solutions
from examples.solution_views import ModelViews
from modello import (
    InstanceDummy,
    Modello,
    UnresolvedSolutionError,
    inspect_model,
)

from ._models import Reading, Sum


def test_affine_joining_equations_and_preferred_coordinates():
    problem = inspect_model(Sum, "allocation", total=10)
    x, y = problem.symbols.x, problem.symbols.y
    family = problem.solutions(x, x).where(x >= 0, y >= 0)
    assert family.parameters == (problem.field("x"),)
    assert family.parameter_domain == Interval(0, 10)
    assert family.expression(y) == 10 - x
    assert family.values(x, y).contains(Tuple(4, 7)) is S.false
    case = family.observe(x=4).one()
    assert (case.x, case.y, case.total) == (4, 6, 10)
    equal = inspect_model(Sum, "equal")
    left, right = equal.symbols.x, equal.symbols.y
    joined = equal.solutions(left).where(Eq(left, right))
    assert joined.parameters == (equal.field("x"),)
    assert joined.expression(right) == left


def test_inconsistent_affine_observations_produce_an_empty_joint_family():
    class RepeatedSum(Modello):
        x = InstanceDummy("x", real=True)
        y = InstanceDummy("y", real=True)
        first = x + y
        second = x + y

    p = inspect_model(RepeatedSum, "contradictory totals", first=1, second=2)
    family = p.solutions()
    assert family.is_empty is True
    assert family.cases() == ()


def test_multivariate_fallback_keeps_the_joint_relation():
    class Circle(Modello):
        x = InstanceDummy("x", real=True)
        y = InstanceDummy("y", real=True)
        squared_radius = x**2 + y**2

    p = inspect_model(Circle, "circle", squared_radius=1)
    family = p.solutions()
    x, y = p.symbols.x, p.symbols.y
    assert len(family.parameters) == 2
    assert isinstance(family.parameter_domain, ConditionSet)
    joint = family.values(x, y)
    assert joint.contains(Tuple(1, 0)) is S.true
    assert joint.contains(Tuple(0, 1)) is S.true
    assert joint.contains(Tuple(0, 0)) is S.false
    assert family.observe(x=0, y=1).one().squared_radius == 1
    reference = Circle("reference", x=1, y=0)
    section = family.section(x, through=reference)
    assert {case.x for case in section.cases()} == {-1, 1}
    views = tuple(ModelViews(family, through=reference).sections())
    whole = next(view for view in views if len(view.controls) == 2)
    assert whole.control_dimension is None
    assert whole.image_dimension(x, y) is None
    discrete = next(view for view in views if view.controls == (p.field("x"),))
    assert discrete.control_dimension == discrete.image_dimension(x) == 0


@pytest.mark.parametrize(
    "assumptions, valid, invalid",
    [
        ({"positive": True}, 2, 0),
        ({"integer": True}, 2, Rational(1, 2)),
        ({"prime": True}, 2, 4),
        ({"zero": True}, 0, 1),
        ({"imaginary": True}, S.ImaginaryUnit, 1),
    ],
)
def test_declared_domains_remain_explicit_for_every_point(assumptions, valid, invalid):
    class Scalar(Modello):
        x = InstanceDummy("x", **assumptions)

    p = inspect_model(Scalar, "domain")
    x = p.symbols.x
    family = p.solutions()
    assert family.values(x).contains(valid) is not S.false
    assert family.observe(x=valid).one().x == valid
    assert family.observe(x=invalid).cases() == ()


def test_a_field_with_no_finite_values_makes_the_family_empty():
    class Infinite(Modello):
        x = InstanceDummy("x", infinite=True)

    family = inspect_model(Infinite, "finite-universe").solutions()
    assert family.is_empty is True
    assert family.cases() == ()


def test_reciprocal_cycle_retains_its_excluded_zero():
    class Reciprocal(Modello):
        x = InstanceDummy("x")
        y = 1 / x
        x = 1 / y

    p = inspect_model(Reciprocal, "reciprocal")
    family = p.solutions()
    assert family.parameter_domain.contains(0) is S.false
    assert family.parameter_domain.contains(S.ImaginaryUnit) is S.true
    assert family.observe(x=0).cases() == ()
    point = family.observe(x=2).one()
    assert (point.x, point.y) == (2, Rational(1, 2))


def test_real_membership_is_not_replaced_by_ordering_a_complex_expression():
    class PositiveImage(Modello):
        y = InstanceDummy("y")
        x = InstanceDummy("x", positive=True)
        x = S.ImaginaryUnit * y

    p = inspect_model(PositiveImage, "complex input")
    family = p.solutions(p.symbols.y)
    assert family.parameter_domain.contains(-S.ImaginaryUnit) is S.true
    assert family.parameter_domain.contains(1) is S.false
    point = family.observe(y=-S.ImaginaryUnit).one()
    assert point.x == 1


@pytest.mark.parametrize("coefficient", [Symbol("a"), Function("f")(1)])
def test_external_coefficients_keep_zero_cases_and_conditional_emptiness(coefficient):
    class Product(Modello):
        x = InstanceDummy("x", real=True)
        product = coefficient * x
        squared = x**2

    p = inspect_model(Product, "conditional", product=0, squared=1)
    family = p.solutions()
    assert family.is_empty is None
    assert family.parameter_domain.xreplace({coefficient: S.Zero}) == FiniteSet(-1, 1)
    assert family.parameter_domain.xreplace({coefficient: S.One}) is S.EmptySet
    # A constant image must still be empty when the condition is false.
    image = family.values(7)
    assert image.xreplace({coefficient: S.One}) is S.EmptySet
    with pytest.raises(UnresolvedSolutionError):
        family.cases()


def test_solver_failure_keeps_conditions_and_the_original_backend_error(monkeypatch):
    failure = NotImplementedError("not implemented")

    def broken(*args, **kwargs):
        raise failure

    monkeypatch.setattr(solutions, "solveset", broken)
    p = inspect_model(Reading, "conditional", squared=4)
    family = p.solutions()
    assert family.issues[0].error is failure
    assert isinstance(family.parameter_domain, ConditionSet)
    assert family.parameter_domain.contains(2) is S.true
    assert family.parameter_domain.contains(1) is S.false
    assert "Unresolved univariate reduction" in str(family)


def test_periodic_family_can_be_bounded_then_materialised():
    class Wave(Modello):
        angle = InstanceDummy("angle", real=True)
        sine = sin(angle)

    p = inspect_model(Wave, "wave")
    angle, sine = p.symbols.angle, p.symbols.sine
    family = p.solutions().where(Eq(sine, 0))
    assert family.parameter_domain.has(ImageSet)
    bounded = family.where(angle >= 0, angle <= 2 * pi)
    assert {case.angle for case in bounded.cases()} == {0, pi, 2 * pi}


def test_rational_affine_backend_failure_leaves_the_joined_relation(monkeypatch):
    failure = NotImplementedError("linear backend")

    def broken(*args, **kwargs):
        raise failure

    monkeypatch.setattr(solutions, "linsolve", broken)
    p = inspect_model(Sum, "fallback", total=1)
    family = p.solutions()
    assert family.issues[0].error is failure
    assert family.parameter_domain.contains(Tuple(1, 0)) is S.true
    assert family.parameter_domain.contains(Tuple(0, 0)) is S.false
