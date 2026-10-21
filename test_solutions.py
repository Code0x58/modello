"""Joint sets must preserve domains, singular cases, and relationships."""

from dataclasses import FrozenInstanceError

import pytest
from sympy import (
    ConditionSet,
    Derivative,
    Eq,
    FiniteSet,
    Function,
    ImageSet,
    Interval,
    MatrixSymbol,
    Rational,
    S,
    Symbol,
    Tuple,
    pi,
    sin,
    sqrt,
)

import modello.solutions as solutions
from modello import (
    ConstraintKind,
    InstanceDummy,
    ModelInputError,
    Modello,
    ModelSolveError,
    ModelValidationError,
    SolutionFamily,
    SolutionSelectionError,
    UnresolvedSolutionError,
    component,
    inspect_model,
)


class Reading(Modello):
    x = InstanceDummy("x", real=True)
    squared = x**2


class Check(Modello):
    reading = component(Reading)
    total = reading.squared + 1


def test_joint_family_refines_to_checked_typed_cases_without_running_legacy_solver(monkeypatch):
    import modello

    problem = inspect_model(Check, "explore")
    check = problem.symbols
    x, total = check.reading.x, check.total
    family = problem.solutions()
    assert isinstance(family, SolutionFamily)
    assert family.parameters == (problem.field("reading", "x"),)
    assert family.parameter_domain is S.Reals
    assert family.expression(total) == x**2 + 1
    assert family.expression(total - check.reading.squared) == 1
    assert family.format(family.expression(total)) == "reading.x**2 + 1"
    assert family.values(total) == Interval(1, S.Infinity)
    assert family.values(x).contains(2) is S.true
    assert family.values(x, total).contains(Tuple(2, 6)) is S.false
    assert family.values(x, total).contains(Tuple(2, 5)) is S.true
    assert family.values(total - check.reading.squared) == FiniteSet(1)
    assert "reading: Reading\n    x [coordinate]\n    squared = x**2" in str(family)
    assert "SolutionFamily(Check" in repr(family)
    with pytest.raises(FrozenInstanceError):
        family.parameters = ()

    def forbidden(*args, **kwargs):
        raise AssertionError("A checked family case must not rerun solve()")

    monkeypatch.setattr(modello, "solve", forbidden)
    observed = family.observe(total=5)
    assert observed.values(x, total) == FiniteSet(Tuple(-2, 5), Tuple(2, 5))
    cases = observed.cases()
    assert {case.reading.x for case in cases} == {-2, 2}
    assert all(isinstance(case, Check) and isinstance(case.reading, Reading) for case in cases)
    first, second = cases
    assert first._modello_bound_fields[Check.total].symbol != second._modello_bound_fields[Check.total].symbol
    assert first._modello_bound_fields[Check.total].kind is ConstraintKind.INPUT
    with pytest.raises(SolutionSelectionError) as caught:
        observed.one()
    assert caught.value.solution_count == 2
    assert caught.value.family is observed
    chosen = observed.where(x > 0).one()
    assert (chosen.reading.x, chosen.reading.squared, chosen.total) == (2, 4, 5)
    assert family.parameters == (problem.field("reading", "x"),) and family.parameter_domain is S.Reals
    assert all(c.input is None for c in problem.constraints)
    assert {c.input for c in observed.problem.constraints if c.input is not None} == {"total"}
    assert observed.symbols.total == total
    assert observed.observe(total=6).is_empty is True
    monkeypatch.setattr(SolutionFamily, "_prepare", forbidden)
    assert observed.observe() is observed
    assert observed.where() is observed


def test_exception_to_family_and_continuous_interval_observations():
    with pytest.raises(ModelSolveError) as caught:
        Check("ambiguous", total=5)
    family = caught.value.problem.solutions()
    assert {case.reading.x for case in family.cases()} == {-2, 2}
    p = inspect_model(Check, "band")
    total = p.symbols.total
    band = p.solutions().where(total >= 4, total <= 6)
    assert band.parameter_domain == Interval(-sqrt(5), -sqrt(3)) | Interval(sqrt(3), sqrt(5))
    with pytest.raises(UnresolvedSolutionError) as unresolved:
        band.one()
    assert unresolved.value.family is band
    assert unresolved.value.check is None
    empty = band.observe(total=2)
    assert empty.cases() == ()
    assert empty.is_empty is True
    assert empty.values(total) is S.EmptySet
    with pytest.raises(SolutionSelectionError) as no_point:
        empty.one()
    assert no_point.value.solution_count == 0


def test_typed_symbol_tree_and_nested_observations_share_constructor_provenance():
    class Joined(Modello):
        left = component(Check)
        right = component(Check)
        alias = left.reading.x

    problem = inspect_model(Joined, "same name")
    symbols = problem.symbols
    assert symbols is problem.symbols
    assert symbols.alias == symbols.left.reading.x
    assert symbols.left.reading.x != symbols.right.reading.x
    assert symbols.left.total != symbols.left.reading.squared + 1
    assert symbols.left.reading.x == problem.field("left", "reading", "x").symbol
    with pytest.raises(FrozenInstanceError):
        symbols.left.reading.x = 4
    with pytest.raises(AttributeError, match="no model attribute"):
        _ = symbols.left.reading.missing

    left = Check("left", reading=Reading("measured", x=2))
    right = Check("right", reading=Reading("measured", x=3))
    family = problem.solutions(symbols.alias).observe(left=left, right=right)
    assert family.values(symbols.left.total, symbols.right.total) == FiniteSet(Tuple(5, 10))
    assert family.symbols.alias == symbols.alias
    assert all(c.input is None for c in problem.constraints)
    copied = {c.path: c.kind for c in family.problem.constraints if c.input == "left"}
    assert copied[("left", "reading", "x")] is ConstraintKind.INPUT
    assert copied[("left", "reading", "squared")] is ConstraintKind.INFERRED
    assert copied[("left", "total")] is ConstraintKind.INFERRED
    case = family.one()
    assert isinstance(case.left, Check) and isinstance(case.left.reading, Reading)
    assert case.left._modello_bound_fields[Check.total].kind is ConstraintKind.INFERRED
    assert case.left.reading._modello_bound_fields[Reading.x].kind is ConstraintKind.INPUT
    assert family.observe(alias=3).cases() == ()
    with pytest.raises(ModelInputError, match="Check instance"):
        family.observe(left={"reading": {"x": 2}})


def test_affine_joining_equations_and_preferred_coordinates():
    class Sum(Modello):
        x = InstanceDummy("x", real=True)
        y = InstanceDummy("y", real=True)
        total = x + y

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
    assert family.parameter_domain.is_empty is None
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
    class Sum(Modello):
        x = InstanceDummy("x", real=True)
        y = InstanceDummy("y", real=True)
        total = x + y

    failure = NotImplementedError("linear backend")

    def broken(*args, **kwargs):
        raise failure

    monkeypatch.setattr(solutions, "linsolve", broken)
    p = inspect_model(Sum, "fallback", total=1)
    family = p.solutions()
    assert family.issues[0].error is failure
    assert family.parameter_domain.contains(Tuple(1, 0)) is S.true
    assert family.parameter_domain.contains(Tuple(0, 0)) is S.false


def test_materialisation_does_not_accept_an_undecided_candidate_audit(monkeypatch):
    from dataclasses import replace

    from modello import ModelProblem

    p = inspect_model(Reading, "point", x=2)
    check = ModelProblem.check

    def undecided(self, candidate):
        return replace(check(self, candidate), verdict="unknown")

    monkeypatch.setattr(ModelProblem, "check", undecided)
    family = p.solutions()
    with pytest.raises(UnresolvedSolutionError) as caught:
        family.one()
    assert caught.value.check.verdict == "unknown"
    assert caught.value.family is family


def test_unsupported_expression_kinds_fail_with_owned_input_errors():
    class Operators(Modello):
        x = InstanceDummy("x", commutative=False)

    class Unevaluated(Modello):
        x = InstanceDummy("x", real=True)
        value = Derivative(Function("f")(x), x)

    with pytest.raises(ModelInputError, match="commutative"):
        inspect_model(Operators, "operators").solutions()
    with pytest.raises(ModelInputError, match="bound-variable"):
        inspect_model(Unevaluated, "derivative").solutions()
    family = inspect_model(Reading, "scalar").solutions()
    for value in (MatrixSymbol("matrix", 2, 2), Symbol("operator", commutative=False)):
        with pytest.raises(ModelInputError, match="commutative scalar"):
            inspect_model(Reading, "unsupported input", x=value).solutions()
        with pytest.raises(ModelInputError, match="commutative scalar"):
            family.values(value)
        with pytest.raises(ModelInputError, match="commutative scalar"):
            family.observe(x=value)


def test_inputs_use_owned_fields_and_exact_scalars():
    p = inspect_model(Reading, "case")
    family = p.solutions()
    other = inspect_model(Reading, "case").symbols.x
    for operation in (
        lambda: p.solutions(other),
        lambda: family.values(other),
        lambda: family.observe(x=other),
        lambda: family.expression(other),
        lambda: family.where(other > 0),
    ):
        with pytest.raises(ModelValidationError, match="another model problem"):
            operation()
    with pytest.raises(ModelInputError, match="Unknown model attribute"):
        family.observe(missing=2)
    with pytest.raises(ModelInputError, match="scalar"):
        family.observe(x={"x": 2})
    with pytest.raises(ModelInputError, match="exact"):
        inspect_model(Reading, "float", x=0.1).solutions()
    with pytest.raises(ModelInputError, match="exact"):
        family.values(0.1)
    with pytest.raises(ModelInputError, match="exact"):
        family.observe(x=0.1)
    with pytest.raises(ModelInputError, match="at least one"):
        family.values()
    with pytest.raises(ModelInputError, match="field symbols"):
        p.solutions(p.symbols.x + 1)
    with pytest.raises(ModelInputError, match="relations"):
        family.where(p.symbols.x + 1)
    for operation in (family.expression, family.values, family.where):
        with pytest.raises(ModelInputError):
            operation("reading.x")
