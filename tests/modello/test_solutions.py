"""Solution families retain joint relations, domains, ownership, and exactness."""

from dataclasses import FrozenInstanceError, replace

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
)

import modello
import modello.solutions as solutions
from modello import (
    ConstraintKind,
    InstanceDummy,
    ModelInputError,
    Modello,
    ModelProblem,
    ModelValidationError,
    SolutionFamily,
    SolutionSelectionError,
    UnresolvedSolutionError,
    component,
    inspect_model,
)

from ._models import Check, Circle, Operators, Reading, Sum, Wave


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


def test_family_materialisation_preserves_bindings_and_provenance_without_legacy_solver(monkeypatch, check_problem):
    problem = check_problem
    check = problem.symbols
    x, total = check.reading.x, check.total
    family = problem.solutions()
    assert family.format(family.expression(total)) == "reading.x**2 + 1"
    assert family.values(x, total).contains(Tuple(2, 6)) is S.false
    assert family.values(x, total).contains(Tuple(2, 5)) is S.true
    assert family.values(total - check.reading.squared) == FiniteSet(1)
    assert "reading: Reading\n    x [coordinate]\n    squared = x**2" in str(family)

    def forbidden(*args, **kwargs):
        raise AssertionError("A checked family case must not rerun solve()")

    monkeypatch.setattr(modello, "solve", forbidden)
    observed = family.observe(total=5)
    cases = observed.cases()
    first, second = cases
    assert first._modello_bound_fields[Check.total].symbol != second._modello_bound_fields[Check.total].symbol
    assert first._modello_bound_fields[Check.total].kind is ConstraintKind.INPUT
    with pytest.raises(SolutionSelectionError) as caught:
        observed.one()
    assert caught.value.solution_count == 2
    assert caught.value.family is observed
    assert family.parameters == (problem.field("reading", "x"),) and family.parameter_domain is S.Reals
    assert all(c.input is None for c in problem.constraints)
    assert {c.input for c in observed.problem.constraints if c.input is not None} == {"total"}
    monkeypatch.setattr(SolutionFamily, "_prepare", forbidden)
    assert observed.observe() is observed
    assert observed.where() is observed


def test_continuous_and_empty_families_retain_selection_error_context(check_problem):
    p = check_problem
    total = p.symbols.total
    band = p.solutions().where(total >= 4, total <= 6)
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
    reference = Check("reference", reading=Reading("reading", x=2))
    assert band.section(through=reference).one().total == 5
    with pytest.raises(ModelValidationError):
        band.section(through=Check("outside the band", reading=Reading("reading", x=3)))
    reference.total = 6
    with pytest.raises(ModelValidationError):
        band.section(through=reference)


def test_typed_symbol_tree_and_nested_observations_share_constructor_provenance():
    class Joined(Modello):
        left = component(Check)
        right = component(Check)
        alias = left.reading.x

    problem = inspect_model(Joined, "same name")
    symbols = problem.symbols
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
    section = problem.solutions().section(symbols.alias, symbols.left.reading.x, through=case)
    assert tuple(field.symbol for field in section.parameters) == (symbols.left.reading.x,)
    assert section.expression(symbols.left.total) == symbols.left.reading.x**2 + 1


def test_failed_component_observation_leaves_existing_snapshots_unchanged(check_problem):
    """Partially captured inputs cannot leak from a failed immutable refinement."""
    problem = check_problem
    family = problem.solutions()
    original = problem.constraints
    source = Reading("source", x=2)
    source.squared = object()
    with pytest.raises(ModelInputError, match=r"reading\.squared.*scalar"):
        family.observe(total=5, reading=source)
    assert problem.constraints is original
    assert family.problem is problem
    assert family.parameter_domain is S.Reals

    source.squared = S(4)
    observed = family.observe(reading=source).observe(total=5)
    assert tuple(c.index for c in observed.problem.constraints) == tuple(range(len(original) + 3))
    assert [(c.path, c.kind, c.input) for c in observed.problem.constraints[len(original) :]] == [
        (("reading", "x"), ConstraintKind.INPUT, "reading"),
        (("reading", "squared"), ConstraintKind.INFERRED, "reading"),
        (("total",), ConstraintKind.INPUT, "total"),
    ]
    assert observed.one().total == 5
    assert family.problem.constraints is original


def test_materialisation_does_not_accept_an_undecided_candidate_audit(monkeypatch):
    p = inspect_model(Reading, "point", x=2)
    reference = Reading("reference", x=2)
    check = ModelProblem.check

    def undecided(self, candidate):
        return replace(check(self, candidate), verdict="unknown")

    monkeypatch.setattr(ModelProblem, "check", undecided)
    family = p.solutions()
    with pytest.raises(UnresolvedSolutionError) as caught:
        family.one()
    assert caught.value.check.verdict == "unknown"
    assert caught.value.family is family
    with pytest.raises(UnresolvedSolutionError) as section:
        family.section(through=reference)
    assert section.value.check.verdict == "unknown"
    assert section.value.family is family


@pytest.mark.parametrize(
    "kind, message",
    [
        pytest.param("operators", "commutative", id="noncommutative-field"),
        pytest.param("derivative", "bound-variable", id="unevaluated-derivative"),
    ],
)
def test_unsupported_expression_kinds_fail_with_owned_input_errors(kind, message):
    class Unevaluated(Modello):
        x = InstanceDummy("x", real=True)
        value = Derivative(Function("f")(x), x)

    model = Operators if kind == "operators" else Unevaluated
    with pytest.raises(ModelInputError, match=message):
        inspect_model(model, kind).solutions()


@pytest.mark.parametrize(
    "value, message",
    [
        pytest.param(0.1, "exact", id="inexact-float"),
        pytest.param(MatrixSymbol("matrix", 2, 2), "commutative scalar", id="matrix"),
        pytest.param(Symbol("operator", commutative=False), "commutative scalar", id="noncommutative-symbol"),
    ],
)
@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda p, value: inspect_model(p.model, p.name, x=value).solutions(), id="prepare"),
        pytest.param(lambda p, value: p.solutions().values(value), id="values"),
        pytest.param(lambda p, value: p.solutions().observe(x=value), id="observe"),
        pytest.param(lambda p, value: p.solutions().expression(value), id="expression"),
    ],
)
def test_family_scalar_boundaries_require_exact_commutative_expressions(reading_problem, value, message, operation):
    with pytest.raises(ModelInputError, match=message):
        operation(reading_problem, value)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda family, other: family.problem.solutions(other), id="coordinates"),
        pytest.param(lambda family, other: family.values(other), id="values"),
        pytest.param(lambda family, other: family.observe(x=other), id="observe"),
        pytest.param(lambda family, other: family.expression(other), id="expression"),
        pytest.param(lambda family, other: family.where(other > 0), id="where"),
        pytest.param(lambda family, other: family.section(other, through=Reading("reference", x=2)), id="section"),
    ],
)
def test_family_operations_reject_fields_from_another_problem(reading_problem, operation):
    other = inspect_model(Reading, reading_problem.name).symbols.x
    with pytest.raises(ModelValidationError, match="another model problem"):
        operation(reading_problem.solutions(), other)


@pytest.mark.parametrize(
    "operation, message",
    [
        pytest.param(lambda f: f.observe(missing=2), "Unknown model attribute", id="unknown-input"),
        pytest.param(lambda f: f.observe(x={"x": 2}), "scalar", id="mapping-value"),
        pytest.param(lambda f: f.values(), "at least one", id="no-output"),
        pytest.param(lambda f: f.problem.solutions(f.symbols.x + 1), "field symbols", id="coordinate-expression"),
        pytest.param(
            lambda f: f.section(f.symbols.squared, through=Reading("reference", x=2)),
            "remaining coordinates",
            id="dependent-control",
        ),
        pytest.param(
            lambda f: f.section(f.symbols.x + 1, through=Reading("reference", x=2)),
            "remaining coordinate symbols",
            id="control-expression",
        ),
        pytest.param(
            lambda f: f.section(f.symbols.x, through=Check("other type", reading=Reading("reference", x=2))),
            "reference state must be a Reading",
            id="wrong-reference-type",
        ),
        pytest.param(lambda f: f.where(f.symbols.x + 1), "relations", id="scalar-condition"),
    ],
)
def test_family_input_errors_identify_the_boundary(reading_problem, operation, message):
    with pytest.raises(ModelInputError, match=message):
        operation(reading_problem.solutions())


def test_family_sections_identify_a_missing_nested_reference_field(check_problem):
    reference = Check("reference", reading=Reading("reading", x=2))
    reference.reading = object()
    with pytest.raises(ModelInputError, match=r"reference state is missing reading\.x"):
        check_problem.solutions().section(through=reference)


@pytest.mark.parametrize("operation", ["expression", "values", "where"])
def test_family_expressions_use_symbols_instead_of_field_path_strings(reading_problem, operation):
    family = reading_problem.solutions()
    with pytest.raises(ModelInputError):
        getattr(family, operation)("reading.x")
