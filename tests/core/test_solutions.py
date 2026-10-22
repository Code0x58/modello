"""Family observations, queries, and materialization retain ownership and exactness."""

from dataclasses import FrozenInstanceError, replace
from itertools import combinations

import pytest
from sympy import (
    Derivative,
    FiniteSet,
    Function,
    MatrixSymbol,
    Rational,
    S,
    Symbol,
    Tuple,
    diff,
)

import modello
from examples.projectile import FreeFlight, Projectile
from examples.solution_families import Execution
from examples.solution_views import ModelView, ModelViews
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

from ._models import Check, Reading


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


def test_model_views_follow_dependencies_and_measure_images_without_pointwise_rank():
    problem = inspect_model(Execution, "views", reference=100)
    execution = problem.symbols
    quantity, impact, fee = execution.quantity, execution.impact, execution.fee
    controls = (quantity, impact, fee)
    family = problem.solutions(*controls)
    reference = Execution("reference", quantity=5, reference=100, impact=Rational(1, 2), fee=1)
    views = tuple(ModelViews(family, through=reference).sections())
    assert {tuple(field.symbol for field in view.controls) for view in views} == {
        selected for size in range(1, 4) for selected in combinations(controls, size)
    }
    whole = next(view for view in views if len(view.controls) == 3)
    cost = whole.family.expression(execution.cost)
    assert diff(cost, fee) == 1
    assert diff(cost, impact) == quantity**2
    assert whole.control_dimension == 3
    assert whole.image_dimension(execution.price, execution.cost) == 2
    assert whole.image_dimension(execution.price, execution.price, execution.reference) == 1
    assert whole.image_dimension(execution.reference) == 0
    reading = inspect_model(Reading, "stationary reference")
    x = reading.symbols.x
    bounded = reading.solutions().where(x >= -1, x <= 1)
    view = next(ModelViews(bounded, through=Reading("zero", x=0)).sections())
    assert view.control_dimension == view.image_dimension(reading.symbols.squared) == 1


def test_projectile_envelope_preserves_physical_reachability_and_coupling():
    forward = Projectile("forward", free=FreeFlight("free", speed=20, x=16, y=0, z=32), gravity=10, target_height=0)
    assert forward.free.time_squared == Rational(16, 5)
    assert forward.height == 16
    problem = Projectile.prepare()
    shot = problem.symbols
    x, y, z = shot.x, shot.y, shot.free.z
    height = problem.solutions(x, y, z, shot.target_height).expression(shot.height)
    envelope = Projectile.envelope()
    safety = envelope.symbols
    ceiling = envelope.expression(safety.height)
    assert envelope.expression(safety.free.z) == 40
    assert (ceiling - (20 - safety.x**2 / 80 - safety.y**2 / 80)).expand() == 0
    rebased = ceiling.subs({safety.x: x, safety.y: y})
    assert (rebased - height - (z - 40) ** 2 / 80).expand() == 0
    region = envelope.where(safety.clearance >= 0, safety.target_height >= 0)
    assert tuple(field.symbol for field in region.parameters) == (safety.x, safety.y, safety.target_height)
    points = ((0, 0, 20), (40, 0, 0), (0, 0, 10), (0, 0, 21), (41, 0, 0), (0, 0, -1))
    for point, expected in zip(points, (S.true, S.true, S.true, S.false, S.false, S.false), strict=True):
        assert region.parameter_domain.contains(Tuple(*point)) is expected
    anchor = Projectile("reference", free=FreeFlight("free", speed=20, x=0, y=0, z=40), gravity=10, target_height=0)
    planar = region.section(safety.x, safety.target_height, through=anchor)
    view = ModelView(planar.parameters, planar)
    assert view.control_dimension is None
    assert view.image_dimension(safety.x, safety.target_height) is None
    assert planar.parameter_domain.contains(Tuple(0, 20)) is S.true
    assert planar.parameter_domain.contains(Tuple(40, 1)) is S.false
    slower = Projectile.envelope(speed=10, gravity=10)
    other = slower.symbols
    assert (slower.expression(other.height) - (5 - other.x**2 / 20 - other.y**2 / 20)).expand() == 0


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
    class Operators(Modello):
        x = InstanceDummy("x", commutative=False)

    class Unevaluated(Modello):
        x = InstanceDummy("x", real=True)
        value = Derivative(Function("f")(x), x)

    model = Operators if kind == "operators" else Unevaluated
    with pytest.raises(ModelInputError, match=message):
        inspect_model(model, kind).solutions()


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(MatrixSymbol("matrix", 2, 2), id="matrix"),
        pytest.param(Symbol("operator", commutative=False), id="noncommutative-symbol"),
    ],
)
@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda p, value: inspect_model(p.model, p.name, x=value).solutions(), id="prepare"),
        pytest.param(lambda p, value: p.solutions().values(value), id="values"),
        pytest.param(lambda p, value: p.solutions().observe(x=value), id="observe"),
    ],
)
def test_family_scalar_boundaries_require_commutative_expressions(reading_problem, value, operation):
    with pytest.raises(ModelInputError, match="commutative scalar"):
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
    "operation",
    [
        pytest.param(lambda p: inspect_model(p.model, p.name, x=0.1).solutions(), id="prepare"),
        pytest.param(lambda p: p.solutions().values(0.1), id="values"),
        pytest.param(lambda p: p.solutions().observe(x=0.1), id="observe"),
        pytest.param(lambda p: p.solutions().expression(0.1), id="expression"),
    ],
)
def test_family_operations_require_exact_scalars(reading_problem, operation):
    with pytest.raises(ModelInputError, match="exact"):
        operation(reading_problem)


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
