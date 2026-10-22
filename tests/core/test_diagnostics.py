"""Diagnosis must distinguish proofs, witnessed feasibility, and lack of knowledge."""

from dataclasses import FrozenInstanceError, replace
from itertools import combinations

import pytest
from sympy import (
    ConditionSet,
    Dummy,
    FiniteSet,
    Function,
    ImageSet,
    S,
    Symbol,
    cos,
    expand,
    linsolve,
    pi,
    sin,
    sqrt,
)

import modello
import modello.diagnostics as diagnostics
from modello import (
    InstanceDummy,
    ModelDefinitionError,
    ModelInputError,
    Modello,
    ModelloError,
    ModelSolveError,
    ModelValidationError,
    SolutionCountError,
    SolverBackendError,
    component,
    inspect_model,
)
from modello.diagnostics import Candidate, FieldValue

from ._models import Check, Reading, Sum


def assert_witness(problem, witness, constraints=None):
    """Verify the mathematical witness directly, independently of report verdicts."""
    assert witness is not None
    values = {item.field.symbol: item.value for item in witness.candidate.values}
    for c in problem.constraints if constraints is None else constraints:
        assert expand(c.left.subs(values) - c.right.subs(values)) == 0
    for field in problem.fields:
        assert not values[field.symbol].free_symbols
        assert values[field.symbol].is_number is True
        assert values[field.symbol].is_finite is True
        for predicate, expected in field.symbol.assumptions0.items():
            assert getattr(values[field.symbol], f"is_{predicate}") == expected


def test_snapshot_preserves_aliases_origins_and_component_boundaries(monkeypatch):
    class Joined(Modello):
        left = component(Reading)
        right = component(Reading)
        alias = left.x
        gap = right.squared - left.squared

    def forbidden(*args, **kwargs):
        raise AssertionError("Introspection must not solve")

    source = Reading("input", x=2)
    monkeypatch.setattr(modello, "solve", forbidden)
    monkeypatch.setattr(diagnostics, "solve", forbidden)
    monkeypatch.setattr(diagnostics, "solveset", forbidden)
    p = inspect_model(Joined, "repl", left=source, gap=0)
    field = p.field("left", "x")
    assert {f.path for f in p.aliases(field)} == {("alias",), ("left", "x")}
    assert p.field("right", "x").symbol != field.symbol
    view = p.at("left")
    assert view.model is Reading
    assert {c.path for c in view.owned_constraints} == {("left", "squared"), ("left", "x")}
    assert any(c.path == ("gap",) for c in view.boundary_constraints)
    assert all(field.symbol in c.symbols for c in p.uses(field))
    assert p.at().owned_constraints == p.constraints
    assert "ModelProblem(Joined" in repr(p)
    assert "  left: Reading\n    c1: squared = x**2 [definition]" in str(p)
    assert {c.input for c in p.constraints if c.kind == "input"} == {"left", "gap"}
    with pytest.raises(FrozenInstanceError):
        p.name = "changed"
    with pytest.raises(KeyError):
        p.at("missing")
    with pytest.raises(KeyError):
        p.field("missing")
    with pytest.raises(KeyError):
        p.without("missing")
    with pytest.raises(ValueError, match="another problem"):
        p.uses(inspect_model(Joined, "repl").field("left", "x"))


def test_inherited_refinement_retains_definition_paths():
    class Positive(Reading):
        x = InstanceDummy("x", positive=True)

    class Refined(Check):
        reading = component(Positive)

    p = inspect_model(Refined, "refined", total=1)
    assert p.at("reading").model is Positive
    assert {c.path for c in p.at("reading").owned_constraints} == {("reading", "squared")}
    result = p.analyse(conflicts=True)
    assert result.status == "inconsistent"
    assert result.conflict is not None
    assert result.conflict.irreducible is True


def test_nested_refinement_keeps_both_definitions_of_an_aliased_quantity():
    """An alias and its component equation retain separate origins when nested."""

    class Doubled(Modello):
        x = InstanceDummy("x", real=True)
        total = 2 * x

    class Joined(Modello):
        reading = component(Doubled)
        observed = reading.total
        observed = reading.x + 3

    class Tripled(Doubled):
        total = 3 * Doubled.x

    class Refined(Joined):
        reading = component(Tripled)

    class Root(Modello):
        left = component(Refined)
        right = component(Joined)

    problem = inspect_model(Root, "origins")
    symbols = problem.symbols
    assert [(c.path, c.left, c.right) for c in problem.constraints] == [
        (("left", "observed"), symbols.left.reading.total, symbols.left.reading.x + 3),
        (("left", "reading", "total"), symbols.left.reading.total, 3 * symbols.left.reading.x),
        (("right", "observed"), symbols.right.reading.total, symbols.right.reading.x + 3),
        (("right", "reading", "total"), symbols.right.reading.total, 2 * symbols.right.reading.x),
    ]
    for case in (Root("constructed"), problem.solutions().one()):
        assert isinstance(case.left.reading, Tripled)
        assert (case.left.reading.x, case.left.observed) == (S(3) / 2, S(9) / 2)
        assert (case.right.reading.x, case.right.observed) == (3, 6)


def test_inspection_does_not_reserve_model_field_names():
    class Named(Modello):
        model = InstanceDummy("model", real=True)

    assert Named("named", model=2).model == 2
    p = inspect_model(Named, "named", model=2)
    assert p.constraints[0].path == ("model",)
    with pytest.raises(TypeError, match="requires a Modello class"):
        inspect_model(Named("instance"), "wrong_input")


def test_exact_linear_conflict_certificate_and_single_input_relaxations(conflicting_sum):
    p = conflicting_sum
    result = p.analyse(conflicts=True, relaxations=True)
    assert result.status == "inconsistent"
    assert result.linear.complete
    assert result.linear.augmented_rank > result.linear.rank
    assert result.conflict.irreducible is True
    certificate = result.conflict.linear_certificate
    assert certificate.verify()
    assert not replace(certificate, constant=S.Zero).verify()
    assert not replace(certificate, terms=(), constant=S.Zero).verify()
    assert expand(sum(w * (c.left - c.right) for c, w in certificate.terms)) == certificate.constant != 0
    assert {r.input for r in result.relaxations} == {"x", "y", "total"}
    for relaxation in result.relaxations:
        remaining = tuple(c for c in p.constraints if c not in relaxation.removed)
        assert_witness(p, relaxation.witness, remaining)
    assert "Certificate:" in str(result)


@pytest.mark.parametrize(
    "x, y, total",
    [
        pytest.param(0, 0, 0, id="zero"),
        pytest.param(-1, 1, 0, id="cancellation"),
        pytest.param(-1, -1, -2, id="negative-total"),
        pytest.param(1, 2, 3, id="positive-total"),
        pytest.param(0, 0, 1, id="positive-conflict"),
        pytest.param(0, 0, -1, id="negative-conflict"),
    ],
)
def test_small_affine_cases_against_independent_exact_answers(x, y, total):
    p = inspect_model(Sum, "grid", x=x, y=y, total=total)
    result = p.analyse()
    assert result.status == ("consistent" if x + y == total else "inconsistent")
    if x + y == total:
        assert_witness(p, result.witness)
    else:
        assert result.conflict.linear_certificate.verify()


@pytest.mark.parametrize(
    "indices",
    [
        pytest.param(indices, id="constraints=" + (",".join(map(str, indices)) or "none"))
        for size in range(5)
        for indices in combinations(range(4), size)
    ],
)
def test_every_subset_of_an_affine_conflict_against_linsolve(conflicting_sum, indices):
    p = conflicting_sum
    assert len(p.constraints) == 4
    subset = tuple(p.constraints[index] for index in indices)
    variables = tuple(field.symbol for field in p.fields)
    actual = replace(p, constraints=subset).analyse()
    solutions = linsolve([c.left - c.right for c in subset], variables)
    # An empty equation list has every assignment as a solution.
    expected = bool(solutions) if subset else True
    assert actual.status == ("consistent" if expected else "inconsistent")


def test_affine_nullity_is_not_dimension_after_domain_restrictions():
    class NonnegativeSum(Sum):
        x = InstanceDummy("x", nonnegative=True)
        y = InstanceDummy("y", nonnegative=True)

    p = inspect_model(NonnegativeSum, "boundary", total=0)
    result = p.analyse()
    assert result.linear.nullity == 1
    assert_witness(p, result.witness)
    values = {item.field.path: item.value for item in result.witness.candidate.values}
    assert values[("x",)] == values[("y",)] == 0
    assert "domains excluded" in str(result)


def test_nonlinear_singular_solution_is_not_inferred_from_rank():
    p = inspect_model(Reading, "singular", squared=0)
    result = p.analyse()
    assert result.status == "consistent"
    assert result.linear.nullity == 1
    assert result.linear.complete is False
    assert_witness(p, result.witness)


@pytest.mark.usefixtures("inconclusive_diagnostics")
def test_empty_solver_output_is_unknown_and_external_witness_can_prove_feasibility():
    p = inspect_model(Reading, "unknown", squared=3)
    result = p.analyse(conflicts=True)
    assert result.status == "unknown"
    assert result.conflict is None
    witness = Candidate((FieldValue(p.field("x"), sqrt(3)), FieldValue(p.field("squared"), S(3))))
    assert p.check(witness).verdict == "satisfied"
    assert replace(p, candidates=(witness,)).analyse().status == "consistent"


@pytest.mark.usefixtures("inconclusive_diagnostics")
def test_unknown_deletion_does_not_certify_irreducibility():
    p = inspect_model(Reading, "conflict", x=0, squared=3)
    result = p.analyse(conflicts=True)
    assert result.status == "inconsistent"
    assert result.conflict.irreducible is None
    assert any(check.status == "unknown" for check in result.conflict.removal_checks)
    assert "irreducibility not established" in str(result)


def test_budget_keeps_a_proven_conflict_without_claiming_a_completed_search(conflicting_sum):
    p = conflicting_sum
    result = p.analyse(conflicts=True, relaxations=True, max_checks=1)
    assert result.status == "inconsistent"
    assert result.checks == 1
    assert result.budget_exhausted
    assert result.conflict.irreducible is None
    assert not result.relaxations
    assert {check.status for check in result.input_checks} == {"unknown"}
    assert "check budget exhausted" in str(result)
    with pytest.raises(ValueError, match="at least 1"):
        p.analyse(max_checks=0)


def test_one_core_cannot_supply_repairs_for_two_independent_conflicts():
    class Independent(Modello):
        x = InstanceDummy("x", real=True)
        x_alias = x
        y = InstanceDummy("y", real=True)
        y_alias = y

    p = inspect_model(Independent, "two", x=1, x_alias=2, y=3, y_alias=4)
    result = p.analyse(conflicts=True, relaxations=True)
    assert result.conflict.irreducible is True
    assert len(result.conflict.constraints) == 2
    assert not result.relaxations
    assert {check.status for check in result.input_checks} == {"inconsistent"}
    assert len(p.analyse().conflict.constraints) == 2


def test_component_relaxations_remove_all_copied_values_together():
    p = inspect_model(Check, "grouped", reading=Reading("source", x=2), total=6)
    result = p.analyse(conflicts=True, relaxations=True)
    assert {r.input for r in result.relaxations} == {"reading", "total"}
    source = next(r for r in result.relaxations if r.input == "reading")
    assert {c.path for c in source.removed} == {("reading", "x"), ("reading", "squared")}
    for relaxation in result.relaxations:
        assert_witness(p, relaxation.witness, tuple(c for c in p.constraints if c not in relaxation.removed))
    assert not any(c.input == "reading" for c in p.without("reading").constraints)


def test_parameter_conditions_and_floats_are_not_generic_rank_proofs():
    parameter = Symbol("parameter", real=True)

    class Parameterised(Modello):
        x = InstanceDummy("x", real=True)
        y = parameter * x

    p = inspect_model(Parameterised, "conditional", x=1, y=1)
    result = p.analyse()
    assert result.status == "unknown"
    assert result.linear.complete is False
    assert parameter in p.constraints[0].symbols
    assert inspect_model(Reading, "float", x=1.0, squared=1.0).analyse().status == "unknown"

    coefficient = Function("coefficient")(1)

    class Opaque(Reading):
        product = coefficient * Reading.x

    p = inspect_model(Opaque, "opaque_coefficient", product=0, squared=1)
    report = p.analyse()
    assert report.status == "unknown"
    assert report.solution_sets[0].values.has(ConditionSet)
    # The zero coefficient admits a checked solution, so the opaque system
    # cannot unconditionally be declared inconsistent.
    concrete = replace(
        p, constraints=tuple(replace(c, right=c.right.xreplace({coefficient: S.Zero})) for c in p.constraints)
    )
    assert_witness(concrete, concrete.analyse().witness)


@pytest.mark.parametrize(
    "value, verdict",
    [
        pytest.param(S.One, "satisfied", id="positive"),
        pytest.param(S.Zero, "violated", id="outside-domain"),
        pytest.param(S.NaN, "violated", id="undefined"),
        pytest.param(Symbol("unknown"), "unknown", id="undecided"),
    ],
)
def test_candidate_audits_check_domains_and_definedness(positive_problem, value, verdict):
    candidate = Candidate((FieldValue(positive_problem.field("x"), value),))
    assert positive_problem.check(candidate).verdict == verdict


@pytest.mark.parametrize("operation", ["audit", "lookup"])
def test_candidate_aliases_cannot_have_conflicting_values(positive_problem, operation):
    p = positive_problem
    candidate = Candidate((FieldValue(p.field("x"), 1), FieldValue(p.field("alias"), 2)))
    with pytest.raises(ModelValidationError, match="aliases"):
        if operation == "audit":
            p.check(candidate)
        else:
            candidate.value(p.field("x"))


def test_candidate_lookup_accepts_recorded_aliases(positive_problem):
    candidate = Candidate((FieldValue(positive_problem.field("alias"), 1),))
    assert candidate.value(positive_problem.field("x")) == 1


@pytest.mark.parametrize("operation", ["audit", "lookup"])
def test_candidate_fields_are_owned_by_their_problem(positive_problem, operation):
    p = positive_problem
    other = inspect_model(p.model, p.name)
    candidate = Candidate((FieldValue(p.field("x"), 1),))
    if operation == "lookup":
        with pytest.raises(KeyError):
            candidate.value(other.field("x"))
    else:
        foreign = Candidate((FieldValue(other.field("x"), 1),))
        with pytest.raises(ModelValidationError, match="another problem"):
            p.check(foreign)


@pytest.mark.parametrize("value", [pytest.param({}, id="mapping"), pytest.param(object(), id="opaque-object")])
def test_field_values_require_scalars(positive_problem, value):
    with pytest.raises(ModelInputError, match="scalar"):
        FieldValue(positive_problem.field("x"), value)


def test_a_derived_pole_is_an_undefined_deduction():
    class Reciprocal(Modello):
        x = InstanceDummy("x", real=True)
        inverse = 1 / x

    pole = inspect_model(Reciprocal, "pole", x=0).analyse()
    assert pole.status == "inconsistent"
    assert any(check.predicate == "defined" and check.actual is False for check in pole.deductions.assumptions)


def test_ambiguous_construction_retains_auditable_candidates():
    with pytest.raises(SolutionCountError) as caught:
        Reading("ambiguous", squared=3)
    assert isinstance(caught.value, ModelloError)
    assert isinstance(caught.value, ValueError)
    assert caught.value.solution_count == caught.value.returned_count == 2
    match caught.value:
        case SolutionCountError(2):
            pass
        case _:
            pytest.fail("Candidate counts must be available for structural matching")
    problem = caught.value.problem
    assert len(problem.candidates) == 2
    assert all(problem.check(candidate).verdict == "satisfied" for candidate in problem.candidates)
    assert problem.analyse().status == "consistent"


@pytest.mark.parametrize(
    "backend_error",
    [
        pytest.param(NotImplementedError("unsupported solver"), id="unsupported"),
        pytest.param(ValueError("solver value error"), id="value-error"),
    ],
)
def test_solver_backend_failure_keeps_its_cause(monkeypatch, backend_error):
    def unsupported(*args, **kwargs):
        raise backend_error

    monkeypatch.setattr(modello, "solve", unsupported)
    monkeypatch.setattr(diagnostics, "solve", unsupported)
    monkeypatch.setattr(diagnostics, "solveset", unsupported)
    with pytest.raises(SolverBackendError) as failed:
        Reading("unsupported", squared=3)
    assert failed.value.backend_error is failed.value.__cause__ is backend_error
    assert failed.value.solution_count is None
    match failed.value:
        case SolverBackendError(backend_error=error):
            assert error is backend_error
        case _:
            pytest.fail("Backend failures must retain the original exception")
    analysis = failed.value.problem.analyse()
    assert analysis.status == "unknown"
    assert type(backend_error).__name__ in analysis.reason
    assert type(backend_error).__name__ in analysis.solution_sets[0].solver_errors[0]
    monkeypatch.setattr(diagnostics, "linear_eq_to_matrix", unsupported)
    assert inspect_model(Sum, "unsupported_linear", x=1, y=2, total=3).analyse().status == "unknown"


def test_solution_count_distinguishes_returned_and_domain_rejected_candidates():
    class Restricted(Reading):
        positive = InstanceDummy("positive", positive=True)
        positive = Reading.x - 10

    with pytest.raises(SolutionCountError) as caught:
        Restricted("outside_domain", squared=9)
    assert caught.value.solution_count == 0
    assert caught.value.returned_count == 2
    assert len(caught.value.problem.candidates) == 2


def test_modello_validation_errors_preserve_builtin_categories():
    with pytest.raises(ModelInputError) as caught:
        Check("bad_type", reading={"x": 2})
    assert isinstance(caught.value, ModelloError) and isinstance(caught.value, TypeError)
    with pytest.raises(ModelDefinitionError) as definition:

        class Invalid(Check):
            reading = InstanceDummy("replacement")

    assert isinstance(definition.value, ModelValidationError)
    assert isinstance(definition.value, ModelloError) and isinstance(definition.value, ValueError)
    with pytest.raises(ModelValidationError):
        inspect_model(Reading, "invalid_budget").analyse(max_checks=0)


def test_exception_creation_and_printing_never_launch_analysis(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Only an explicit analyse() call may launch diagnosis")

    monkeypatch.setattr(diagnostics._Analyser, "run", forbidden)
    with pytest.raises(ModelSolveError) as caught:
        Sum("plain_failure", x=1, y=2, total=4)
    assert "0 solutions reported" in str(caught.value)
    assert "total = x + y [definition]" in str(caught.value.problem)


def test_unevaluated_function_values_are_not_numeric_witnesses():
    class Scalar(Modello):
        x = InstanceDummy("x")

    p = inspect_model(Scalar, "formal", x=Function("f")(1))
    assert p.analyse().status == "unknown"


def test_noncommutative_algebra_remains_inspectable_without_scalar_proofs():
    class Operators(Modello):
        x = InstanceDummy("x", commutative=False)
        y = InstanceDummy("y", commutative=False)
        commutator = x * y - y * x

    p = inspect_model(Operators, "operators")
    assert "x*y - y*x" in str(p)
    assert p.analyse().status == "unknown"


def test_solution_sets_preserve_infinite_families_and_unresolved_conditions():
    class Periodic(Modello):
        angle = InstanceDummy("angle", real=True)
        sine = sin(angle)

    p = inspect_model(Periodic, "periodic", sine=0)
    report = p.analyse()
    (result,) = report.solution_sets
    assert result.field == p.field("angle")
    assert result.domain is S.Reals
    assert result.values.has(ImageSet)
    assert result.values.contains(7 * pi) is S.true
    assert result.values.contains(pi / 2) is S.false
    assert result.constraints == (p.constraints[0],)
    assert result.substitutions == (FieldValue(p.field("sine"), 0),)
    assert_witness(p, report.witness)
    assert "Univariate solution sets (necessary conditions):" in str(report)

    class Transcendental(Modello):
        x = InstanceDummy("x", real=True)
        residual = cos(x) - x

    class Envelope(Modello):
        reading = component(Transcendental)
        residual = reading.residual

    report = inspect_model(Envelope, "unresolved", residual=0).analyse()
    assert report.solution_sets[0].values.has(ConditionSet)
    assert report.status == "unknown"
    assert "reading: Transcendental\n      x in ConditionSet(x," in str(report)


@pytest.mark.parametrize(
    ("assumptions", "expected"),
    [
        ({"real": True}, FiniteSet(-sqrt(3), sqrt(3))),
        ({"positive": True}, FiniteSet(sqrt(3))),
        ({"nonnegative": True}, FiniteSet(sqrt(3))),
        ({"negative": True}, FiniteSet(-sqrt(3))),
        ({"nonpositive": True}, FiniteSet(-sqrt(3))),
        ({"integer": True}, S.EmptySet),
        ({"rational": True}, S.EmptySet),
        ({"complex": True}, FiniteSet(-sqrt(3), sqrt(3))),
    ],
)
def test_solution_set_domains_and_finite_witnesses(assumptions, expected):
    class DomainReading(Reading):
        x = InstanceDummy("x", **assumptions)

    p = inspect_model(DomainReading, "domain", squared=3)
    report = p.analyse()
    assert report.solution_sets[0].values == expected
    assert report.status == ("inconsistent" if expected is S.EmptySet else "consistent")
    if report.witness is not None:
        assert_witness(p, report.witness)


def test_nonempty_solution_set_is_not_a_whole_model_witness():
    class Prime(Reading):
        x = InstanceDummy("x", prime=True)

    report = inspect_model(Prime, "extra_predicate", squared=16).analyse()
    assert report.solution_sets[0].values == FiniteSet(4)
    assert report.status == "unknown"
    assert report.witness is None
    assert all(check.verdict == "violated" for check in report.candidates)


def test_univariate_sets_intersect_all_applicable_equations():
    class Powers(Reading):
        cubed = Reading.x**3

    p = inspect_model(Powers, "incompatible_roots", squared=4, cubed=3)
    report = p.analyse()
    assert report.solution_sets[0].values is S.EmptySet
    assert report.solution_sets[0].constraints == p.constraints[:2]
    assert report.status == "inconsistent"


def test_nested_reports_show_component_types_and_equation_origins():
    class Envelope(Modello):
        check = component(Check)
        total = check.total + 1

    p = inspect_model(Envelope, "nested", total=0)
    assert "\n  check: Check\n    c1: total = reading.squared + 1" in str(p)
    assert "\n    reading: Reading\n      c2: squared = x**2" in str(p)
    report = p.analyse(conflicts=True)
    assert report.status == "inconsistent"
    rendered = str(report)
    assert "\n    check: Check\n      c1: total = reading.squared + 1" in rendered
    assert "\n      reading: Reading\n        c2: squared = x**2" in rendered
    assert "reading: Reading" in rendered


@pytest.mark.parametrize(
    "reference, left, right, offset",
    [
        pytest.param(("review", "check", "total"), "reading.x", "total - 1", -1, id="same-component"),
        pytest.param(("review", "x"), "check.reading.x", "x + 1", 1, id="parent-component"),
        pytest.param(("review", "peer", "x"), "check.reading.x", "peer.x + 1", 1, id="sibling-component"),
        pytest.param(("x",), "review.check.reading.x", "x + 1", 1, id="root"),
    ],
)
def test_cross_layer_equations_share_the_nearest_common_ancestor(reference, left, right, offset):
    class Review(Modello):
        check = component(Check)
        peer = component(Reading)
        x = InstanceDummy("x", real=True)

    class Batch(Modello):
        review = component(Review)
        x = InstanceDummy("x", real=True)

    expression = Batch
    for part in reference:
        expression = getattr(expression, part)
    source = Reading("source", x=expression + offset)
    check = Check("check", reading=source)
    review = Review("review", check=check)
    problem = inspect_model(Batch, "batch", review=review)
    rendered = str(problem)
    assert f"{left} = {right} [input 'review']" in rendered
    assert "squared = x**2 [definition]" in rendered
    # Rendering changes neither recorded paths nor the expressions they identify.
    copied = next(c for c in problem.constraints if c.path == ("review", "check", "reading", "x"))
    assert copied.right == problem.field(*reference).symbol + offset
    assert problem.format(copied.right) == f"{'.'.join(reference)} {'+' if offset > 0 else '-'} 1"


def test_bare_field_paths_cannot_be_confused_with_external_symbol_identities():
    positive = Symbol("x", positive=True)
    negative = Symbol("x", negative=True)
    independent = Dummy("x", real=True)
    dotted = Symbol("reading.x", real=True)

    class Names(Modello):
        x = InstanceDummy("x", real=True)
        reading = component(Reading)
        total = x + positive - negative + independent + dotted

    p = inspect_model(Names, "names")
    assert p.format(p.field("x").symbol) == "x"
    assert p.format(p.field("reading", "x").symbol) == "reading.x"
    assert p.format(dotted) == "external[reading.x]"
    labels = {p.format(symbol) for symbol in (positive, negative, independent)}
    assert labels == {"external[x#1]", "external[x#2]", "external[x#3]"}
    rendered = str(p)
    assert all(label in rendered for label in labels)
    assert "self." not in rendered
    assert p.constraints[0].right == p.field("x").symbol + positive - negative + independent + dotted
