"""Diagnosis must distinguish proofs, witnessed feasibility, and lack of knowledge."""

import json
from dataclasses import FrozenInstanceError, replace
from itertools import combinations, product

import pytest
from sympy import (
    ConditionSet,
    Dummy,
    Eq,
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
    ConstraintKind,
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


class Reading(Modello):
    x = InstanceDummy("x", real=True)
    squared = x**2


class Check(Modello):
    reading = component(Reading)
    total = reading.squared + 1


class Sum(Modello):
    x = InstanceDummy("x", real=True)
    y = InstanceDummy("y", real=True)
    total = x + y


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


def test_inspection_does_not_reserve_model_field_names():
    class Named(Modello):
        model = InstanceDummy("model", real=True)

    assert Named("named", model=2).model == 2
    p = inspect_model(Named, "named", model=2)
    assert p.constraints[0].path == ("model",)
    with pytest.raises(TypeError, match="requires a Modello class"):
        inspect_model(Named("instance"), "wrong_input")


def test_exact_linear_conflict_certificate_and_single_input_relaxations():
    p = inspect_model(Sum, "bad", x=1, y=2, total=4)
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


def test_small_affine_cases_against_independent_exact_answers():
    for x, y, total in product((-1, 0, 1), repeat=3):
        p = inspect_model(Sum, "grid", x=x, y=y, total=total)
        result = p.analyse()
        assert result.status == ("consistent" if x + y == total else "inconsistent")
        if x + y == total:
            assert_witness(p, result.witness)
        else:
            assert result.conflict.linear_certificate.verify()


def test_every_subset_of_an_affine_conflict_against_linsolve():
    p = inspect_model(Sum, "subsets", x=1, y=2, total=4)
    variables = tuple(field.symbol for field in p.fields)
    for size in range(len(p.constraints) + 1):
        for subset in combinations(p.constraints, size):
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


def test_empty_solver_output_is_unknown_and_external_witness_can_prove_feasibility(monkeypatch):
    monkeypatch.setattr(diagnostics, "solve", lambda *args, **kwargs: [])
    monkeypatch.setattr(diagnostics, "solveset", lambda f, x, domain: ConditionSet(x, Eq(f, 0), domain))
    p = inspect_model(Reading, "unknown", squared=3)
    result = p.analyse(conflicts=True)
    assert result.status == "unknown"
    assert result.conflict is None
    witness = Candidate((FieldValue(p.field("x"), sqrt(3)), FieldValue(p.field("squared"), S(3))))
    assert p.check(witness).verdict == "satisfied"
    assert replace(p, candidates=(witness,)).analyse().status == "consistent"


def test_unknown_deletion_does_not_certify_irreducibility(monkeypatch):
    monkeypatch.setattr(diagnostics, "solve", lambda *args, **kwargs: [])
    monkeypatch.setattr(diagnostics, "solveset", lambda f, x, domain: ConditionSet(x, Eq(f, 0), domain))
    p = inspect_model(Reading, "conflict", x=0, squared=3)
    result = p.analyse(conflicts=True)
    assert result.status == "inconsistent"
    assert result.conflict.irreducible is None
    assert any(check.status == "unknown" for check in result.conflict.removal_checks)
    assert "irreducibility not established" in str(result)


def test_budget_keeps_a_proven_conflict_without_claiming_a_completed_search():
    p = inspect_model(Sum, "budget", x=1, y=2, total=4)
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


def test_copied_values_retain_construction_inputs_in_both_solve_directions():
    direct = inspect_model(Reading, "direct", x=2)
    assert direct.constraints[-1].kind is ConstraintKind.INPUT
    assert str(direct.constraints[-1].kind) == "input"
    assert f"{direct.constraints[-1].kind:>7}" == "  input"
    assert json.dumps(direct.constraints[-1].kind) == '"input"'
    assert "[input 'x']" in str(direct)

    class Positive(Reading):
        x = InstanceDummy("x", positive=True)
        alias = x

    for source, supplied, inferred in (
        (Positive("forward", x=2), "x", "squared"),
        (Positive("backward", squared=4), "squared", "x"),
        (Positive("alias", alias=2), "x", "squared"),
    ):
        p = inspect_model(Check, "copy", reading=source)
        assert {(c.path[-1], c.kind) for c in p.constraints if c.input == "reading"} == {
            (supplied, "input"),
            (inferred, "inferred"),
        }
        assert f"{inferred} = " in str(p)
        assert "[inferred in 'reading']" in str(p)
        assert "[input 'reading']" in str(p)

        # Copying a populated descendant or the whole parent preserves the same
        # distinction; it is not guessed from the field's declaration.
        parent = Check("parent", reading=source)
        descendant = inspect_model(Check, "descendant", reading=parent.reading)
        assert [(c.path, c.kind) for c in descendant.constraints] == [(c.path, c.kind) for c in p.constraints]

        class Envelope(Modello):
            check = component(Check)

        nested = inspect_model(Envelope, "nested", check=parent)
        assert (
            next(c for c in nested.constraints if c.input and c.path == ("check", "reading", supplied)).kind == "input"
        )
        assert (
            next(c for c in nested.constraints if c.input and c.path == ("check", "reading", inferred)).kind
            == "inferred"
        )
        assert next(c for c in nested.constraints if c.path == ("check", "total") and c.input).kind == "inferred"

    class PositiveCheck(Check):
        reading = component(Positive)

    joined = PositiveCheck("joint", total=5)
    copied = inspect_model(Check, "copy_joint", reading=joined.reading)
    assert {c.kind for c in copied.constraints if c.input == "reading"} == {"inferred"}


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


def test_candidate_audits_cover_domains_undefined_values_and_case_identity():
    class Positive(Modello):
        x = InstanceDummy("x", positive=True)
        alias = x

    p = inspect_model(Positive, "candidate")
    assert p.check(Candidate((FieldValue(p.field("x"), S.Zero),))).verdict == "violated"
    assert p.check(Candidate((FieldValue(p.field("x"), S.NaN),))).verdict == "violated"
    assert p.check(Candidate((FieldValue(p.field("x"), Symbol("unknown")),))).verdict == "unknown"
    with pytest.raises(ValueError, match="aliases"):
        p.check(Candidate((FieldValue(p.field("x"), S.One), FieldValue(p.field("alias"), S(2)))))
    candidate = Candidate((FieldValue(p.field("alias"), 1),))
    assert candidate.value(p.field("x")) == 1
    with pytest.raises(KeyError):
        candidate.value(inspect_model(Positive, "other").field("x"))
    with pytest.raises(ValueError, match="aliases"):
        Candidate((FieldValue(p.field("x"), S.One), FieldValue(p.field("alias"), S(2)))).value(p.field("x"))
    with pytest.raises(ValueError, match="another problem"):
        p.check(Candidate((FieldValue(inspect_model(Positive, "candidate").field("x"), S.One),)))
    with pytest.raises(TypeError, match="scalar"):
        FieldValue(p.field("x"), {})
    with pytest.raises(TypeError, match="scalar"):
        FieldValue(p.field("x"), object())

    class Reciprocal(Modello):
        x = InstanceDummy("x", real=True)
        inverse = 1 / x

    pole = inspect_model(Reciprocal, "pole", x=0).analyse()
    assert pole.status == "inconsistent"
    assert any(check.predicate == "defined" and check.actual is False for check in pole.deductions.assumptions)


@pytest.mark.parametrize("backend_error", (NotImplementedError("unsupported solver"), ValueError("solver value error")))
def test_constructor_failure_keeps_candidates_and_backend_cause(monkeypatch, backend_error):
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


def test_cross_layer_equations_share_the_nearest_common_ancestor():
    class Review(Modello):
        check = component(Check)
        peer = component(Reading)
        x = InstanceDummy("x", real=True)

    class Batch(Modello):
        review = component(Review)
        x = InstanceDummy("x", real=True)

    def snapshot(expression):
        source = Reading("source", x=expression)
        check = Check("check", reading=source)
        review = Review("review", check=check)
        return inspect_model(Batch, "batch", review=review)

    for expression, left, right, reference, offset in (
        (Batch.review.check.total - 1, "reading.x", "total - 1", ("review", "check", "total"), -1),
        (Batch.review.x + 1, "check.reading.x", "x + 1", ("review", "x"), 1),
        (Batch.review.peer.x + 1, "check.reading.x", "peer.x + 1", ("review", "peer", "x"), 1),
        (Batch.x + 1, "review.check.reading.x", "x + 1", ("x",), 1),
    ):
        problem = snapshot(expression)
        rendered = str(problem)
        assert f"{left} = {right} [input 'review']" in rendered
        assert "squared = x**2 [definition]" in rendered
        # Rendering changes neither recorded paths nor the expressions they identify.
        copied = next(c for c in problem.constraints if c.path == ("review", "check", "reading", "x"))
        assert copied.right == problem.field(*reference).symbol + offset
        assert problem.format(copied.right) == f"{'.'.join(reference)} {'+' if offset > 0 else '-'} 1"


@pytest.mark.parametrize("replacement", (9, S(9)))
def test_changed_component_results_are_inputs_instead_of_stale_inferences(replacement):
    source = Reading("source", x=2)
    source.squared = replacement
    p = inspect_model(Check, "changed", reading=source)
    copied = next(c for c in p.constraints if c.input and c.path == ("reading", "squared"))
    assert copied.kind == "input"
    assert copied.right == 9
    assert p.analyse().status == "inconsistent"


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
