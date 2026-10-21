"""Diagnosis must distinguish proofs, witnessed feasibility, and lack of knowledge."""

from dataclasses import FrozenInstanceError, replace
from itertools import combinations, product

import pytest
from sympy import Function, S, Symbol, expand, linsolve, sqrt

import modello
import modello.diagnostics as diagnostics
from modello import InstanceDummy, Modello, ModelSolveError, component, inspect_model
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
    assert "definition left.squared" in str(p)
    assert {c.input for c in p.constraints if c.kind == "observation"} == {"left", "gap"}
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
    p = inspect_model(Reading, "unknown", squared=3)
    result = p.analyse(conflicts=True)
    assert result.status == "unknown"
    assert result.conflict is None
    witness = Candidate((FieldValue(p.field("x"), sqrt(3)), FieldValue(p.field("squared"), S(3))))
    assert p.check(witness).verdict == "satisfied"
    assert replace(p, candidates=(witness,)).analyse().status == "consistent"


def test_unknown_deletion_does_not_certify_irreducibility(monkeypatch):
    monkeypatch.setattr(diagnostics, "solve", lambda *args, **kwargs: [])
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


def test_constructor_failure_keeps_candidates_and_backend_cause(monkeypatch):
    with pytest.raises(ModelSolveError) as caught:
        Reading("ambiguous", squared=3)
    problem = caught.value.problem
    assert len(problem.candidates) == 2
    assert all(problem.check(candidate).verdict == "satisfied" for candidate in problem.candidates)
    assert problem.analyse().status == "consistent"

    def unsupported(*args, **kwargs):
        raise NotImplementedError("example unsupported solver")

    monkeypatch.setattr(modello, "solve", unsupported)
    monkeypatch.setattr(diagnostics, "solve", unsupported)
    with pytest.raises(ModelSolveError) as caught:
        Reading("unsupported", squared=3)
    assert isinstance(caught.value.__cause__, NotImplementedError)
    assert caught.value.solution_count is None
    analysis = caught.value.problem.analyse()
    assert analysis.status == "unknown"
    assert "NotImplementedError" in analysis.reason
    monkeypatch.setattr(diagnostics, "linear_eq_to_matrix", unsupported)
    assert inspect_model(Sum, "unsupported_linear", x=1, y=2, total=3).analyse().status == "unknown"


def test_exception_creation_and_printing_never_launch_analysis(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Only an explicit analyse() call may launch diagnosis")

    monkeypatch.setattr(diagnostics._Analyser, "run", forbidden)
    with pytest.raises(ModelSolveError) as caught:
        Sum("plain_failure", x=1, y=2, total=4)
    assert "0 solutions reported" in str(caught.value)
    assert "definition total" in str(caught.value.problem)


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
    assert "self.x*self.y - self.y*self.x" in str(p)
    assert p.analyse().status == "unknown"
