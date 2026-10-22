"""Typed failures retain context without implicitly launching diagnosis."""

import pytest

import modello
import modello.diagnostics as diagnostics
from modello import (
    InstanceDummy,
    inspect_model,
)
from modello.errors import (
    ModelDefinitionError,
    ModelInputError,
    ModelloError,
    ModelSolveError,
    ModelValidationError,
    SolutionCountError,
    SolverBackendError,
)

from ._models import Check, Reading, Sum


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
    with pytest.raises(SolverBackendError) as failed:
        Reading("unsupported", squared=3)
    assert failed.value.backend_error is failed.value.__cause__ is backend_error
    assert failed.value.solution_count is None
    match failed.value:
        case SolverBackendError(backend_error=error):
            assert error is backend_error
        case _:
            pytest.fail("Backend failures must retain the original exception")


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
