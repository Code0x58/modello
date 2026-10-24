"""Unsupported problem classes fail explicitly; they are not upstream xfails."""

import pytest
from sympy import Derivative, Matrix, Rational, S, Symbol

from modello import (
    Coordinate,
    Index,
    InstanceDummy,
    Modello,
    Recurrence,
    Series,
    Vector,
    inspect_model,
    model_recurrence,
    model_vector,
)
from modello.contrib.stochastic import ItoProcess
from modello.errors import ModelInputError


@pytest.mark.parametrize("lag", [-2, -1, 0, 2])
def test_horizon_independent_recurrences_require_causal_lags_with_enough_history(lag):
    n = Index("n")
    T = Series("T", n, real=True)
    with pytest.raises(ModelInputError, match="positive lags"):
        Recurrence(T[n], T[n - lag] + 1, initial={0: 1})


def test_a_nonlinear_finite_law_does_not_claim_to_have_a_symbolic_closed_form():
    n = Index("n")
    T = Series("T", n, real=True)
    law = Recurrence(T[n], T[n - 1] ** 2 + 1, initial={0: 1})
    assert law.at(4).doit() == 677
    position = Symbol("position", integer=True, nonnegative=True)
    query = law.at(position)
    assert query.doit() == query

    class Model(Modello):
        steps = InstanceDummy("steps", integer=True, nonnegative=True)
        evolution = law
        terminal = evolution.at(steps)

    with pytest.raises(ModelInputError, match="Evaluate bound-variable"):
        inspect_model(Model, "no-closed-form").solutions()


def test_multiple_laws_for_one_series_require_an_explicit_choice():
    n = Index("n")
    T = Series("T", n, real=True)
    first = Recurrence(T[n], T[n - 1] + 1, initial={0: 0})
    second = Recurrence(T[n], 2 * T[n - 1], initial={0: 1})
    with pytest.raises(ModelInputError, match="multiple recurrence"):

        class Ambiguous(Modello):
            a, b = first, second
            terminal = T[3]

    class Selected(Modello):
        a, b = first, second
        left, right = a.at(3), b.at(3)

    assert (Selected("selected").left, Selected("selected").right) == (3, 8)


@pytest.mark.parametrize("relative", [0, 1])
def test_cross_series_cycles_must_make_progress_towards_an_initial_boundary(relative):
    class Circular(Modello):
        n = Index("n")
        A, B = Series("A", n, real=True), Series("B", n, real=True)
        first = Recurrence(A[n], B[n + relative] + 1, initial={0: 1})
        second = Recurrence(B[n], A[n] + 1, initial={0: 1})
        terminal = A[2]

    with pytest.raises(ModelInputError, match="Cyclic|decrease in time"):
        inspect_model(Circular, "circular").solutions()


@pytest.mark.parametrize("operation", ["mean", "variance", "exact"])
def test_time_dependent_coefficients_do_not_claim_unsupported_exact_moments(operation):
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    law = ItoProcess(x, t, 1, t)
    arguments = (1, 1, 0) if operation == "exact" else (1, 1)
    with pytest.raises(NotImplementedError, match="coefficients"):
        getattr(law, operation)(*arguments).doit()


@pytest.mark.parametrize("invalid", [Matrix([1, 2]), [1, 2]])
def test_discrete_series_are_scalar_families_rather_than_implicit_tensor_processes(invalid):
    n = Index("n")
    T = Series("T", n)
    with pytest.raises((ModelInputError, TypeError)):
        Recurrence(T[n], invalid, initial={0: 1})


def test_series_queries_are_not_finite_python_negative_indices():
    n = Index("n")
    T = Series("T", n, real=True)
    law = Recurrence(T[n], T[n - 1] + 1, initial={0: 1})
    assert law.at(-1).doit() is S.NaN
    assert law.window(0, 2).doit()[-1] == 3


@pytest.mark.parametrize("coordinate", [Rational(1, 2), S.ImaginaryUnit, True])
def test_discrete_queries_reject_noninteger_or_boolean_positions(coordinate):
    n = Index("n")
    T = Series("T", n, real=True)
    with pytest.raises(ModelInputError):
        T[coordinate]


def test_unbound_local_coordinates_are_not_observable_model_parameters():
    class Unbound(Modello):
        n = Index("n")
        values = Vector("x", size=3, real=True)
        sample = values[n]

    with pytest.raises(ModelInputError, match="indices"):
        inspect_model(Unbound, "unbound", values=[1, 2, 3]).solutions()


def test_calculus_does_not_silently_choose_a_branch_through_an_algebraic_cycle():
    class Implicit(Modello):
        x = InstanceDummy("x", real=True)
        y = InstanceDummy("y", real=True)
        x = y**2
        y = x + 1
        slope = Derivative(x, y)

    with pytest.raises(ModelInputError, match="cyclic"):
        inspect_model(Implicit, "implicit").solutions()


def test_model_vectors_need_a_concrete_member_schema():
    class VariableWidth(Modello):
        size = InstanceDummy("size", integer=True, positive=True)
        values = Vector("x", size=size, real=True)

    with pytest.raises(ModelInputError, match="concrete"):

        class Invalid(Modello):
            rows = model_vector(VariableWidth, size=2)


def test_unknown_update_paths_are_definition_errors_at_the_public_boundary():
    class State(Modello):
        value = InstanceDummy("value", real=True)

    with pytest.raises(ModelInputError, match="no scalar field"):

        class Invalid(Modello):
            states = model_vector(State, size=2)
            n = Index("n")
            law = model_recurrence(states[n], typo=states[n - 1].value)


def test_mismatched_dimensions_are_rejected_before_compiling_a_model_window():
    class Window(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        values = Vector("x", size=N, real=True)
        other = Vector("y", size=N + 1, real=True)

    with pytest.raises(ModelInputError, match="length|size|dimension|shape"):
        inspect_model(Window, "mismatched", values=[1, 2], other=[1, 2])


def test_unrelated_abstract_shape_symbols_are_not_assumed_equal():
    N = Symbol("N", integer=True, nonnegative=True)
    M = Symbol("M", integer=True, nonnegative=True)
    with pytest.raises(ModelInputError, match="same size"):
        Vector("x", size=N) + Vector("y", size=M)
