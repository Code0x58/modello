"""Indexed model equations use ordinary fields and a finite history window."""

from copy import deepcopy

import pytest
from sympy import MatrixSymbol, Sum

from modello import (
    Index,
    InstanceDummy,
    Modello,
    ModelObservations,
    ModelRecurrence,
    Vector,
    inspect_model,
    model_recurrence,
    model_vector,
)
from modello.errors import ModelDefinitionError, ModelInputError, SolutionCountError


class State(Modello):
    value = InstanceDummy("value", real=True)
    twice = 2 * value


class Event(Modello):
    amount = InstanceDummy("amount", real=True)


class Walk(Modello):
    N = InstanceDummy("N", integer=True, nonnegative=True)
    states = model_vector(State, size=N + 1)
    events = model_vector(Event, size=N)
    n = Index("n")
    law = model_recurrence(states[n], value=states[n - 1].value + events[n - 1].amount)
    terminal = states[-1]


def test_indexed_models_solve_forward_and_inverse_without_transition_components():
    observed = ModelObservations({0: State("initial", value=1)})
    forward = Walk("forward", states=observed, events=[Event("a", amount=2), Event("b", amount=4)])
    assert forward.N == 2
    assert forward.states.project(lambda state: state.value) == Vector(1, 3, 7)
    assert forward.terminal.twice == 14
    assert forward.terminal is forward.states[-1]
    assert not hasattr(forward.states, "steps")
    inverse = Walk(
        "inverse",
        events=[Event("a", amount=2), None],
        states=ModelObservations({0: State("initial", value=1), -1: State("target", value=7)}),
    )
    assert inverse.events[1].amount == 4
    assert inverse.states[1].twice == 6
    assert Walk("empty", states=observed, events=[]).terminal.value == 1


def test_source_law_is_scoped_and_reconstructible_while_finite_equations_use_member_paths():
    first = inspect_model(Walk, "same", N=2)
    second = inspect_model(Walk, "same", N=2)
    law = first.symbols.law
    assert isinstance(law, ModelRecurrence)
    assert law.func(*law.args) == deepcopy(law) == law
    assert law.start == 1
    assert law.index not in law.free_symbols
    assert law != second.symbols.law
    assert {c.path for c in first.constraints if c.path[0] == "states"} >= {
        ("states", "1", "value"),
        ("states", "2", "value"),
        ("states", "0", "twice"),
        ("states", "2", "twice"),
    }
    observed = first.solutions().observe(
        states=ModelObservations({0: State("initial", value=1)}),
        events=[Event("a", amount=2), Event("b", amount=4)],
    )
    assert observed.one().terminal.value == 7


def test_refinement_rebinds_new_laws_and_parameters_before_compilation():
    class Base(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rate = InstanceDummy("rate", real=True)
        states = model_vector(State, size=N + 1)
        n = Index("n")
        law = model_recurrence(states[n], value=states[n - 1].value + rate)

    class Refined(Base):
        N = InstanceDummy("N", integer=True, positive=True)
        rate = InstanceDummy("rate", positive=True)
        law = model_recurrence(Base.states[Base.n], value=Base.states[Base.n - 1].value + Base.rate)

    case = Refined("refined", N=2, rate=2, states=ModelObservations({0: State("initial", value=1)}))
    assert case.states[-1].value == 5
    assert not case.states[-1].value.free_symbols
    with pytest.raises(ModelDefinitionError, match="model_recurrence"):

        class Invalid(Base):
            law = Base.law.subs(Base.rate, 2)

    with pytest.raises(ModelDefinitionError, match="model_recurrence"):

        class Detached(Modello):
            law = Base.law


def test_second_order_history_is_ordinary_model_members():
    class Fibonacci(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        states = model_vector(State, size=N + 1)
        n = Index("n")
        law = model_recurrence(states[n], start=2, value=states[n - 1].value + states[n - 2].value)

    initial = ModelObservations({0: State("zero", value=0), 1: State("one", value=1)})
    case = Fibonacci("fib", N=6, states=initial)
    assert case.states.project(lambda state: state.value) == Vector(0, 1, 1, 2, 3, 5, 8)
    with pytest.raises(ModelInputError, match="history"):
        Fibonacci("too_short", N=0)


def test_redeclared_model_windows_keep_the_inherited_symbolic_shape():
    class Refined(Walk):
        N = InstanceDummy("N", integer=True, positive=True)
        states = model_vector(State, size=Walk.N + 1)

    case = Refined(
        "refined",
        N=2,
        states=ModelObservations({0: State("initial", value=1)}),
        events=[Event("event", amount=2), None],
    )
    assert case.states[1].value == 3
    assert case.states[-1].value == case.events[1].amount + 3

    with pytest.raises(ModelDefinitionError, match="Cannot change the size"):

        class ChangedWindow(Walk):
            states = model_vector(State, size=2)


def test_indexed_field_laws_require_a_declared_model_vector():
    from tests.modello.test_model_vectors import Path

    with pytest.raises(ModelDefinitionError, match="state window with model_vector"):

        class Invalid(Path):
            n = Index("n")
            law = model_recurrence(Path.states[n], balance=Path.states[n - 1].balance - 1)


@pytest.mark.parametrize("lag", [0, -1, 2])
def test_noncausal_or_insufficient_history_is_rejected_before_index_wrapping(lag):
    with pytest.raises(ModelDefinitionError, match="positive lags"):

        class Invalid(Modello):
            N = InstanceDummy("N", integer=True, nonnegative=True)
            states = model_vector(State, size=N + 1)
            n = Index("n")
            law = model_recurrence(states[n], value=states[n - lag].value)


def test_relative_input_indices_cannot_wrap_to_the_end_of_a_data_vector():
    class BadInput(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        events = model_vector(Event, size=N)
        states = model_vector(State, size=N + 1)
        n = Index("n")
        law = model_recurrence(states[n], value=states[n - 1].value + events[n - 2].amount)

    with pytest.raises(ModelInputError, match="outside"):
        BadInput("bad", N=2)


def test_other_declarations_cannot_capture_same_named_target_members():
    class Other(Modello):
        states = model_vector(State, size=3)

    with pytest.raises(ModelDefinitionError, match="own declared"):

        class Invalid(Modello):
            states = model_vector(State, size=3)
            n = Index("n")
            law = model_recurrence(Other.states[n], value=Other.states[n - 1].value + 1)

    with pytest.raises(ModelDefinitionError, match="own declared"):

        class MissingTarget(Modello):
            n = Index("n")
            law = model_recurrence(Other.states[n], value=Other.states[n - 1].value + 1)


def test_model_field_updates_reject_matrix_values():
    with pytest.raises(ModelInputError, match="scalar"):

        class Invalid(Modello):
            states = model_vector(State, size=2)
            n = Index("n")
            law = model_recurrence(states[n], value=MatrixSymbol("A", 2, 2))


def test_nested_reductions_keep_distinct_local_indices():
    class Weighted(Modello):
        states = model_vector(State, size=3)
        weights = Vector("w", size=2, real=True)
        n, j = Index("n"), Index("j")
        law = model_recurrence(states[n], value=states[n - 1].value + Sum(weights[j], (j, 0, 1)))

    case = Weighted("weighted", weights=[1, 2], states=ModelObservations({0: State("initial", value=1)}))
    assert case.states[-1].value == 7


def test_sparse_observations_are_typed_immutable_and_do_not_determine_window_size():
    supplied = {0: State("initial", value=1)}
    observed = ModelObservations(supplied)
    supplied.clear()
    assert len(observed) == 1
    with pytest.raises(TypeError):
        observed.records[0] = State("changed")
    with pytest.raises(ModelInputError):
        Walk("undetermined", states=observed)
    with pytest.raises(ModelInputError):
        ModelObservations({True: State("bad")})
    with pytest.raises(ModelInputError, match="outside"):
        Walk("outside", N=2, states=ModelObservations({3: State("bad")}))


def test_member_equations_remain_constraints_alongside_recurrent_fields():
    class Overconstrained(Modello):
        states = model_vector(State, size=2)
        n = Index("n")
        law = model_recurrence(states[n], value=states[n - 1].value + 1, twice=0)

    with pytest.raises(SolutionCountError):
        Overconstrained("conflict", states=ModelObservations({0: State("initial", value=1)}))
