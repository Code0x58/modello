"""State equations do not change meaning when users regroup model components."""

import pytest
from sympy import Rational, Sum

from modello import Index, InstanceDummy, Modello, ModelObservations, Vector, component, model_recurrence, model_vector
from modello.errors import ModelInputError


class Scalar(Modello):
    value = InstanceDummy("value", real=True)


class Pair(Modello):
    left = component(Scalar)
    right = component(Scalar)


@pytest.mark.parametrize("grouped", [False, True], ids=["component-laws", "whole-model-law"])
@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "terminal-inverse"])
def test_current_step_dependencies_are_invariant_under_equation_grouping(solve_model, grouped, inverse):
    class Path(Modello):
        states = model_vector(Pair, size=3)
        opening = states[0]
        terminal = states[-1]
        n = Index("n")
        if grouped:
            law = model_recurrence(
                states[n],
                **{
                    "left.value": states[n - 1].left.value + 1,
                    "right.value": 2 * states[n].left.value,
                },
            )
        else:
            left_law = model_recurrence(states[n].left, value=states[n - 1].left.value + 1)
            right_law = model_recurrence(states[n].right, value=2 * states[n].left.value)

    observations = (
        {
            "opening": Pair("seed", right=Scalar("right", value=0)),
            "terminal": Pair("target", right=Scalar("right", value=6)),
        }
        if inverse
        else {"opening": Pair("seed", left=Scalar("left", value=1), right=Scalar("right", value=0))}
    )
    case = solve_model(Path, **observations)
    assert [(state.left.value, state.right.value) for state in case.states] == [(1, 0), (2, 4), (3, 6)]
    assert case.opening is case.states[0]
    assert case.terminal is case.states[-1]


def test_model_valued_implicit_euler_uses_the_existing_joint_solver(solve_model):
    class State(Modello):
        value = InstanceDummy("value", positive=True)

    class Decay(Modello):
        step = InstanceDummy("step", positive=True)
        rate = InstanceDummy("rate", nonnegative=True)
        states = model_vector(State, size=4)
        n = Index("n")
        law = model_recurrence(states[n], value=states[n - 1].value - step * rate * states[n].value)

    case = solve_model(Decay, step=Rational(1, 2), rate=1, states=ModelObservations({0: State("seed", value=1)}))
    assert tuple(state.value for state in case.states) == (1, Rational(2, 3), Rational(4, 9), Rational(8, 27))


@pytest.mark.parametrize("count", [1, 2, 5])
def test_explicit_terminal_references_support_a_boundary_value_model(solve_model, count):
    class Bridge(Modello):
        N = InstanceDummy("N", integer=True, positive=True)
        states = model_vector(Scalar, size=N + 1)
        n = Index("n")
        law = model_recurrence(states[n], value=states[n - 1].value + (states[-1].value - states[0].value) / N)

    case = solve_model(
        Bridge, N=count, states=ModelObservations({0: Scalar("start", value=1), -1: Scalar("end", value=7)})
    )
    assert tuple(state.value for state in case.states) == tuple(1 + Rational(6 * i, count) for i in range(count + 1))


@pytest.mark.parametrize("rewritten", [False, True], ids=["offset-in-summand", "offset-in-limits"])
def test_a_finite_sliding_input_sum_is_independent_of_index_notation(solve_model, rewritten):
    class Filter(Modello):
        inputs = Vector("u", size=4, real=True)
        states = model_vector(Scalar, size=5)
        n, j = Index("n"), Index("j")
        law = model_recurrence(
            states[n],
            start=2,
            value=states[n - 1].value
            + (Sum(inputs[j], (j, n - 2, n - 1)) if rewritten else Sum(inputs[n - 1 - j], (j, 0, 1))),
        )

    case = solve_model(
        Filter, inputs=[1, 2, 4, 8], states=ModelObservations({0: Scalar("zero", value=0), 1: Scalar("one", value=1)})
    )
    assert tuple(state.value for state in case.states) == (0, 1, 4, 10, 22)


def test_nested_scalar_vectors_keep_their_model_members_and_can_be_updated_by_path(solve_model):
    class State(Modello):
        samples = Vector("x", size=2, real=True)
        holding = component(Scalar)

    class Path(Modello):
        states = model_vector(State, size=3)
        n = Index("n")
        law = model_recurrence(
            states[n],
            **{
                "samples[0]": states[n - 1].samples[0] + 1,
                "samples[1]": 2 * states[n - 1].samples[1],
                "holding.value": states[n - 1].holding.value + 3,
            },
        )

    seed = State("seed", samples=[1, 2], holding=Scalar("holding", value=5))
    case = solve_model(Path, states=ModelObservations({0: seed}))
    assert case.states[-1].samples == Vector(3, 8)
    assert case.states[-1].holding.value == 11
    assert isinstance(case.states[-1].holding, Scalar)
    assert case.states[0] is not seed


@pytest.mark.parametrize("index", [-10, 10])
def test_explicit_out_of_window_model_references_are_rejected(solve_model, index):
    class Invalid(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        states = model_vector(Scalar, size=N + 1)
        n = Index("n")
        law = model_recurrence(states[n], value=states[index].value)

    with pytest.raises(ModelInputError):
        solve_model(Invalid, N=2)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_zero_history_equations_can_be_implicit_at_every_position(solve_model, count):
    class Static(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        inputs = Vector("u", size=N, real=True)
        states = model_vector(Scalar, size=N)
        n = Index("n")
        law = model_recurrence(states[n], start=0, value=2 * states[n].value + inputs[n])

    entries = list(range(1, count + 1))
    case = solve_model(Static, inputs=entries)
    assert case.N == count
    assert tuple(state.value for state in case.states) == tuple(-value for value in entries)


@pytest.mark.parametrize("observe_initial", [False, True], ids=["terminal-boundary", "initial-boundary"])
def test_disjoint_laws_over_one_state_vector_share_parameters_and_nested_views(solve_model, observe_initial):
    class State(Modello):
        quantity = InstanceDummy("quantity", real=True)
        cash = InstanceDummy("cash", real=True)

    class Account(Modello):
        rate = InstanceDummy("rate", real=True)
        states = model_vector(State, size=4)
        opening, terminal = states[0], states[-1]
        n = Index("n")
        inventory = model_recurrence(states[n], quantity=states[n - 1].quantity + 2)
        cashflow = model_recurrence(states[n], cash=states[n - 1].cash - rate * states[n].quantity)

    boundary = (
        {"opening": State("opening", quantity=1, cash=100)}
        if observe_initial
        else {"terminal": State("terminal", quantity=7, cash=55)}
    )
    case = solve_model(Account, rate=3, **boundary)
    assert [(state.quantity, state.cash) for state in case.states] == [(1, 100), (3, 91), (5, 76), (7, 55)]
    assert case.opening is case.states[0]
    assert case.terminal is case.states[-1]


@pytest.mark.parametrize("explicit", [False, True], ids=["series-query", "explicit-law-query"])
def test_scalar_series_can_drive_model_valued_equations_with_the_same_relative_coordinate(solve_model, explicit):
    from modello import Recurrence, Series

    class Driven(Modello):
        n = Index("n")
        T = Series("T", n, real=True)
        forcing = Recurrence(T[n], T[n - 1] + 1, initial={0: 0})
        states = model_vector(Scalar, size=4)
        law = model_recurrence(states[n], value=states[n - 1].value + (forcing.at(n) if explicit else T[n]))

    case = solve_model(Driven, states=ModelObservations({0: Scalar("opening", value=1)}))
    assert tuple(state.value for state in case.states) == (1, 2, 4, 7)
