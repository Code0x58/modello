"""Finite causal equations agree across query laws and joint model sequences."""

import pytest
from sympy import Abs, Derivative, Eq, FiniteSet, Limit, Piecewise, Product, S, Subs, Sum, Symbol, Tuple

from modello import (
    Index,
    InstanceDummy,
    Modello,
    Recurrence,
    Series,
    Vector,
    inspect_model,
    model_recurrence,
    model_vector,
    vector,
)
from modello.errors import ModelInputError


@pytest.fixture(params=["series", "models"])
def sequence_model(request):
    def declare(step, *, domain=None, count=3):
        requirements = {"real": True} if domain is None else domain

        if request.param == "series":

            class Sequence(Modello):
                opening = InstanceDummy("opening", **requirements)
                a = InstanceDummy("a", real=True)
                inputs = Vector(2, 3, 5)
                n = Index("n")
                T = Series("T", n, **requirements)
                law = Recurrence(T[n], step(lambda k, data=T: data[k], n, a, inputs), initial={0: opening})
                levels = law.window(0, count)
                closing = law.at(count)
        else:

            class State(Modello):
                value = InstanceDummy("value", **requirements)

            class Sequence(Modello):
                a = InstanceDummy("a", real=True)
                inputs = Vector(2, 3, 5)
                states = model_vector(State, size=count + 1)
                n = Index("n")
                law = model_recurrence(states[n], value=step(lambda k, data=states: data[k].value, n, a, inputs))
                opening = states[0].value
                levels = states.project(lambda state: state.value)
                closing = states[-1].value

        return Sequence

    return declare


@pytest.mark.parametrize(
    "step,observations,expected",
    [
        (lambda T, n, a, u: a * T(n - 1) + 1, {"opening": 1, "a": 2}, (1, 3, 7, 15)),
        (lambda T, n, a, u: a * T(n - 1) + 1, {"closing": 15, "a": 2}, (1, 3, 7, 15)),
        (lambda T, n, a, u: a * T(n - 1) + 1, {"opening": 2, "a": 0}, (2, 1, 1, 1)),
        (lambda T, n, a, u: a * T(n - 1) + 1, {"opening": 2, "a": 1}, (2, 3, 4, 5)),
        (lambda T, n, a, u: T(n - 1) ** 2 + a, {"opening": 2, "a": 1}, (2, 5, 26, 677)),
        (
            lambda T, n, a, u: Piecewise((T(n - 1) + a, Eq(n, 1)), (2 * T(n - 1), True)),
            {"opening": 1, "a": 2},
            (1, 3, 6, 12),
        ),
        (lambda T, n, a, u: T(n - 1) + u[n - 1] + Limit(n**2, n, 0), {"opening": 0, "a": 1}, (0, 2, 5, 10)),
    ],
    ids=[
        "affine-forward",
        "affine-inverse",
        "zero-coefficient",
        "unit-coefficient",
        "nonlinear",
        "conditional",
        "native-limit-scope",
    ],
)
def test_finite_causal_values_agree(sequence_model, solve_model, step, observations, expected):
    case = solve_model(sequence_model(step), **observations)
    assert tuple(case.levels) == expected
    assert case.closing == expected[-1]


def test_nonlinear_inverse_queries_keep_both_joint_alternatives(sequence_model):
    model = sequence_model(lambda T, n, a, u: T(n - 1) ** 2 + a)
    family = inspect_model(model, "inverse", closing=677, a=1).solutions()
    assert family.values(family.symbols.levels) == FiniteSet(Tuple(-2, 5, 26, 677), Tuple(2, 5, 26, 677))


@pytest.mark.parametrize(
    "domain,step,observations",
    [
        ({"positive": True}, lambda T, n, a, u: -T(n - 1), {"opening": 1, "a": 1}),
        ({"integer": True}, lambda T, n, a, u: T(n - 1) / 2, {"opening": 4, "a": 1}),
        ({"real": True}, lambda T, n, a, u: T(n - 1) / a, {"opening": 0, "a": 0}),
        ({"real": True}, lambda T, n, a, u: (T(n - 1) ** 2 - 1) / (T(n - 1) - 1), {"opening": 1, "a": 1}),
    ],
    ids=["positive-intermediate", "integer-intermediate", "zero-denominator", "rational-hole"],
)
def test_finite_paths_preserve_the_same_domains(sequence_model, domain, step, observations):
    family = inspect_model(sequence_model(step, domain=domain), "invalid", **observations).solutions()
    assert family.is_empty is True


def test_positive_absolute_updates_keep_the_same_domain(sequence_model, solve_model):
    case = solve_model(sequence_model(lambda T, n, a, u: Abs(T(n - 1)) + a, domain={"positive": True}), opening=2, a=1)
    assert tuple(case.levels) == (2, 3, 4, 5)


def _moving_sum(T, n, a, u):
    j = Index("j")
    return T(n - 1) + Sum(u[j], (j, n - 2, n - 1))


def _variable_width_sum(T, n, a, u):
    j = Index("j")
    return T(n - 1) + Sum(u[j], (j, n - 2, 1))


def _nested_sum(T, n, a, u):
    j, k = Index("j"), Index("k")
    return T(n - 1) + Sum(u[k], (k, j - 1, j), (j, 0, n - 1))


def _moving_product(T, n, a, u):
    j = Index("j")
    return T(n - 1) + Product(u[j], (j, n - 2, n - 1))


def _moving_comprehension(T, n, a, u):
    j = Index("j")
    return T(n - 1) + vector(u[j], (j, n - 2, n - 1)).sum()


def _relative_substitution(T, n, a, u):
    j = Symbol("j")
    return T(n - 1) + Subs(u[j], j, n - 2)


def _compound_substitution(T, n, a, u):
    j, k = Index("j"), Index("k")
    return T(n - 1) + Subs(u[j + k], j + k, n - 2)


def _nested_substitution(T, n, a, u):
    j, k = Symbol("j"), Symbol("k")
    return T(n - 1) + Subs(Subs(u[j], j, k), k, n - 2)


def _compound_chain(T, n, a, u):
    j, k = Symbol("j"), Symbol("k")
    return T(n - 1) + Subs(u[j], (j, j + k), (j + k, n - 2))


def _substituted_body_pole(T, n, a, u):
    j = Symbol("j")
    return T(n - 1) + Subs((j**2 - 1) / (j - 1), j, n)


def _substituted_point_pole(T, n, a, u):
    j = Symbol("j")
    return T(n - 1) + Subs(1 / j, j, 1 / (n - 1))


@pytest.mark.parametrize(
    "step",
    [
        lambda T, n, a, u: T(n - 1) + u[n - 2],
        _moving_sum,
        _variable_width_sum,
        _nested_sum,
        _moving_product,
        _moving_comprehension,
        _relative_substitution,
        _compound_substitution,
        _nested_substitution,
        _compound_chain,
        _substituted_body_pole,
        _substituted_point_pole,
    ],
    ids=[
        "direct",
        "moving-sum",
        "variable-width-sum",
        "nested-bounds",
        "moving-product",
        "comprehension",
        "substitution",
        "compound-substitution",
        "nested-substitution",
        "compound-chain",
        "substituted-body-pole",
        "substituted-point-pole",
    ],
)
def test_undefined_sequence_steps_are_rejected(sequence_model, step):
    # A series returns an undefined query; a prepared model can reject the window.
    try:
        family = inspect_model(sequence_model(step), "outside", opening=0, a=1).solutions()
    except ModelInputError as error:
        assert "window" in str(error)
    else:
        assert family.is_empty is True


def _empty_prefix(T, n, a, u):
    j = Index("j")
    return T(n - 1) + Sum(u[j], (j, 0, n - 2))


def _independent_binders(T, n, a, u):
    j = Index("j")
    return T(n - 1) + Sum(u[j], (j, n - 1, n - 1)) + Sum(u[j], (j, -1, -1))


def _shared_spelling(T, n, a, u):
    return T(n - 1) + Sum(u[n], (n, 0, n - 1))


def _substituted_position(T, n, a, u):
    j = Symbol("j")
    return T(n - 1) + Subs(u[j], j, n - 1)


def _substituted_absolute(T, n, a, u):
    j = Index("j")
    return T(n - 1) + Subs(u[j], j, -1)


def _substituted_compound_position(T, n, a, u):
    j, k = Index("j"), Index("k")
    return T(n - 1) + Subs(u[j + k] + u[-1], j + k, n - 1)


def _substituted_derivative(T, n, a, u):
    j = Symbol("j")
    return T(n - 1) + Subs(u[j] + Derivative(a**2, a), j, n - 1)


@pytest.mark.parametrize(
    "step,expected",
    [
        (_empty_prefix, (0, 0, 2, 7)),
        (lambda T, n, a, u: T(n - 1) + u[-1], (0, 5, 10, 15)),
        (_independent_binders, (0, 7, 15, 25)),
        (_shared_spelling, (0, 2, 7, 17)),
        (_substituted_position, (0, 2, 5, 10)),
        (_substituted_absolute, (0, 5, 10, 15)),
        (_substituted_compound_position, (0, 7, 15, 25)),
        (_substituted_derivative, (0, 4, 9, 16)),
    ],
    ids=[
        "empty-range",
        "absolute-last-entry",
        "independent-binders",
        "bound-free-spelling",
        "substituted-position",
        "substituted-absolute",
        "compound-substitution",
        "substituted-derivative",
    ],
)
def test_relative_and_absolute_scopes_compose(sequence_model, solve_model, step, expected):
    case = solve_model(sequence_model(step), opening=0, a=1)
    assert tuple(case.levels) == expected


def test_a_scalar_query_retains_symbolic_input_size_conditions():
    n = Index("n")
    N = Symbol("N", integer=True, nonnegative=True)
    u = Vector("u", size=N, real=True)
    T = Series("T", n, real=True)
    law = Recurrence(T[n], T[n - 1] + u[n - 1], initial={0: 0})
    value = law.at(1).doit()
    assert value.subs(N, 0) is S.NaN
    assert value.subs(N, 1) == u[0].subs(N, 1)


@pytest.mark.parametrize("compound", [False, True], ids=["symbol", "compound"])
def test_ordered_substitution_keeps_an_independent_final_input(compound):
    n = Index("n")
    j, k = Symbol("j"), Symbol("k")
    u = Vector(2, 3, 5)
    body = Subs(u[j], (j + k, j), (n - 2, j + k)) if compound else Subs(u[k], (j, k), (n - 2, j))
    T = Series("T", n, real=True)
    law = Recurrence(T[n], T[n - 1] + body, initial={0: 0})
    assert law.at(1).doit().subs({j: -1, k: 0}).refine() == 5


@pytest.mark.parametrize("affine", [False, True], ids=["nonlinear-history", "affine-formula"])
def test_an_explicit_query_can_supply_initial_history(affine):
    n = Index("n")
    A, T = Series("A", n, real=True), Series("T", n, real=True)
    inner = Recurrence(A[n], A[n - 1] + 1, initial={0: 1})
    body = 2 * T[n - 1] if affine else T[n - 1] ** 2 + 1
    outer = Recurrence(T[n], body, initial={0: inner.at(2)})
    expected = [12, 3, 6, 12] if affine else [101, 3, 10, 101]
    assert [outer.at(point).doit() for point in (2, 0, 1, 2)] == expected
    if affine:
        k = Index("k", nonnegative=True)
        assert outer.at(k).doit() == 3 * 2**k


@pytest.mark.parametrize("origin", [0, 10, -3])
def test_scalar_cached_queries_preserve_their_mathematical_origin(origin):
    n = Index("n")
    T = Series("T", n, real=True)
    inputs = Vector(1, 1, 1)
    law = Recurrence(T[n], T[n - 1] + inputs[n - origin - 1], initial={origin: 0})
    assert law.index == n
    assert law.origin == origin
    assert law.start == origin + 1
    assert [law.at(point).doit() for point in (origin + 2, origin + 1, origin + 3)] == [2, 1, 3]


def test_query_history_and_a_whole_declared_window_are_different_problems():
    n = Index("n")
    T = Series("T", n, positive=True)
    law = Recurrence(T[n], Piecewise((T[n - 1] + 1, n < 2), (-1, True)), initial={0: 1})
    assert law.at(1).doit() == 2
    assert law.at(2).doit() is S.NaN

    class State(Modello):
        value = InstanceDummy("value", positive=True)

    class FullWindow(Modello):
        states = model_vector(State, size=3)
        n = Index("n")
        law = model_recurrence(states[n], value=Piecewise((states[n - 1].value + 1, n < 2), (-1, True)))
        opening = states[0].value
        queried = states[1].value

    assert inspect_model(FullWindow, "all-states", opening=1).solutions().is_empty is True
