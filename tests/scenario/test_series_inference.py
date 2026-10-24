"""Relative lags, input lengths, missing innovations, and observation windows."""

import pytest
from sympy import S, Sum, simplify

from modello import Index, InstanceDummy, Modello, Recurrence, Series, Vector
from modello.errors import ModelInputError


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("observe_states", [False, True], ids=["innovations-to-states", "states-to-innovations"])
def test_any_recurrence_order_can_use_a_sum_of_relative_lags_and_infer_its_window(
    solve_model, order, count, observe_states
):
    class Autoregression(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        history = Vector("h", size=order, real=True)
        coefficients = Vector("a", size=order, real=True)
        innovations = Vector("e", size=N - order + 1, real=True)
        n, j = Index("n"), Index("j")
        T = Series("T", n, real=True)
        law = Recurrence(
            T[n],
            Sum(coefficients[j] * T[n - 1 - j], (j, 0, order - 1)) + innovations[n - order],
            initial=dict(enumerate(history)),
        )
        levels = law.window(0, N)

    expected = list(range(1, order + 1))
    innovations = list(range(1, count + 1))
    for innovation in innovations:
        expected.append(sum(expected[-order:]) + innovation)
    observations = {"levels": expected} if observe_states else {"innovations": innovations}
    case = solve_model(Autoregression, history=list(range(1, order + 1)), coefficients=[1] * order, **observations)
    assert case.N == order + count - 1
    assert case.levels == Vector(expected)
    assert case.innovations == Vector(innovations)


@pytest.mark.parametrize("aggregate", [False, True], ids=["unrolled", "sum"])
def test_an_unweighted_sum_of_lags_has_the_same_causality_and_values_as_unrolled_terms(aggregate):
    n, j = Index("n"), Index("j")
    T = Series("T", n, integer=True)
    body = Sum(T[n - j], (j, 1, 3)) if aggregate else T[n - 1] + T[n - 2] + T[n - 3]
    law = Recurrence(T[n], body, initial={0: 1, 1: 2, 2: 3})
    assert law.window(0, 6).doit() == Vector(1, 2, 3, 6, 11, 20, 37)


def test_a_sum_cannot_hide_a_noncausal_self_reference():
    n, j = Index("n"), Index("j")
    T = Series("T", n, real=True)
    with pytest.raises(ModelInputError, match="positive lags"):
        Recurrence(T[n], Sum(T[n - j], (j, 0, 2)), initial={0: 1, 1: 2})


def test_a_finite_nonlinear_query_composes_with_index_dependent_forcing():
    n = Index("n")
    T = Series("T", n, real=True)
    inputs = Vector(1, -2, 3, -4)
    law = Recurrence(T[n], T[n - 1] ** 2 + inputs[n - 1], initial={0: 1})
    assert law.window(0, 4).doit() == Vector(1, 2, 2, 7, 45)


def test_a_symbolic_affine_query_retains_zero_unit_and_negative_coefficient_branches():
    from sympy import Symbol

    n = Index("n")
    T = Series("T", n, real=True)
    a = Symbol("a", real=True)
    law = Recurrence(T[n], a * T[n - 1] + 3, initial={0: 2})
    formula = law.at(5).doit()
    for coefficient in (0, 1, -1, 2):
        expected = S(2)
        for _ in range(5):
            expected = coefficient * expected + 3
        assert simplify(formula.subs(a, coefficient) - expected) == 0


@pytest.mark.parametrize("order", [1, 2])
def test_symbolic_closed_forms_do_not_depend_on_whether_the_lags_were_aggregated(order):
    from sympy import Symbol

    n, j = Index("n"), Index("j")
    T = Series("T", n, integer=True)
    history = {point: point + 1 for point in range(order)}
    summed = Recurrence(T[n], Sum(T[n - j], (j, 1, order)), initial=history)
    unrolled = Recurrence(T[n], sum(T[n - lag] for lag in range(1, order + 1)), initial=history)
    point = Symbol("point", integer=True, nonnegative=True)
    assert summed.at(point).doit() == unrolled.at(point).doit()
    assert simplify(summed.at(point).doit().subs(point, 7) - summed.at(7).doit()) == 0
