"""Relative laws retain their indices while sharing ordinary scalar solving."""

import pickle
from copy import deepcopy

import pytest
from sympy import Piecewise, Rational, S, Symbol, simplify

from modello import Coordinate, Index, InstanceDummy, Modello, Recurrence, Series, Vector, component, inspect_model
from modello.errors import ModelDefinitionError, ModelInputError, SolutionCountError


class Growth(Modello):
    n = Index("n", nonnegative=True)
    T = Series("T", n, real=True)
    opening = InstanceDummy("opening", positive=True)
    ratio = InstanceDummy("ratio", positive=True)
    periods = InstanceDummy("periods", integer=True, nonnegative=True)
    law = Recurrence(T[n], ratio * T[n - 1], initial={0: opening})
    terminal = law.at(periods)
    direct = T[periods]
    path = law.window(0, 4)


def test_series_are_horizon_independent_and_negative_indices_are_relative():
    n = Index("n")
    series = Series("T", n, real=True)
    assert series[-1].index == -1
    assert series[n - 1].free_symbols == {n}
    assert series[n - 1].subs(n, 0) == series[-1]
    assert series[n].is_real is True
    assert series != Series("T", n, real=True)
    assert deepcopy(series) == pickle.loads(pickle.dumps(series)) == series
    t = Coordinate("t", real=True)
    continuous = Series("X", t, real=True)
    assert continuous[Rational(1, 2)].is_real is True
    for bad in (True, Rational(1, 2), 1.0):
        with pytest.raises(ModelInputError):
            series[bad]


def test_recurrence_finite_values_and_windows_preserve_zero_and_unit_coefficients():
    n = Index("n")
    series = Series("T", n, real=True)
    a = Symbol("a", real=True)
    law = Recurrence(series[n], a * series[n - 1] + 1, initial={0: 2})
    assert law.at(0).doit() == 2
    assert law.at(3).doit().subs(a, 0) == 1
    assert law.at(3).doit().subs(a, 1) == 5
    assert law.window(2, 4).doit().subs(a, 1).doit() == Vector(4, 5, 6)
    assert law.at(-1).doit() is S.NaN
    formula = law.at(n).doit()
    assert formula.subs({n: 3, a: 0}) == 1
    assert formula.subs({n: 3, a: 1}) == 5


def test_native_geometric_sum_preserves_conditional_zero_and_unit_coefficients():
    n = Index("n", nonnegative=True)
    T = Series("T", n, real=True)
    switch = Symbol("switch", real=True)
    coefficient = Piecewise((0, switch < 0), (1, True))
    law = Recurrence(T[n], coefficient * T[n - 1] + 1, initial={0: 2})
    formula = law.at(n).doit()
    assert formula != law.at(n)
    for position in (0, 1, 4):
        assert formula.subs({switch: -1, n: position}) == (2 if position == 0 else 1)
        assert formula.subs({switch: 1, n: position}) == 2 + position


def test_second_order_history_and_shifted_origin():
    n = Index("n", nonnegative=True)
    series = Series("F", n)
    fibonacci = Recurrence(series[n], series[n - 1] + series[n - 2], initial={0: 0, 1: 1})
    assert fibonacci.at(10).doit() == 55
    formula = fibonacci.at(n).doit()
    assert simplify(formula.subs(n, 10)) == 55
    shifted = Recurrence(series[n], 2 * series[n - 1], initial={10: 3})
    assert shifted.at(9).doit() is S.NaN
    assert shifted.at(10).doit() == 3
    assert shifted.at(12).doit() == 12


@pytest.mark.parametrize("positions", [(0, 1, 2, 3, 4), (4, 1, 3, 0, 4)], ids=["in-order", "revisited"])
def test_nonlinear_causal_law_can_be_evaluated_without_claiming_a_closed_form(positions):
    n = Index("n")
    series = Series("X", n)
    law = Recurrence(series[n], series[n - 1] ** 2 + 1, initial={0: 0})
    values = (0, 1, 2, 5, 26)
    for point in positions:
        assert law.at(point).doit() == values[point]
    assert law.at(n).doit() == law.at(n)


@pytest.mark.parametrize("lag,initial", [(0, {0: 1}), (-1, {0: 1}), (2, {0: 1}), (1, {0: 1, 2: 3})])
def test_noncausal_or_incomplete_histories_are_explicit(lag, initial):
    n = Index("n")
    series = Series("T", n)
    with pytest.raises(ModelInputError):
        Recurrence(series[n], series[n - lag], initial=initial)


def test_relative_laws_solve_forward_and_inverse(solve_model):
    result = solve_model(Growth, opening=1, ratio=2, periods=4)
    assert result.terminal == 16
    assert result.direct == result.terminal
    assert result.path == Vector(1, 2, 4, 8, 16)
    assert result.law.at(6).doit() == 64
    assert solve_model(Growth, terminal=16, ratio=2, periods=4).opening == 1
    assert solve_model(Growth, terminal=16, opening=1, periods=4).ratio == 2
    assert solve_model(Growth, terminal=16, opening=1, ratio=2).periods == 4


def test_relative_law_inspection_retains_templates_and_projects_direct_queries():
    problem = inspect_model(Growth, "family", opening=1, ratio=2, periods=4)
    assert problem.solutions().expression(problem.symbols.T[4]) == 16
    assert problem.symbols.law == problem.templates[("law",)]
    assert set(problem.templates) == {("T",), ("law",)}
    assert all(field.path[0] not in {"n", "T", "law"} for field in problem.fields)
    assert "T[n - 1]" in problem.format(problem.symbols.law)
    assert r"T_{n - 1}" in problem.latex(problem.symbols.law)
    with pytest.raises(TypeError):
        problem.templates[("T",)] = Series("other")


def test_component_scopes_and_law_overrides_rebind_all_views():
    class Pair(Modello):
        left = component(Growth)
        right = component(Growth)
        total = left.terminal + right.terminal

    source = Growth("source", opening=1, ratio=2, periods=4)
    pair = Pair("pair", left=source, right=source)
    assert pair.total == 32
    problem = inspect_model(Pair, "scope")
    assert problem.symbols.left.T != problem.symbols.right.T
    assert problem.symbols.left.law.at(2).free_symbols != problem.symbols.right.law.at(2).free_symbols

    class Shifted(Growth):
        law = Recurrence(Growth.T[Growth.n], Growth.ratio * Growth.T[Growth.n - 1] + 1, initial={0: Growth.opening})

    assert Shifted("override", opening=1, ratio=2, periods=4).terminal == 31

    class Refined(Growth):
        T = Series("T", Growth.n, real=True)
        ratio = InstanceDummy("ratio", real=True)
        law = Recurrence(Growth.T[Growth.n], Growth.ratio * Growth.T[Growth.n - 1] + 1, initial={0: Growth.opening})

    assert Refined("refined", opening=1, ratio=-2, periods=2).terminal == 3


def test_relative_law_inspection_does_not_evaluate(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Inspection must retain the law")

    monkeypatch.setattr(Recurrence, "_at", unexpected)
    problem = inspect_model(Growth, "raw", opening=1, ratio=2, periods=4)
    assert problem.templates[("law",)].target.index == Growth.n
    assert len(problem.vector_definitions[("path",)]) == 5


def test_series_domains_cover_intermediate_states_and_are_retained_in_flat_models():
    class PositivePath(Modello):
        n = Index("n")
        T = Series("T", n, positive=True)
        coefficient = InstanceDummy("coefficient", real=True)
        law = Recurrence(T[n], coefficient * T[n - 1], initial={0: 1})
        terminal = law.at(2)

    with pytest.raises(SolutionCountError):
        PositivePath("invalid", coefficient=-1)
    assert inspect_model(PositivePath, "invalid", coefficient=-1).solutions().is_empty is True
    assert PositivePath("valid", coefficient=2).terminal == 4
    assert PositivePath("open").terminal.has(S.NaN)
    assert inspect_model(PositivePath, "consistent").analyse().status != "inconsistent"
    assert inspect_model(PositivePath, "family").solutions().observe(coefficient=2).one().terminal == 4


def test_a_distant_query_uses_a_verified_formula_without_materializing_the_history(monkeypatch):
    n = Index("n", nonnegative=True)
    T = Series("T", n, positive=True)
    law = Recurrence(T[n], 2 * T[n - 1], initial={0: 1})

    def unexpected(*args, **kwargs):
        raise AssertionError("A closed-form query must not materialize the history")

    monkeypatch.setattr(Recurrence, "_at", unexpected)
    assert law.at(10000).doit() == 2**10000


def test_replacing_a_component_cannot_leave_a_reference_to_its_removed_law():
    class Defined(Modello):
        n = Index("n")
        T = Series("T", n, real=True)
        opening = InstanceDummy("opening", real=True)
        law = Recurrence(T[n], 2 * T[n - 1], initial={0: opening})

    class Values(Modello):
        n = Index("n")
        T = Series("T", n, real=True)
        opening = InstanceDummy("opening", real=True)

    class Parent(Modello):
        child = component(Defined)
        closing = child.law.at(3)

    with pytest.raises(ModelDefinitionError, match=r"child\.law"):

        class Invalid(Parent):
            child = component(Values)
