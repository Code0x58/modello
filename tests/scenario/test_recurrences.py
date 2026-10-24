"""Causal histories, variable forcing, law composition, and query domains."""

import pytest
from sympy import Rational, S, Sum, Symbol, simplify

from modello import Index, InstanceDummy, Modello, Recurrence, Series, Vector, inspect_model
from modello.errors import ModelInputError


@pytest.mark.parametrize("coefficient", [-2, 0, 1, 2, Rational(1, 2)])
@pytest.mark.parametrize("origin", [-2, 0, 4])
def test_affine_recurrences_match_an_independent_iteration_at_shifted_origins(coefficient, origin):
    n = Index("n")
    T = Series("T", n, real=True)
    law = Recurrence(T[n], coefficient * T[n - 1] + n, initial={origin: 3})
    expected = {origin: 3}
    for point in range(origin + 1, origin + 6):
        expected[point] = coefficient * expected[point - 1] + point
    for point in (origin + 5, origin, origin + 2, origin + 5):
        assert simplify(law.at(point).doit() - expected[point]) == 0
    assert law.at(origin - 1).doit() is S.NaN


def test_symbolic_queries_keep_the_integer_position_domain_after_simplification():
    n = Index("n")
    T = Series("T", n, positive=True)
    law = Recurrence(T[n], 2 * T[n - 1], initial={0: 1})
    position = Symbol("position", real=True)
    formula = law.at(position).doit()
    assert formula.subs(position, 3) == 8
    assert formula.subs(position, Rational(1, 2)) is S.NaN
    assert formula.subs(position, -1) is S.NaN


def test_declared_series_compose_without_repeating_the_other_laws_closed_form(solve_model):
    class Coupled(Modello):
        n = Index("n")
        A, B = Series("A", n, real=True), Series("B", n, real=True)
        first = Recurrence(A[n], A[n - 1] + 1, initial={0: 1})
        second = Recurrence(B[n], B[n - 1] + A[n - 1], initial={0: 0})
        terminal = B[3]

    assert solve_model(Coupled).terminal == 6


def test_mutually_driven_series_can_compose_when_time_strictly_decreases(solve_model):
    class Coupled(Modello):
        n = Index("n")
        A, B = Series("A", n, real=True), Series("B", n, real=True)
        first = Recurrence(A[n], B[n - 1] + 1, initial={0: 1})
        second = Recurrence(B[n], A[n - 1] + 2, initial={0: 0})
        left, right = A[4], B[4]

    assert (solve_model(Coupled).left, solve_model(Coupled).right) == (7, 6)


def test_family_queries_and_native_sums_see_the_same_composed_series_laws():
    class Coupled(Modello):
        n = Index("n")
        A, B = Series("A", n, real=True), Series("B", n, real=True)
        first = Recurrence(A[n], A[n - 1] + 1, initial={0: 1})
        second = Recurrence(B[n], B[n - 1] + A[n - 1], initial={0: 0})
        total = Sum(B[n], (n, 0, 3))

    problem = inspect_model(Coupled, "queries")
    family = problem.solutions()
    assert family.one().total == 10
    assert family.expression(problem.symbols.B[3]) == 6


def test_a_second_order_boundary_can_infer_an_unobserved_history_value(solve_model):
    class Difference(Modello):
        n = Index("n")
        T = Series("T", n, real=True)
        first = InstanceDummy("first", real=True)
        second = InstanceDummy("second", real=True)
        law = Recurrence(T[n], 2 * T[n - 1] - T[n - 2], initial={0: first, 1: second})
        terminal = T[4]

    case = solve_model(Difference, first=1, terminal=9)
    assert case.second == 3
    assert case.law.window(0, 4).doit() == Vector(1, 3, 5, 7, 9)


def test_an_unexposed_invalid_intermediate_state_cannot_be_hidden_by_a_valid_terminal():
    n = Index("n")
    T = Series("T", n, positive=True)
    law = Recurrence(T[n], -T[n - 1], initial={0: 1})
    assert law.at(2).doit() is S.NaN
    assert law.at(0).doit() == 1


@pytest.mark.parametrize("history", [{0: 1, 2: 3}, {0: 1, 1: 2, 1.5: 3}])
def test_history_gaps_and_noninteger_positions_are_explicit_errors(history):
    n = Index("n")
    T = Series("T", n)
    with pytest.raises(ModelInputError):
        Recurrence(T[n], T[n - 1] + 1, initial=history)


def test_alpha_equivalent_declarations_are_one_law_rather_than_an_ambiguous_choice(solve_model):
    class Equivalent(Modello):
        n = Index("n")
        T = Series("T", n, real=True)
        first = Recurrence(T[n], T[n - 1] + 1, initial={0: 1})
        second = first.as_dummy()
        terminal = T[3]

    assert solve_model(Equivalent).terminal == 4
