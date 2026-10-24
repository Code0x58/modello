"""Native symbolic reconstruction preserves function kinds, scopes, and domains."""

from copy import deepcopy
from pickle import dumps, loads

import pytest
from sympy import Abs, E, Integral, S, Sum, Symbol, simplify

from modello import Coordinate, Index, Modello, Recurrence, Series, Vector, inspect_model, vector
from modello.contrib.stochastic import ItoProcess
from modello.errors import ModelInputError


def reconstructed(expression):
    return (
        expression.func(*expression.args),
        deepcopy(expression),
        loads(dumps(expression)),
    )


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
def test_series_kind_is_part_of_the_signature_after_coordinate_reconstruction(discrete):
    coordinate = Index("u") if discrete else Coordinate("u", real=True)
    series = Series("X", coordinate, positive=True)
    renamed = series.xreplace({coordinate: Symbol("renamed")})
    for candidate in (*reconstructed(series), renamed):
        assert candidate.discrete is discrete
        assert candidate.symbol == series.symbol
        assert candidate[2].assumptions0["positive"] is True
    assert series.subs(coordinate, 2) == series


@pytest.mark.parametrize("form", ["values", "symbolic-size", "comprehension"])
def test_vector_reconstruction_preserves_coordinates_and_free_bound_endpoints(form):
    first, second = Index("u"), Index("u")
    if form == "values":
        expression = Vector(first + second, first * second)
        observations = {first: 2, second: 3}
        expected = Vector(5, 6)
    elif form == "symbolic-size":
        expression = Vector("x", size=first, positive=True)
        observations = {first: 3}
        expected = expression.subs(observations)
    else:
        expression = vector(first + second, (first, second, second + 2))
        observations = {second: 2}
        expected = Vector(4, 5, 6)
    for candidate in (*reconstructed(expression), expression.as_dummy()):
        assert candidate.free_symbols == expression.free_symbols
        assert candidate.subs(observations).doit() == expected


def test_vector_alpha_renaming_distinguishes_a_bound_coordinate_from_its_external_endpoint():
    index = Index("u")
    expression = vector(index, (index, index, index + 2))
    for candidate in (*reconstructed(expression), expression.as_dummy()):
        assert candidate.free_symbols == {index}
        assert candidate.subs(index, 3).doit() == Vector(3, 4, 5)


def test_recurrence_alpha_renaming_preserves_a_discrete_history_and_native_sum():
    n = Index("n")
    series = Series("X", n, positive=True)
    law = Recurrence(series[n], series[n - 1] ** 2 + 1, initial={0: 1})
    renamed = law.as_dummy()
    assert renamed.target.index != n
    assert renamed.target.series.discrete
    assert renamed.target.series.symbol == series.symbol
    for candidate in (*reconstructed(law), renamed):
        assert candidate.at(3).doit() == 26
    total = Sum(law.at(n), (n, 0, 3))
    assert total.doit() == total.as_dummy().doit() == 34


def test_recurrence_history_uses_the_same_function_after_signature_coordinate_relabeling():
    n, renamed = Index("n"), Index("renamed")
    series = Series("X", n)
    alias = series.xreplace({n: renamed})
    law = Recurrence(series[n], alias[n - 1] + 1, initial={0: 0})
    assert law.at(3).doit() == 3
    with pytest.raises(ModelInputError, match="same series"):
        Recurrence(series[n], series[n - 1] + 1, initial={0: alias[0]})


@pytest.mark.parametrize("delay", [0, -1], ids=["same-step", "future-step"])
def test_signature_coordinate_relabeling_cannot_bypass_causal_recurrence_validation(delay):
    n, renamed = Index("n"), Index("renamed")
    series = Series("X", n)
    alias = series.xreplace({n: renamed})
    with pytest.raises(ModelInputError, match="positive lags"):
        Recurrence(series[n], alias[n - delay] + 1, initial={0: 0})


@pytest.fixture(params=[False, True], ids=["coordinate-state", "series-state"])
def process(request):
    # Equal display names must never turn independent coordinates into aliases.
    time = Coordinate("u", real=True)
    state = Series("u", time, positive=True)[time] if request.param else Coordinate("u", positive=True)
    return ItoProcess(state, Abs(state), Abs(state), time), state, time


def test_ito_declarations_reconstruct_and_alpha_rename_without_losing_state_domains(process):
    law, state, time = process
    renamed = law.as_dummy()
    assert renamed.state != state
    assert renamed.time != time
    assert renamed.driver.index == renamed.time
    assert renamed.driver.symbol == law.driver.symbol
    for candidate in (*reconstructed(law), renamed):
        assert candidate.mean(2, 1).doit() == 2 * E
        assert candidate.mean(-1, 1).doit() is S.NaN
        assert candidate.variance(0, 1).doit() is S.NaN


@pytest.mark.parametrize(
    "operation, expected",
    [
        ("drift", lambda state, time: 6 * state**3 + 2 * time),
        ("diffusion", lambda state, time: 3 * state**3),
        ("generator", lambda state, time: 6 * state**3),
    ],
)
def test_ito_transform_reconstruction_keeps_its_external_coordinate_frame(process, operation, expected):
    law, state, time = process
    quantity = getattr(law, operation)(state**3 + time**2)
    for candidate in (*reconstructed(quantity), quantity.as_dummy()):
        assert candidate.free_symbols == quantity.free_symbols
        assert simplify(candidate.doit() - expected(state, time)) == 0
        assert candidate.subs(state, 2).doit() == quantity.subs(state, 2).doit()


@pytest.mark.parametrize("operation, arguments", [("mean", (2, 1)), ("variance", (2, 1)), ("exact", (2, 1, 0))])
def test_conditional_queries_reconstruct_without_changing_their_law_or_domains(process, operation, arguments):
    law, _, _ = process
    quantity = getattr(law, operation)(*arguments)
    for candidate in (*reconstructed(quantity), quantity.as_dummy()):
        assert simplify(candidate.doit() - quantity.doit()) == 0
    invalid = law.mean(-1, 1)
    assert invalid.as_dummy().doit() is S.NaN


@pytest.mark.parametrize("native, factor", [(Sum, 3), (Integral, 2)], ids=["sum", "integral"])
def test_native_binders_do_not_capture_the_free_coordinates_of_ito_transforms(process, native, factor):
    law, state, _ = process
    index = Index("u")
    expression = native(law.drift(state**3), (index, 0, 2))
    for candidate in (*reconstructed(expression), expression.as_dummy()):
        assert simplify(candidate.doit() - factor * 6 * state**3) == 0
        assert candidate.subs(state, 2).doit() == factor * 48


@pytest.mark.parametrize("native, last, expected", [(Sum, 3, 6 * E), (Integral, 2, 3 * E / 2)], ids=["sum", "integral"])
def test_native_binders_can_vary_a_conditional_initial_value_without_rebinding_its_state(
    process, native, last, expected
):
    law, _, _ = process
    initial = Symbol("u", positive=True)
    expression = native(law.mean(initial, 1), (initial, 1, last))
    assert expression.doit() == expression.as_dummy().doit() == expected


def test_coordinate_relabeling_and_observation_substitution_preserve_the_declared_domain(process):
    law, state, time = process
    renamed = law.xreplace({state: Symbol("state"), time: Symbol("time")})
    assert renamed.driver.index == renamed.time
    assert renamed.mean(2, 1).doit() == 2 * E
    assert renamed.mean(-1, 1).doit() is S.NaN
    initial = Symbol("initial", real=True)
    for candidate in (law, law.as_dummy(), renamed):
        query = candidate.mean(initial, 1)
        assert query.subs(initial, -1).doit() is S.NaN
        assert query.xreplace({initial: S.NegativeOne}).doit() is S.NaN


def test_contextual_series_lookup_uses_function_identity_after_native_alpha_renaming():
    class Coupled(Modello):
        n = Index("n")
        A, B = Series("A", n, real=True), Series("B", n, real=True)
        first = Recurrence(A[n], A[n - 1] + 1, initial={0: 1})
        second = Recurrence(B[n], B[n - 1] + A[n - 1], initial={0: 0})
        total = Sum(B[n], (n, 0, 3)).as_dummy()

    assert Coupled("constructor").total == 10
    assert inspect_model(Coupled, "family").solutions().one().total == 10


def test_alpha_renamed_laws_rebind_inherited_queries_without_changing_their_kind():
    class Discrete(Modello):
        n = Index("n")
        T = Series("T", n, positive=True)
        law = Recurrence(T[n], 2 * T[n - 1], initial={0: 1})
        terminal = law.at(3)

    class RenamedDiscrete(Discrete):
        law = Discrete.law.as_dummy()

    assert RenamedDiscrete("renamed").terminal == 8

    class Continuous(Modello):
        t = Coordinate("u", real=True)
        x = Coordinate("u", positive=True)
        law = ItoProcess(x, x, x, t)
        drift = law.drift(x**3)
        mean = law.mean(2, 1)

    class RenamedContinuous(Continuous):
        law = Continuous.law.as_dummy()

    result = RenamedContinuous("renamed")
    assert result.drift == 6 * Continuous.x**3
    assert result.mean == 2 * E
