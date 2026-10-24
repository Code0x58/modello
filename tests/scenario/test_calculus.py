"""Finite data, calculus, and observation substitution compose in either order."""

import pytest
from sympy import Derivative, Eq, Integral, Limit, Piecewise, Rational, Sum, exp, simplify

from modello import Coordinate, Index, InstanceDummy, Modello, Vector, inspect_model, vector


def test_vector_objectives_support_gradients_hessians_and_mixed_partials(solve_model):
    class Objective(Modello):
        samples = Vector("x", size=3, real=True)
        scale = InstanceDummy("a", real=True)
        energy = scale * samples.dot(samples)
        slope = Derivative(energy, scale)
        curvature = Derivative(energy, samples[1], 2)
        mixed = Derivative(energy, scale, samples[1])

    case = solve_model(Objective, samples=[1, 2, 3], scale=5)
    assert (case.energy, case.slope, case.curvature, case.mixed) == (70, 14, 10, 4)


def test_native_sums_integrals_and_moving_endpoint_derivatives_compose(solve_model):
    class Exposure(Modello):
        coefficients = Vector("a", size=3, real=True)
        duration = InstanceDummy("duration", positive=True)
        i = Index("i")
        t = Coordinate("t", real=True)
        accumulated = Integral(Sum(coefficients[i] * t**i, (i, 0, 2)), (t, 0, duration))
        endpoint_rate = Derivative(accumulated, duration)

    case = solve_model(Exposure, coefficients=[1, 2, 3], duration=2)
    assert (case.accumulated, case.endpoint_rate) == (14, 17)


@pytest.mark.parametrize("endpoint", [0, Rational(1, 2), 1])
def test_limits_are_evaluated_after_the_observed_endpoint_is_applied(solve_model, endpoint):
    class Response(Modello):
        endpoint = InstanceDummy("endpoint", real=True)
        t = Coordinate("t", real=True)
        response = Limit((exp(t) - 1) / t, t, endpoint)

    case = solve_model(Response, endpoint=endpoint)
    expected = 1 if endpoint == 0 else (exp(endpoint) - 1) / endpoint
    assert simplify(case.response - expected) == 0


@pytest.mark.parametrize("parameter, expected", [(0, 1), (1, Rational(1, 2)), (-2, Rational(1, 2))])
def test_parameterized_limits_retain_exceptional_parameter_values(solve_model, parameter, expected):
    class LimitModel(Modello):
        a = InstanceDummy("a", real=True)
        t = Coordinate("t", real=True)
        value = Limit((t + a) / (t + 2 * a), t, 0)

    assert solve_model(LimitModel, a=parameter).value == expected


def test_implicit_euler_is_an_ordinary_finite_joint_equation_system(solve_model):
    """A recurrence need not impose an evaluation schedule on the flat solver."""

    class Decay(Modello):
        initial = InstanceDummy("initial", positive=True)
        rate = InstanceDummy("rate", nonnegative=True)
        step = InstanceDummy("step", positive=True)
        values = Vector("x", size=4, positive=True)
        n = Index("n")
        values = vector(Piecewise((initial, Eq(n, 0)), (values[n - 1] - step * rate * values[n], True)), (n, 0, 3))

    case = solve_model(Decay, initial=1, rate=1, step=Rational(1, 2))
    assert case.values == Vector(1, Rational(2, 3), Rational(4, 9), Rational(8, 27))


def test_a_family_response_derivative_is_distinct_from_an_unconstrained_partial_derivative():
    class Budget(Modello):
        weights = Vector("w", size=2, nonnegative=True)
        total = weights.sum()
        energy = weights.dot(weights)

    problem = inspect_model(Budget, "budget", total=1)
    family = problem.solutions(problem.symbols.weights)
    parameter = family.parameters[0].symbol
    partial = family.expression(Derivative(problem.symbols.energy, parameter))
    response = family.expression(problem.symbols.energy)
    assert partial == 2 * parameter
    assert family.expression(Derivative(response, parameter)) == 4 * parameter - 2
    assert family.where(Eq(Derivative(response, parameter), 0)).one().weights == Vector(Rational(1, 2), Rational(1, 2))


def test_one_symbol_can_be_bound_in_a_body_and_free_in_the_same_operators_limits():
    """This is the same lexical scope used by native Sum and Integral."""
    n = Index("n")
    formula = vector(n**2, (n, n, n + 2))
    assert n in formula.free_symbols
    assert formula.subs(n, 3).doit() == Vector(9, 16, 25)
    assert Sum(n**2, (n, n, n + 2)).subs(n, 3).doit() == 50


def test_a_nested_sum_with_an_outer_dependent_bound_keeps_the_full_domain(solve_model):
    class Triangular(Modello):
        samples = Vector("x", size=3, real=True)
        i, j = Index("i"), Index("j")
        total = Sum(samples[j], (j, 0, i), (i, 0, 2))

    assert solve_model(Triangular, samples=[1, 2, 4]).total == 11
