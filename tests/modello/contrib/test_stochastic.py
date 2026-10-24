"""Itô corrections, conditional Brownian moments, and flat parameter inference."""

import pytest
from sympy import Abs, Piecewise, Rational, S, Symbol, conjugate, exp, log, simplify

from modello import Coordinate, InstanceDummy, Modello, Series
from modello.contrib.stochastic import ItoProcess


def test_brownian_chain_rule_and_explicit_time_dependence():
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    law = ItoProcess(x, 0, 1, t)
    assert law.drift(x**2).doit() == 1
    assert law.diffusion(x**2).doit() == 2 * x
    assert law.quadratic_variation_rate(x**2).doit() == 4 * x**2
    function = exp(x - t / 2)
    assert law.drift(function).doit() == 0
    assert law.generator(function).doit() == function / 2
    assert law.drift(x**3).subs(x, 2).doit() == 6


def test_geometric_brownian_transform_includes_the_second_order_correction():
    x, t = Coordinate("x", positive=True), Coordinate("t", real=True)
    mu, sigma = Symbol("mu", real=True), Symbol("sigma", positive=True)
    law = ItoProcess(x, mu * x, sigma * x, t)
    assert law.drift(log(x)).doit() == mu - sigma**2 / 2
    assert law.diffusion(log(x)).doit() == sigma


def test_the_laws_real_state_premise_applies_to_coefficients_and_transforms():
    x, t = Coordinate("x"), Coordinate("t", real=True)
    mu, sigma = Symbol("mu", real=True), Symbol("sigma", positive=True)
    law = ItoProcess(x, mu * conjugate(x), sigma * conjugate(x), t)
    assert law.mean(1, 1).doit() == exp(mu)
    assert simplify(law.drift(conjugate(x) ** 2).doit() - (2 * mu + sigma**2) * x**2) == 0


def test_the_declared_positive_state_domain_is_retained_in_coefficient_recognition():
    t = Coordinate("t", real=True)
    x = Series("X", t, positive=True)[t]
    mu, sigma = Symbol("mu", real=True), Symbol("sigma", positive=True)
    law = ItoProcess(x, mu * Abs(x), sigma * Abs(x), t)
    assert law.mean(1, 1).doit() == exp(mu)
    assert law.drift(log(x)).doit() == mu - sigma**2 / 2


def test_affine_coefficient_recognition_preserves_parameter_dependent_cases():
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    switch = Symbol("switch", real=True)
    law = ItoProcess(x, Piecewise((2 * x + 3, switch > 0), (3 * x + 1, True)), 2, t)
    mean = law.mean(1, 1).doit()
    assert simplify(mean.subs(switch, 1) - (5 * exp(2) - 3) / 2) == 0
    assert simplify(mean.subs(switch, -1) - (4 * exp(3) - 1) / 3) == 0


def test_named_formulas_are_expanded_before_the_ito_derivatives_and_observations(solve_model):
    class Square(Modello):
        t = Coordinate("t", real=True)
        state = InstanceDummy("state", positive=True)
        mu = InstanceDummy("mu", real=True)
        sigma = InstanceDummy("sigma", positive=True)
        process = ItoProcess(state, mu * state, sigma * state, t)
        squared = state**2
        drift = process.drift(squared)

    assert solve_model(Square, state=3, mu=1, sigma=2).drift == 54


def test_flat_model_can_infer_diffusion_parameters_from_transformed_coefficients(solve_model):
    class LogModel(Modello):
        t = Coordinate("t", real=True)
        state = Coordinate("S", positive=True)
        mu = InstanceDummy("mu", real=True)
        sigma = InstanceDummy("sigma", positive=True)
        process = ItoProcess(state, mu * state, sigma * state, t)
        log_drift = process.drift(log(state))
        log_diffusion = process.diffusion(log(state))

    result = solve_model(LogModel, log_drift=Rational(2, 25), log_diffusion=Rational(1, 5))
    assert result.mu == Rational(1, 10)
    assert result.sigma == Rational(1, 5)


def test_exact_arithmetic_and_geometric_values_and_conditional_moments():
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    arithmetic = ItoProcess(x, 5, 3, t)
    assert arithmetic.exact(2, 4, 1).doit() == 25
    assert arithmetic.mean(2, 4).doit() == 22
    assert arithmetic.variance(2, 4).doit() == 36
    geometric = ItoProcess(x, log(2) * x, log(Rational(5, 4)) * x, t)
    assert geometric.mean(100, 1).doit() == 200
    sigma = Symbol("sigma", positive=True)
    geometric = ItoProcess(x, log(2) * x, sigma * x, t)
    assert simplify(geometric.variance(100, 1).doit() - 40000 * (exp(sigma**2) - 1)) == 0
    assert arithmetic.mean(2, -1).doit() is S.NaN


def test_unsupported_exact_sde_classes_do_not_get_an_ordinary_ode_solution():
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    process = ItoProcess(x, -x, 1, t)
    with pytest.raises(NotImplementedError, match="weighted Wiener integral"):
        process.exact(1, 1, 0).doit()


def test_affine_additive_moments_include_mean_reversion_and_the_zero_rate_case():
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    rate = Symbol("rate", real=True)
    process = ItoProcess(x, rate * (4 - x), 2, t)
    mean, variance = process.mean(1, 1).doit(), process.variance(1, 1).doit()
    assert mean.subs(rate, 0) == 1
    assert variance.subs(rate, 0) == 4
    assert simplify(mean.subs(rate, log(2)) - Rational(5, 2)) == 0
    assert simplify(variance.subs(rate, log(2)) - Rational(3, 2) / log(2)) == 0


def test_zero_time_requires_a_zero_wiener_increment():
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    process = ItoProcess(x, 0, 1, t)
    assert process.exact(3, 0, 0).doit() == 3
    assert process.exact(3, 0, 1).doit() is S.NaN


def test_zero_diffusion_preserves_exact_deterministic_limits():
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    assert ItoProcess(x, 2 * x, 0, t).exact(1, 1, 0).doit() == exp(2)
    assert simplify(ItoProcess(x, 3 - 2 * x, 0, t).exact(0, 1, 0).doit() - Rational(3, 2) * (1 - exp(-2))) == 0


@pytest.mark.parametrize("as_series", [False, True], ids=["coordinate", "series"])
def test_conditional_queries_retain_the_initial_state_domain(as_series):
    t = Coordinate("t", real=True)
    state = Series("X", t, positive=True)[t] if as_series else Coordinate("x", positive=True)
    process = ItoProcess(state, state, state / 2, t)
    assert process.exact(-1, 1, 0).doit() is S.NaN
    assert process.mean(0, 1).doit() is S.NaN
    assert process.variance(-1, 1).doit() is S.NaN
    assert process.mean(2, 0).doit() == 2
    opening = Symbol("opening", real=True)
    mean = process.mean(opening, 1).doit()
    assert mean.subs(opening, -1) is S.NaN
    assert mean.subs(opening, 2) == 2 * exp(1)
