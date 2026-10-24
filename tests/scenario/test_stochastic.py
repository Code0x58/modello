"""Itô calculus composes with finite operators and respects universal identities."""

import pytest
from sympy import Rational, S, Sum, exp, log, simplify

from modello import Coordinate, Index, InstanceDummy, Modello, Series, Vector
from modello.contrib.stochastic import ItoProcess
from modello.errors import ModelInputError


@pytest.mark.parametrize("drift", [-2, 0, Rational(1, 2)])
@pytest.mark.parametrize("diffusion", [0, 2])
@pytest.mark.parametrize("elapsed", [0, 1, Rational(3, 2)])
def test_arithmetic_values_and_moments_match_the_defining_increment_law(drift, diffusion, elapsed):
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    process = ItoProcess(x, drift, diffusion, t)
    increment = 0 if elapsed == 0 else 1
    assert process.mean(3, elapsed).doit() == 3 + drift * elapsed
    assert process.variance(3, elapsed).doit() == diffusion**2 * elapsed
    assert process.exact(3, elapsed, increment).doit() == 3 + drift * elapsed + diffusion * increment


@pytest.mark.parametrize("as_series", [False, True], ids=["coordinate", "series-state"])
def test_time_dependent_nonlinear_coefficients_still_support_local_ito_calculus(as_series):
    t = Coordinate("t", real=True)
    x = Series("X", t, real=True)[t] if as_series else Coordinate("x", real=True)
    a, b = t - x**3, 1 + x**2
    process = ItoProcess(x, a, b, t)
    function = x**2 + t**2
    assert simplify(process.drift(function).doit() - (2 * t + 2 * x * a + b**2)) == 0
    assert simplify(process.diffusion(function).doit() - 2 * x * b) == 0
    assert simplify(process.generator(function).doit() - (2 * x * a + b**2)) == 0
    assert simplify(process.quadratic_variation_rate(function).doit() - 4 * x**2 * b**2) == 0


@pytest.mark.parametrize("body_kind", ["native-sum", "vector-sum", "polynomial"])
def test_standalone_ito_calculus_is_invariant_under_finite_body_materialization(body_kind):
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    i = Index("i")
    body = {
        "native-sum": Sum(x ** (i + 1), (i, 0, 2)),
        "vector-sum": Vector(x, x**2, x**3).sum(),
        "polynomial": x + x**2 + x**3,
    }[body_kind]
    process = ItoProcess(x, 0, 1, t)
    drift = process.drift(body).doit()
    diffusion = process.diffusion(body).doit()
    assert drift.subs(x, 0) == 1
    assert simplify(drift - (1 + 3 * x)) == 0
    assert simplify(diffusion - (1 + 2 * x + 3 * x**2)) == 0
    assert diffusion.subs(x, 0) == 1


@pytest.mark.parametrize("coefficient", ["affine", "nonlinear"])
@pytest.mark.parametrize("query, expected", [("exact", 3), ("mean", 3), ("variance", 0)])
def test_zero_time_identities_do_not_require_a_closed_form_for_the_process(coefficient, query, expected):
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    process = ItoProcess(x, 4 - x if coefficient == "affine" else -(x**3), 2, t)
    arguments = (3, 0, 0) if query == "exact" else (3, 0)
    assert getattr(process, query)(*arguments).doit() == expected


@pytest.mark.parametrize("query", ["exact", "mean", "variance"])
def test_zero_time_preserves_the_initial_state_domain_even_for_unsupported_coefficients(query):
    x, t = Coordinate("x", positive=True), Coordinate("t", real=True)
    process = ItoProcess(x, -(x**3), 1, t)
    arguments = (0, 0, 0) if query == "exact" else (0, 0)
    assert getattr(process, query)(*arguments).doit() is S.NaN


def test_nonzero_time_unsupported_exact_classes_are_explicitly_rejected():
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    with pytest.raises(NotImplementedError):
        ItoProcess(x, -(x**3), 2, t).mean(3, 1).doit()
    with pytest.raises(NotImplementedError, match="weighted"):
        ItoProcess(x, 4 - x, 2, t).exact(3, 1, 0).doit()


def test_invalid_zero_time_realizations_are_not_silently_accepted():
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    process = ItoProcess(x, -(x**3), 1, t)
    assert process.exact(3, 0, 1).doit() is S.NaN


def test_moment_targets_infer_drift_and_nonnegative_diffusion_through_the_flat_solver(solve_model):
    class Measurement(Modello):
        x, t = Coordinate("x", real=True), Coordinate("t", real=True)
        drift = InstanceDummy("mu", real=True)
        diffusion = InstanceDummy("sigma", nonnegative=True)
        process = ItoProcess(x, drift, diffusion, t)
        mean = process.mean(2, 2)
        variance = process.variance(2, 2)

    case = solve_model(Measurement, mean=8, variance=18)
    assert (case.drift, case.diffusion) == (3, 3)


def test_an_additional_process_is_not_treated_as_a_deterministic_constant():
    t = Coordinate("t", real=True)
    X, Y = Series("X", t, real=True), Series("Y", t, real=True)
    process = ItoProcess(X[t], 0, 1, t)
    with pytest.raises(ModelInputError, match="unmodelled"):
        process.drift(X[t] * Y[t])


def test_geometric_coefficients_and_log_transform_have_matching_conditional_moments():
    x, t = Coordinate("x", positive=True), Coordinate("t", real=True)
    process = ItoProcess(x, log(2) * x, Rational(1, 3) * x, t)
    assert process.mean(5, 2).doit() == 20
    assert simplify(process.variance(5, 2).doit() - 400 * (exp(Rational(2, 9)) - 1)) == 0
    assert process.diffusion(log(x)).doit() == Rational(1, 3)


@pytest.mark.parametrize("rate", [-2, 0, 1])
@pytest.mark.parametrize("diffusion", [0, 2])
def test_affine_additive_moments_keep_the_zero_rate_branch(rate, diffusion):
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    process = ItoProcess(x, rate * x + 3, diffusion, t)
    elapsed = S(2)
    integrated = elapsed if rate == 0 else (exp(rate * elapsed) - 1) / rate
    variance_integral = elapsed if rate == 0 else (exp(2 * rate * elapsed) - 1) / (2 * rate)
    assert simplify(process.mean(5, elapsed).doit() - (5 * exp(rate * elapsed) + 3 * integrated)) == 0
    assert simplify(process.variance(5, elapsed).doit() - diffusion**2 * variance_integral) == 0
    if diffusion == 0:
        assert process.exact(5, elapsed, 1).doit() == process.mean(5, elapsed).doit()


def test_a_realized_zero_wiener_increment_is_not_the_conditional_mean_of_a_geometric_process():
    x, t = Coordinate("x", positive=True), Coordinate("t", real=True)
    law = ItoProcess(x, x, x, t)
    assert law.exact(2, 1, 0).doit() == 2 * exp(Rational(1, 2))
    assert law.mean(2, 1).doit() == 2 * exp(1)


def test_model_definitions_expand_inside_ito_coefficients_and_transforms(solve_model):
    class Transformed(Modello):
        x = InstanceDummy("x", real=True)
        t = Coordinate("t", real=True)
        rate = InstanceDummy("rate", real=True)
        scale = InstanceDummy("scale", nonnegative=True)
        drift = rate * x
        noise = scale * x
        square = x**2
        process = ItoProcess(x, drift, noise, t)
        transformed_drift = process.drift(square)
        transformed_diffusion = process.diffusion(square)

    case = solve_model(Transformed, x=2, rate=2, scale=3)
    assert case.transformed_drift == 52
    assert case.transformed_diffusion == 24


@pytest.mark.parametrize("representation", ["expanded", "native-sum", "vector-sum"])
@pytest.mark.parametrize("kind", ["arithmetic", "geometric", "affine"])
def test_exact_coefficient_classification_is_invariant_under_finite_materialization(representation, kind):
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    i = Index("i")
    slope, intercept, noise = {
        "arithmetic": (0, 6, S(3)),
        "geometric": (6, 0, 3 * x),
        "affine": (-1, 2, S(2)),
    }[kind]
    expected_mean, expected_variance = {
        "arithmetic": (8, 9),
        "geometric": (2 * exp(6), 4 * exp(12) * (exp(9) - 1)),
        "affine": (2, 2 * (1 - exp(-2))),
    }[kind]

    def body(first, second):
        if representation == "vector-sum":
            return Vector(first, second).sum()
        if representation == "native-sum":
            return Sum(first + (second - first) * i, (i, 0, 1))
        return first + second

    law = ItoProcess(x, body(slope * x, intercept), body(noise / 2, noise / 2), t)
    assert simplify(law.mean(2, 1).doit() - expected_mean) == 0
    assert simplify(law.variance(2, 1).doit() - expected_variance) == 0


def test_populated_laws_keep_symbolic_branches_until_parameter_observations_are_applied(solve_model):
    from sympy import Piecewise

    class Conditional(Modello):
        mode = InstanceDummy("mode", real=True)
        x = InstanceDummy("x", real=True)
        t = Coordinate("t", real=True)
        unsupported = ItoProcess(x, -(x**3), 1, t)
        coefficient = Piecewise((2, mode > 0), (unsupported.mean(1, 1), True))
        process = ItoProcess(x, coefficient, 1, t)
        mean = process.mean(1, 1)

    case = solve_model(Conditional, mode=1, x=0)
    assert case.mean == 3
    assert case.process.mean(1, 1).doit() == 3
