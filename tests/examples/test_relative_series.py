"""Discrete and continuous examples use the same flat observations and adapters."""

from sympy import Rational, exp, log, simplify

from examples.relative_series import (
    BrownianCompensator,
    GeometricDiffusion,
    MeanReversion,
    RelativeGrowth,
    autoregressive_path,
)
from examples.symbolic import Known
from modello import Vector, inspect_model


def test_one_growth_law_supports_different_queries_and_a_separate_window():
    case = RelativeGrowth("long", opening=1, ratio=2, periods=10)
    assert case.closing == 1024
    assert case.samples == Vector(1, 2, 4, 8, 16)
    assert case.law.at(20).doit() == 2**20


def test_relative_innovation_path_can_infer_an_earlier_unknown_from_a_later_boundary():
    Path = autoregressive_path(4)
    case = Path("inverse", opening=2, retention=Rational(1, 2), innovations=[1, 0, None, 0], closing=Rational(5, 4))
    assert case.innovations == Vector(1, 0, 2, 0)
    assert case.levels == Vector(2, 2, 1, Rational(5, 2), Rational(5, 4))
    assert autoregressive_path(0)("idle", opening=2, retention=1, innovations=[]).levels == Vector(2)


def test_geometric_law_can_infer_parameters_without_a_separate_calibrator():
    case = GeometricDiffusion(
        "inferred", opening=100, duration=1, wiener_increment=0, log_drift=Rational(2, 25), log_diffusion=Rational(1, 5)
    )
    assert case.growth == Rational(1, 10)
    assert case.volatility == Rational(1, 5)
    assert case.closing == 100 * exp(Rational(2, 25))
    assert case.mean == 100 * exp(Rational(1, 10))
    assert inspect_model(
        GeometricDiffusion,
        "family",
        opening=100,
        duration=1,
        wiener_increment=0,
        log_drift=Rational(2, 25),
        log_diffusion=Rational(1, 5),
    ).solutions().one().growth == Rational(1, 10)


def test_mean_reversion_and_brownian_compensation_use_plain_model_inputs():
    case = MeanReversion("mean", rate=log(2), equilibrium=4, volatility=2, opening=1, duration=1)
    assert case.mean == Rational(5, 2)
    assert simplify(case.variance - Rational(3, 2) / log(2)) == 0
    assert BrownianCompensator("zero drift", drift=0).rate == 1


def test_compiled_views_keep_law_parameters_in_step_with_numeric_fields():
    source = RelativeGrowth("compiled", opening=1, periods=4)
    view = source.compile()
    result = view.evaluate(Known(source.ratio, 2))
    assert float(result.closing) == 16
    assert float(result.law.at(6).doit()) == 64
    assert result.law.target.series == source.law.target.series
