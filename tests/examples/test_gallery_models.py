"""Each gallery model demonstrates a distinct relationship or integration."""

from operator import attrgetter

import pytest
from sympy import Rational, S, Tuple, exp

from examples.gallery.models import (
    AccountSequence,
    GeometricDiffusion,
    LinearExposure,
    LinearTrend,
    ObservedAutoregression,
    VectorSummary,
)
from examples.portfolio.models import Execution, HoldingState
from examples.symbolic import Known, SymbolicModel
from modello import ModelObservations, Vector, inspect_model
from modello.errors import ModelInputError


@pytest.mark.parametrize(
    "model, observations, expected",
    [
        (
            VectorSummary,
            {"samples": [1, None, 3], "total": 6},
            {"N": 3, "samples": Vector(1, 2, 3), "product": 6, "squared_norm": 14},
        ),
        (VectorSummary, {"samples": []}, {"N": 0, "total": 0, "product": 1, "squared_norm": 0}),
        (
            ObservedAutoregression,
            {"opening": 2, "retention": Rational(1, 2), "innovations": [1, 0, None, 0], "closing": Rational(5, 4)},
            {"N": 4, "innovations": Vector(1, 0, 2, 0)},
        ),
        (LinearExposure, {"opening": 2, "duration": 3, "exposure": 9}, {"closing": 4}),
        (
            LinearTrend,
            {"times": [0, 1, 2, 3], "observations": [1, 2, 2, 4], "slope_gradient": 0, "intercept_gradient": 0},
            {"N": 4, "slope": Rational(9, 10), "intercept": Rational(9, 10), "squared_error": Rational(7, 10)},
        ),
        (
            GeometricDiffusion,
            {
                "opening": 100,
                "duration": 1,
                "wiener_increment": 0,
                "log_drift": Rational(2, 25),
                "log_diffusion": Rational(1, 5),
            },
            {
                "growth": Rational(1, 10),
                "volatility": Rational(1, 5),
                "closing": 100 * exp(Rational(2, 25)),
                "mean": 100 * exp(Rational(1, 10)),
            },
        ),
    ],
    ids=["vector-inverse", "empty-vector", "relative-inverse", "integral-inverse", "gradient-fit", "ito-inverse"],
)
def test_gallery_models_use_plain_observations_for_forward_and_inverse_solves(model, observations, expected):
    case = model("example", **observations)
    for path, value in expected.items():
        assert attrgetter(path)(case) == value


def test_repeated_calibration_times_keep_the_underdetermined_line_family():
    family = inspect_model(
        LinearTrend, "singular", times=[1, 1, 1], observations=[1, 2, 3], slope_gradient=0, intercept_gradient=0
    ).solutions()
    assert len(family.parameters) == 1
    assert family.expression(family.symbols.slope + family.symbols.intercept) == 2
    assert family.expression(family.symbols.squared_error) == 2
    assert family.observe(slope=0).one().intercept == 2
    assert family.observe(slope=1).one().intercept == 1
    assert family.values(family.symbols.fitted).contains(Tuple(2, 2, 2)) is S.true


def test_longer_calibration_retains_every_vector_output_and_exact_gradient():
    size = 128
    family = inspect_model(
        LinearTrend,
        "longer",
        times=list(range(size)),
        observations=[2 * i + 1 for i in range(size)],
        slope_gradient=0,
        intercept_gradient=0,
    ).solutions()
    assert family.parameters == ()
    assert family.expression(family.symbols.slope) == 2
    assert family.expression(family.symbols.intercept) == 1
    assert family.expression(family.symbols.residuals) == Vector([0] * size)
    assert family.expression(family.symbols.fitted) == Vector([2 * i + 1 for i in range(size)])


@pytest.fixture
def account_observations():
    return {
        "states": ModelObservations({0: HoldingState("opening", quantity=3, cash=1000)}),
        "fills": [Execution("buy", quantity=2, price=100, fee=1), Execution("sell", quantity=-1, fee=1)],
    }


def test_numerical_views_preserve_nested_model_types_and_terminal_identity(account_observations):
    class NumericalSequence(AccountSequence, SymbolicModel):
        pass

    source = NumericalSequence("source", **account_observations)
    result = source.compile().evaluate(Known(source.fills[1].price, 110))
    assert isinstance(result, NumericalSequence)
    assert isinstance(result.states[-1], HoldingState)
    assert float(result.terminal.cash) == 908
    assert result.states[-1] is result.terminal
    assert result.terminal is not source.terminal
    assert source.fills[1].price.free_symbols


def test_numerical_views_rebind_law_parameters_alongside_vector_fields():
    class NumericalSequence(ObservedAutoregression, SymbolicModel):
        pass

    source = NumericalSequence("source", opening=1, innovations=[0, 0, 0])
    result = source.compile().evaluate(Known(source.retention, 2))
    assert tuple(map(float, result.levels)) == (1, 2, 4, 8)
    assert float(result.law.at(3).doit()) == 8
    source.levels = Vector(1, 2)
    with pytest.raises(ModelInputError, match="levels"):
        source.compile()
