"""Finite path equations work forward, inversely, and through numeric views."""

import pytest
from sympy import Rational

from examples.symbolic import Known
from examples.timeseries import HoldingState, LinearExposure, inventory_path, return_path, rolling_window
from modello import Vector, inspect_model
from modello.errors import ModelInputError, SolutionCountError


def test_inventory_path_retains_each_event_boundary_and_settlement():
    Inventory = inventory_path(4)
    opening = HoldingState("opening", quantity=5, cash=1000)
    result = Inventory("observed", before=opening, fills=[2, -1, None, -2], prices=[100, 110, 105, 120], cash=835)
    assert result.fills == Vector(2, -1, 3, -2)
    assert result.quantities == Vector(5, 7, 6, 9, 7)
    assert result.quantity == result.after.quantity == result.quantities[-1] == 7
    assert result.cash == result.after.cash == 835
    assert result.turnover == 8
    assert opening.quantity == 5 and opening.cash == 1000


def test_compiled_inventory_retains_vectors_components_and_the_concrete_type():
    Inventory = inventory_path(3)
    case = Inventory("symbolic", before=HoldingState("opening", quantity=10, cash=100), prices=[2, 3, 4])
    view = case.compile()
    result = view.evaluate(*(Known(field, value) for field, value in zip(case.fills, (1.0, 2.0, -3.0), strict=True)))
    assert type(result) is Inventory
    assert isinstance(result.before, HoldingState)
    assert tuple(map(float, result.quantities)) == (10, 11, 13, 10)
    assert float(result.cash) == 104
    assert case.fills.free_symbols
    assert len(tuple(view.walk(result))) == len(type(case)._modello_definition.fields)


def test_returns_compound_in_order_and_allow_an_inverse_observation():
    Returns = return_path(3)
    case = Returns("forward", before=100, returns=[Rational(1, 10), Rational(-1, 20), Rational(1, 5)])
    assert case.wealth == Vector(100, 110, Rational(209, 2), Rational(627, 5))
    assert case.after == Rational(627, 5)
    assert case.total_return == Rational(127, 500)
    inverse = Returns("inverse", before=100, returns=[Rational(1, 10), None, Rational(1, 5)], after=case.after)
    assert inverse.returns == case.returns
    family = inspect_model(Returns, "family", before=100, returns=[Rational(1, 10), None, Rational(1, 5)]).solutions()
    assert family.observe(after=case.after).one().returns == case.returns


@pytest.mark.parametrize("bad_return", [-1, -2])
def test_positive_gross_factors_exclude_total_loss_and_negative_wealth(bad_return):
    with pytest.raises(SolutionCountError):
        return_path(2)("bad", before=100, returns=[0, bad_return])


def test_rolling_means_use_complete_trailing_windows_and_can_infer_a_sample():
    Window = rolling_window(5, 3)
    case = Window("forward", samples=[1, 2, 3, 4, 5])
    assert case.means == Vector(2, 3, 4)
    inverse = Window("inverse", samples=[1, 2, None, 4, 5], means=[None, 3, None])
    assert inverse.samples == case.samples
    assert inverse.means == case.means
    assert rolling_window(3, 1)("identity", samples=[1, 2, 3]).means == Vector(1, 2, 3)


def test_continuous_exposure_integrates_the_declared_interpolation():
    forward = LinearExposure("forward", opening=2, closing=4, duration=3)
    assert forward.exposure == 9
    inverse = LinearExposure("inverse", opening=2, exposure=9, duration=3)
    assert inverse.closing == 4
    assert inspect_model(LinearExposure, "family", opening=2, exposure=9, duration=3).solutions().one().closing == 4


@pytest.mark.parametrize("periods", [0, -1, True, 1.5])
def test_path_factories_require_positive_finite_shapes(periods):
    for factory in (inventory_path, return_path):
        with pytest.raises(ValueError, match="positive integers"):
            factory(periods)
    with pytest.raises(ValueError):
        rolling_window(3, periods)


def test_rolling_width_cannot_exceed_the_observation_count():
    with pytest.raises(ValueError, match="width"):
        rolling_window(2, 3)


def test_compilation_rejects_an_overwritten_vector_shape():
    source = rolling_window(3, 2)("source", samples=[1, 2, 3])
    source.samples = Vector(1, 2)
    with pytest.raises(ModelInputError, match="samples"):
        source.compile()
