# mypy: warn-unused-ignores
"""Axis semantics and the reusable expression plan, independent of finance."""

from datetime import date, datetime, timedelta, timezone

import pytest
from sympy import Symbol

from examples.named_members import NamedMap
from examples.scalar_plan import ScalarPlan
from examples.series import Series


def test_series_alignment_lags_and_trailing_windows():
    times = tuple(date(2026, 1, day) for day in range(1, 5))
    series = Series(times, (1, 2, 4, 8))
    assert series.map(lambda value: value * 2).values == (2, 4, 8, 16)
    assert series.lag().times == times[1:]
    assert series.lag().values == (1, 2, 4)
    assert series.rolling_mean(2) == Series(times[1:], (1.5, 3, 6))
    assert series.lag(10).values == ()
    assert series.rolling_mean(10).values == ()
    with pytest.raises(ValueError, match="same observation times"):
        series.combine(series.lag(), lambda a, b: a + b)


@pytest.mark.parametrize("window", [0, -1, True, 1.5])
def test_invalid_series_windows(window):
    series = Series((date(2026, 1, 1),), (1,))
    for operation in (series.lag, series.rolling_mean):
        with pytest.raises(ValueError, match="positive integer"):
            operation(window)


def test_empty_series_is_a_valid_event_projection():
    empty = Series((), ())
    assert empty.map(lambda value: value + 1).values == ()
    assert empty.drawdowns().values == ()
    assert empty.as_vector().shape == (0, 1)
    with pytest.raises(ValueError, match="No observation"):
        empty.asof(date(2026, 1, 1))


def test_sampling_is_backward_only_and_freshness_is_explicit():
    times = (date(2026, 1, 1), date(2026, 1, 3))
    series = Series(times, (10, 30))
    assert series.sample((times[0], date(2026, 1, 2), times[1])).values == (10, 10, 30)
    assert series.asof(times[1]) == 30
    with pytest.raises(ValueError, match="No observation"):
        series.asof(date(2025, 12, 31))
    with pytest.raises(ValueError, match="stale"):
        series.asof(date(2026, 1, 2), max_age=timedelta(hours=1))


def test_timestamp_axes_normalize_offsets_and_reject_naive_or_mixed_times():
    offset = timezone(timedelta(hours=2))
    at = datetime(2026, 1, 1, 12, tzinfo=offset)
    series = Series((at,), (1,))
    assert series.times == (datetime(2026, 1, 1, 10, tzinfo=timezone.utc),)
    assert series.asof(at) == 1
    with pytest.raises(ValueError, match="timezone"):
        Series((at.replace(tzinfo=None),), (1,))
    with pytest.raises(ValueError, match="timezone"):
        series.asof(at.replace(tzinfo=None))
    with pytest.raises(ValueError, match="either dates or timestamps"):
        Series((date(2026, 1, 1), at), (1, 2))


def test_named_plan_agrees_between_points_and_vectors():
    p, q = Symbol("p"), Symbol("q")
    plan = ScalarPlan(NamedMap({"price": p, "quantity": q}), NamedMap({"value": p * q, "ratio": p / q}))
    assert plan.evaluate(NamedMap({"quantity": 2, "price": 10})) == {"value": 20, "ratio": 5}
    times = (date(2026, 1, 1), date(2026, 1, 2))
    result = plan.over(times, NamedMap({"quantity": Series(times, (2, 4)), "price": Series(times, (10, 12))}))
    assert result["value"].values == (20, 48)
    assert result["ratio"].values == (5, 3)
    with pytest.raises(ValueError, match="same observation times"):
        plan.over(times, NamedMap({"quantity": Series(times, (2, 4)), "price": Series(times[:1], (10,))}))
    with pytest.raises(ValueError, match="Member names differ"):
        plan.evaluate(NamedMap({"price": 10}))
    with pytest.raises(ValueError, match="finite"):
        plan.evaluate(NamedMap({"price": float("nan"), "quantity": 2}))


def test_plan_handles_constants_and_rejects_unbound_or_aliased_inputs():
    p, q = Symbol("p"), Symbol("q")
    plan = ScalarPlan(NamedMap({}), NamedMap({"cash": 100}))
    assert plan.evaluate(NamedMap({})) == {"cash": 100}
    assert plan.over((date(2026, 1, 1),), NamedMap({}))["cash"].values == (100,)
    with pytest.raises(ValueError, match="independent symbols"):
        ScalarPlan(NamedMap({"p": p, "q": p}), NamedMap({"value": p}))
    with pytest.raises(ValueError, match="unspecified inputs"):
        ScalarPlan(NamedMap({"p": p}), NamedMap({"value": p + q}))


def _typing_contract(timestamps: Series[datetime], dates: Series[date]) -> None:
    timestamp_axis: tuple[datetime, ...] = timestamps.times
    assert timestamp_axis
    _wrong_axis: tuple[datetime, ...] = dates.times  # type: ignore[assignment]
    timestamps.combine(dates, lambda a, b: a + b)  # type: ignore[arg-type]
