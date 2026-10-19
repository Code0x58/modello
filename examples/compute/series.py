"""Small numeric series with explicit alignment and causal time operations."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import isfinite
from statistics import mean
from typing import Generic, TypeVar, cast

Time = TypeVar("Time", date, datetime)


def utc_time(value: datetime) -> datetime:
    """Require an aware timestamp and normalize it to UTC."""
    if value.utcoffset() is None:
        raise ValueError("Timestamps must include a timezone")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Series(Generic[Time]):
    """Finite observations on a unique, increasing date or UTC timestamp axis."""

    times: tuple[Time, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Own the data and validate the axis independently of its values."""
        times = tuple(self.times)
        if any(isinstance(t, datetime) for t in times):
            if not all(isinstance(t, datetime) for t in times):
                raise ValueError("Use either dates or timestamps on one axis")
            times = cast(tuple[Time, ...], tuple(utc_time(t) for t in times if isinstance(t, datetime)))
        elif not all(isinstance(t, date) for t in times):
            raise ValueError("An observation axis needs dates or timestamps")
        values = tuple(float(value) for value in self.values)
        if len(times) != len(values):
            raise ValueError("A series needs one value per observation time")
        if any(left >= right for left, right in zip(times, times[1:])):
            raise ValueError("Observation times must be unique and strictly increasing")
        if not all(isfinite(value) for value in values):
            raise ValueError("Series values must be finite")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    def map(self, operation: Callable[[float], float]) -> Series[Time]:
        """Apply a pointwise calculation without changing its axis."""
        return Series(self.times, tuple(operation(value) for value in self.values))

    def combine(self, other: Series[Time], operation: Callable[[float, float], float]) -> Series[Time]:
        """Combine exactly aligned observations."""
        if self.times != other.times:
            raise ValueError("Series must have the same observation times")
        return Series(self.times, tuple(operation(a, b) for a, b in zip(self.values, other.values, strict=True)))

    def lag(self, periods: int = 1) -> Series[Time]:
        """Put earlier values on later timestamps, omitting the warmup prefix."""
        self._window(periods)
        return Series(self.times[periods:], self.values[: max(0, len(self.values) - periods)])

    def rolling_mean(self, window: int) -> Series[Time]:
        """Use trailing full windows, labelled with their final observation."""
        self._window(window)
        return Series(
            self.times[window - 1 :],
            tuple(mean(self.values[i - window + 1 : i + 1]) for i in range(window - 1, len(self.values))),
        )

    @staticmethod
    def _window(periods: int) -> None:
        if isinstance(periods, bool) or not isinstance(periods, int) or periods <= 0:
            raise ValueError("A window must be a positive integer")

    def asof(self, at: Time, *, max_age: timedelta | None = None) -> float:
        """Select the latest earlier observation, with an optional freshness limit."""
        if isinstance(at, datetime):
            at = cast(Time, utc_time(at))
        index = bisect_right(self.times, at) - 1
        if index < 0:
            raise ValueError(f"No observation is available at {at}")
        if max_age is not None and (max_age < timedelta(0) or at - self.times[index] > max_age):
            raise ValueError(f"Observation at {at} is stale")
        return self.values[index]

    def sample(self, times: tuple[Time, ...], *, max_age: timedelta | None = None) -> Series[Time]:
        """Explicitly sample a new axis using only earlier observations."""
        return Series(times, tuple(self.asof(at, max_age=max_age) for at in times))

    def simple_returns(self) -> Series[Time]:
        """Label price ratios minus one with their interval's ending time."""
        if len(self.values) < 2 or any(value <= 0 for value in self.values):
            raise ValueError("Returns need at least two strictly positive observations")
        return Series(self.times[1:], tuple(end / start - 1 for start, end in zip(self.values, self.values[1:])))

    def drawdowns(self) -> Series[Time]:
        """Measure the fraction below the running peak."""
        if any(value <= 0 for value in self.values):
            raise ValueError("Drawdowns need strictly positive observations")
        peak = 0.0
        values = []
        for value in self.values:
            peak = max(peak, value)
            values.append(value / peak - 1)
        return Series(self.times, tuple(values))
