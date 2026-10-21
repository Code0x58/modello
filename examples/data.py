"""Named data, causal series, and explicit provider requests; independent of Modello."""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from itertools import accumulate
from math import isfinite
from statistics import mean
from types import MappingProxyType
from typing import Generic, TypeVar, cast

_Value = TypeVar("_Value", covariant=True)


_Other = TypeVar("_Other")


_Result = TypeVar("_Result")


@dataclass(frozen=True, init=False, eq=False)
class NamedMap(Mapping[str, _Value], Generic[_Value]):
    """Own named members; projections preserve names and combinations align them."""

    _members: Mapping[str, _Value]

    def __init__(self, members: Mapping[str, _Value] | Iterable[tuple[str, _Value]]) -> None:
        """Copy membership, rejecting duplicate or empty names."""
        pairs = members.items() if isinstance(members, Mapping) else members
        owned = {}
        for name, value in pairs:
            if not isinstance(name, str) or not name:
                raise ValueError("Member names must be nonempty strings")
            if name in owned:
                raise ValueError(f"Duplicate member name {name!r}")
            owned[name] = value
        object.__setattr__(self, "_members", MappingProxyType(owned))

    def __getitem__(self, name: str) -> _Value:
        """Look up one member by identity."""
        return self._members[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate names in declaration order."""
        return iter(self._members)

    def __len__(self) -> int:
        """Return the number of named members."""
        return len(self._members)

    def map(self, project: Callable[[_Value], _Result]) -> NamedMap[_Result]:
        """Project each value onto the same named domain."""
        return NamedMap((name, project(value)) for name, value in self.items())

    def combine(self, other: Mapping[str, _Other], operation: Callable[[_Value, _Other], _Result]) -> NamedMap[_Result]:
        """Combine matching names, requiring exactly the same member domain."""
        missing = self.keys() - other.keys()
        extra = other.keys() - self.keys()
        if missing or extra:
            raise ValueError(f"Member names differ: missing {sorted(missing)}, extra {sorted(extra)}")
        return NamedMap((name, operation(value, other[name])) for name, value in self.items())


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

    def differences(self, periods: int = 1) -> Series[Time]:
        """Subtract earlier observations, labelling changes with their ending time."""
        self._window(periods)
        return Series(
            self.times[periods:],
            tuple(end - start for start, end in zip(self.values, self.values[periods:])),
        )

    def cumulative_sum(self) -> Series[Time]:
        """Accumulate observations from zero, preserving their times."""
        return Series(self.times, tuple(accumulate(self.values)))

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


class Provider(ABC, Generic[_Value]):
    """A configured request returning one type of data when explicitly loaded.

    Providers must be hashable with stable equality during a load. Frozen
    dataclasses can identify equivalent requests by their type and parameters;
    ordinary objects retain distinct identities unless they define equality.
    """

    @abstractmethod
    def load(self) -> _Value:
        """Fetch the requested data; construction itself does not perform I/O."""


def load_providers(providers: NamedMap[Provider[_Value]]) -> NamedMap[_Value]:
    """Load each provider type and request once per batch, preserving member names.

    The cache is local to this call: a later load can observe updated sources.
    Errors propagate to the caller without substituting missing observations.
    """
    loaded: dict[tuple[type[Provider[_Value]], Provider[_Value]], _Value] = {}

    def load(provider: Provider[_Value]) -> _Value:
        key = (type(provider), provider)
        if key not in loaded:
            loaded[key] = provider.load()
        return loaded[key]

    return providers.map(load)
