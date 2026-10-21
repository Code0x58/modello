# mypy: warn-unused-ignores
"""Named data, observation axes, causal sampling, and provider request identity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pytest

from examples.data import NamedMap, Provider, Series, load_providers


def test_membership_is_owned_and_read_only():
    source = {"left": 1, "right": 2}
    members = NamedMap(source)
    source["left"] = 99
    source["new"] = 3
    assert members == {"left": 1, "right": 2}
    with pytest.raises(TypeError):
        members["left"] = 10


def test_projection_keeps_identity_and_declaration_order():
    members = NamedMap({"second": 2, "first": 1})
    projected = members.map(lambda value: str(value * 10))
    assert tuple(projected) == ("second", "first")
    assert projected == {"second": "20", "first": "10"}


def test_combinations_align_by_name_not_position():
    left = NamedMap({"a": 10, "b": 20})
    right = NamedMap({"b": 2, "a": 1})
    combined = left.combine(right, lambda first, second: first - second)
    assert tuple(combined.items()) == (("a", 9), ("b", 18))
    assert combined == {"b": 18, "a": 9}


@pytest.mark.parametrize("other", [{"a": 1}, {"a": 1, "b": 2, "c": 3}, {"a": 1, "c": 3}])
def test_combinations_require_the_same_domain(other):
    with pytest.raises(ValueError, match="Member names differ"):
        NamedMap({"a": 1, "b": 2}).combine(other, lambda a, b: a + b)


@pytest.mark.parametrize("name", ["", 1, None])
def test_member_names_are_nonempty_strings(name):
    with pytest.raises(ValueError, match="nonempty strings"):
        NamedMap([(name, 1)])


def test_duplicate_names_are_not_silently_overwritten():
    with pytest.raises(ValueError, match="Duplicate member name 'same'"):
        NamedMap([("same", 1), ("same", 2)])


@pytest.mark.parametrize(
    "dates, values",
    [
        ((date(2026, 1, 1),), (1, 2)),
        ((date(2026, 1, 1), date(2026, 1, 1)), (1, 2)),
        ((date(2026, 1, 2), date(2026, 1, 1)), (1, 2)),
        ((date(2026, 1, 1),), (float("nan"),)),
        ((date(2026, 1, 1),), (float("inf"),)),
    ],
)
def test_invalid_vectors(dates, values):
    """Malformed observation axes and nonfinite data are rejected before calculation."""
    with pytest.raises(ValueError):
        Series(dates, values)


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
    for operation in (series.lag, series.differences, series.rolling_mean):
        with pytest.raises(ValueError, match="positive integer"):
            operation(window)


def test_empty_series_is_a_valid_event_projection():
    empty = Series((), ())
    assert empty.map(lambda value: value + 1).values == ()
    assert empty.drawdowns().values == ()
    assert empty.differences() == empty.cumulative_sum() == empty
    with pytest.raises(ValueError, match="No observation"):
        empty.asof(date(2026, 1, 1))


@pytest.mark.parametrize("timestamps", [False, True], ids=("dates", "timestamps"))
def test_changes_and_accumulation_preserve_interval_and_observation_axes(timestamps):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) if timestamps else date(2026, 1, 1)
    times = tuple(start + timedelta(days=day) for day in (0, 1, 4, 8))
    series = Series(times, (100, 110, 95, 120))
    changes = series.differences()
    assert changes == Series(times[1:], (10, -15, 25))
    assert series.differences(2) == Series(times[2:], (-5, 10))
    assert series.differences(4) == series.differences(10) == Series((), ())
    # Accumulating changes reconstructs levels after the opening observation,
    # without inserting observations between dates or inventing an opening time.
    assert changes.cumulative_sum().map(lambda value: value + 100) == Series(times[1:], (110, 95, 120))
    assert Series(times[:1], (100,)).differences() == Series((), ())


def test_accumulation_rejects_nonfinite_results():
    series = Series((date(2026, 1, 1), date(2026, 1, 2)), (1e308, 1e308))
    with pytest.raises(ValueError, match="finite"):
        series.cumulative_sum()


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


def _series_typing_contract(timestamps: Series[datetime], dates: Series[date]) -> None:
    timestamp_axis: tuple[datetime, ...] = timestamps.times
    assert timestamp_axis
    _wrong_axis: tuple[datetime, ...] = dates.times  # type: ignore[assignment]
    timestamps.combine(dates, lambda a, b: a + b)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TextProvider(Provider[str]):
    text: str

    def load(self) -> str:
        return self.text


def test_loading_reuses_equivalent_requests_and_preserves_member_names():
    calls = []

    @dataclass(frozen=True)
    class RecordingProvider(TextProvider):
        def load(self) -> str:
            calls.append(self.text)
            return super().load()

    requests = NamedMap(
        {
            "first": RecordingProvider("same"),
            "second": RecordingProvider("same"),
            "third": RecordingProvider("different"),
        }
    )
    assert load_providers(requests) == {"first": "same", "second": "same", "third": "different"}
    assert calls == ["same", "different"]
    assert load_providers(requests) == {"first": "same", "second": "same", "third": "different"}
    assert calls == ["same", "different", "same", "different"]
    assert load_providers(NamedMap({})) == {}


def test_provider_type_is_part_of_request_equality():
    class OtherProvider(TextProvider):
        # Even permissive custom equality cannot conflate provider types.
        def __eq__(self, other: object) -> bool:
            return isinstance(other, TextProvider) and self.text == other.text

        __hash__ = TextProvider.__hash__

        def load(self) -> str:
            return "other: " + self.text

    one, two = TextProvider("id"), OtherProvider("id")
    assert one == two
    requests = NamedMap({"one": one, "two": two})
    assert load_providers(requests) == {"one": "id", "two": "other: id"}


def test_provider_errors_propagate_without_becoming_missing_data():
    class FailedProvider(Provider[str]):
        def load(self) -> str:
            raise OSError("source unavailable")

    with pytest.raises(OSError, match="source unavailable"):
        load_providers(NamedMap({"one": FailedProvider()}))


def _provider_typing_contract() -> None:
    provider: Provider[str] = TextProvider("value")
    result: str = provider.load()
    results: NamedMap[str] = load_providers(NamedMap({"one": provider}))
    assert result == results["one"]
    _wrong_provider: Provider[int] = provider  # type: ignore[assignment]
    _wrong_result: int = results["one"]  # type: ignore[assignment]
