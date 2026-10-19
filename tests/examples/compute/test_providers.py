# mypy: warn-unused-ignores
"""Typed provider requests, per-batch reuse, and explicit refresh behavior."""

from dataclasses import dataclass

import pytest

from examples.compute.named_map import NamedMap
from examples.compute.providers import Provider, load_providers


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
        def load(self) -> str:
            return "other: " + self.text

    requests = NamedMap({"one": TextProvider("id"), "two": OtherProvider("id")})
    assert load_providers(requests) == {"one": "id", "two": "other: id"}


def test_provider_errors_propagate_without_becoming_missing_data():
    class FailedProvider(Provider[str]):
        def load(self) -> str:
            raise OSError("source unavailable")

    with pytest.raises(OSError, match="source unavailable"):
        load_providers(NamedMap({"one": FailedProvider()}))


def _typing_contract() -> None:
    provider: Provider[str] = TextProvider("value")
    result: str = provider.load()
    results: NamedMap[str] = load_providers(NamedMap({"one": provider}))
    assert result == results["one"]
    _wrong_provider: Provider[int] = provider  # type: ignore[assignment]
    _wrong_result: int = results["one"]  # type: ignore[assignment]
