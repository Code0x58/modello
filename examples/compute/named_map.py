"""Immutable named numeric domains with projection and exact-name alignment."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, TypeVar

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
