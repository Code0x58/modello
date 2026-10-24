"""Structural input types, independent of declarations and solving."""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from fractions import Fraction
from typing import Any, Protocol, TypeAlias


class SymbolicScalar(Protocol):
    """A scalar expression accepted by the constructor's input annotations."""

    @property
    def free_symbols(self) -> set[Any]:
        """Return the expression's free symbols."""
        ...

    @property
    def is_commutative(self) -> bool | None:
        """Return the expression's commutativity assumption."""
        ...


class SymbolicCondition(Protocol):
    """A symbolic relation or Boolean accepted by solution-family refinement."""

    @property
    def free_symbols(self) -> set[Any]:
        """Return the condition's free symbols."""
        ...

    def as_set(self) -> object:
        """Return the condition's set representation, when available."""
        ...


ScalarValue: TypeAlias = SymbolicScalar | int | float | complex | str | Decimal | Fraction


class SymbolicVector(Protocol):
    """Shape and iteration supported by native symbolic vector inputs.

    Runtime conversion also checks the supported container type and dimensions.
    """

    @property
    def shape(self) -> tuple[Any, ...]: ...

    def __iter__(self) -> Iterator[ScalarValue | None]: ...
