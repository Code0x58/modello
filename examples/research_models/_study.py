"""Presentation records shared by the separately executable model studies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from modello.contrib.research import ResearchCase, ResearchError, ResearchReport, ResearchResult


@dataclass(frozen=True)
class Computation:
    result: ResearchResult
    kind: str = "exact scalar checks"


@dataclass(frozen=True)
class Blocked:
    report: ResearchReport
    message: str

    @classmethod
    def from_error(cls, error: ResearchError) -> Blocked:
        return cls(error.report, str(error))


@dataclass(frozen=True)
class Variant:
    """One immutable research branch and the methods attempted on it."""

    case: ResearchCase
    computations: Mapping[str, Computation] = field(default_factory=dict)
    blocked: Mapping[str, Blocked] = field(default_factory=dict)
    observations: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("computations", "blocked", "observations"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True)
class Study:
    key: str
    title: str
    description: str
    sources: tuple[str, ...]
    variants: Mapping[str, Variant]
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "variants", MappingProxyType(dict(self.variants)))
