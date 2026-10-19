"""Typed, explicit data acquisition, independent of models and financial data."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from examples.compute.named_map import NamedMap

_Value = TypeVar("_Value", covariant=True)


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
