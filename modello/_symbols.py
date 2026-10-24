"""Independent quantity identities and local indices, without model machinery."""

from __future__ import annotations

from typing import Any

from sympy import Dummy, Symbol

from modello.errors import ModelInputError


class InstanceDummy(Dummy):
    """A symbol that is rebound independently for each model instance."""

    def bound(self, model_name: str) -> BoundInstanceDummy:
        """Return a fresh symbol with this symbol's assumptions."""
        return BoundInstanceDummy(model_name + "_" + self.name, **self.assumptions0)


class BoundInstanceDummy(InstanceDummy):
    """A symbol belonging to a particular model instance."""


class Coordinate(Dummy):
    """A local coordinate, excluded from observable model quantities.

    Use ``Coordinate('t', real=True)`` for continuous time and ``Index`` for
    discrete positions. Coordinates retain independent identities and may be
    used as differentiation variables or locally bound formula variables.
    """


class Index(Coordinate):
    """A local integer index, rather than an observable model quantity.

    Declare ``i = Index('i')`` inside a model and use it in SymPy limits such
    as ``Sum(x[i], (i, 0, len(x) - 1))``. Each declaration has its own identity.
    Limits bind it in the usual SymPy way; an unbound index remains symbolic.
    """

    def __new__(cls, name: str = "i", dummy_index: int | None = None, **assumptions: Any) -> Index:
        if assumptions.get("integer", True) is not True:
            raise ModelInputError("An Index must be an integer")
        assumptions["integer"] = True
        return super().__new__(cls, name, dummy_index=dummy_index, **assumptions)


def _minimal_assumptions(symbol: Symbol) -> dict[str, bool]:
    """Discard requirements implied by the remaining declared assumptions."""
    requirements = {key: value for key, value in symbol.assumptions0.items() if key != "commutative"}
    for key in sorted(requirements):
        others = {name: value for name, value in requirements.items() if name != key}
        if Symbol("requirements", **others).assumptions0.get(key) == requirements[key]:
            del requirements[key]
    return requirements
