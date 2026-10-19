"""Compile views of ordinary Modello models without adding execution to the core."""

from __future__ import annotations

from functools import cache
from typing import ClassVar, TypeAlias, cast

from sympy import Expr

from examples.compute.named_map import NamedMap
from examples.compute.scalar_plan import ScalarPlan
from examples.compute.series import Series, Time
from modello import Modello

ViewFields: TypeAlias = tuple[str, ...] | NamedMap[Expr]


class SymbolicModel(Modello):
    """One equation definition, with different knowns, unknowns, and runtime views."""

    inputs: ClassVar[tuple[str, ...]] = ()
    outputs: ClassVar[tuple[str, ...]] = ()

    def compile(self, *, inputs: ViewFields | None = None, outputs: ViewFields | None = None) -> ScalarPlan:
        """Compile this solved or partially bound case's selected remaining inputs."""
        input_names = self.inputs if inputs is None else inputs
        output_names = self.outputs if outputs is None else outputs
        return ScalarPlan(
            input_names
            if isinstance(input_names, NamedMap)
            else NamedMap((name, cast(Expr, getattr(self, name))) for name in input_names),
            output_names
            if isinstance(output_names, NamedMap)
            else NamedMap((name, cast(Expr, getattr(self, name))) for name in output_names),
        )

    @classmethod
    @cache
    def plan(cls) -> ScalarPlan:
        """Reuse the class's declared forward view; inverse cases use the same constructor."""
        return cls(cls.__name__).compile()

    @classmethod
    def over(cls, times: tuple[Time, ...], values: NamedMap[Series[Time]]) -> NamedMap[Series[Time]]:
        """Evaluate the model's declared view on an explicit observation axis."""
        return cls.plan().over(times, values)
