"""Compile typed views of Modello cases without adding execution to the core."""

from __future__ import annotations

from functools import cache
from typing import TypeVar, cast

from examples.model_plan import ModelPlan
from modello import Modello, SymbolicScalar

Model = TypeVar("Model", bound="SymbolicModel")


class SymbolicModel(Modello):
    """Reuse one equation definition through model-preserving numerical views."""

    def compile(self: Model, *, inputs: tuple[SymbolicScalar, ...] | None = None) -> ModelPlan[Model]:
        """Compile this case's remaining independent fields and retain its model type."""
        return ModelPlan(self, inputs=inputs)

    @classmethod
    def plan(cls: type[Model]) -> ModelPlan[Model]:
        """Cache an unbound view; bound or inverse cases use their own compile method."""
        return cast(ModelPlan[Model], _model_plan(cls))


@cache
def _model_plan(model: type[SymbolicModel]) -> ModelPlan[SymbolicModel]:
    return model(model.__name__).compile()
