"""Capture scalar and component observations in a problem's existing scope."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from sympy import Expr, Symbol, SympifyError, simplify

from modello._model import _BoundField
from modello._problem import ConstraintKind, FieldPath, ModelConstraint
from modello.errors import ModelInputError

if TYPE_CHECKING:
    from modello import Modello, ScalarValue
    from modello.diagnostics import ModelProblem


class _InputCapture:
    """Collect observations while retaining aliases, origins, and input groups."""

    def __init__(self, problem: ModelProblem) -> None:
        self.model = problem.model
        self.fields = {field.path: field.symbol for field in problem.fields}
        self.bindings: dict[Symbol, Symbol] = {
            dummy: self.fields[path] for path, dummy in self.model._modello_definition.fields.items()
        }
        self.constraints = list(problem.constraints)

    def capture(self, values: Mapping[str, ScalarValue | Modello]) -> tuple[ModelConstraint, ...]:
        """Append observations in constructor order without changing the source."""
        for attr, value in values.items():
            if attr in self.model._modello_namespace.dummies:
                self._add_scalar((attr,), value, self.bindings, attr)
            else:
                self._add_component(attr, value)
        return tuple(self.constraints)

    def _add_scalar(
        self,
        path: FieldPath,
        raw: object,
        scope: Mapping[Symbol, Symbol],
        group: str,
        source: _BoundField | None = None,
    ) -> None:
        """Validate a scalar, rebind its expression, and record its provenance."""
        try:
            value = simplify(raw)
        except (SympifyError, TypeError, AttributeError) as error:
            raise ModelInputError(f"{'.'.join(path)} must be a SymPy-compatible scalar") from error
        if not isinstance(value, Expr):
            raise ModelInputError(f"{'.'.join(path)} must be a SymPy-compatible scalar")
        kind = ConstraintKind.INPUT if source is None else source.copied_kind(raw)
        self.constraints.append(
            ModelConstraint(len(self.constraints), path, kind, self.fields[path], value.xreplace(scope), group)
        )

    def _add_component(self, attr: str, value: object) -> None:
        """Copy the declared child shape into an independent parent scope."""
        child = self.model._modello_namespace.nested_models.get(attr)
        if child is None:
            raise ModelInputError(f"Unknown model attribute {attr!r}")
        child_model = child._modello_model
        if not isinstance(value, child_model):
            raise ModelInputError(f"{attr} must be a {child_model.__name__} instance")
        prefix = (attr,)
        declared_fields = child_model._modello_definition.fields
        source_fields = type(value)._modello_definition.fields
        for path in declared_fields:
            if path not in source_fields:
                raise ModelInputError(f"Child instance is missing {'.'.join((attr, *path))}")
        scope = {
            **self.bindings,
            **{dummy: self.fields[prefix + path] for path, dummy in declared_fields.items()},
            **{
                value._modello_bound_fields[source_fields[path]].symbol: self.fields[prefix + path]
                for path in declared_fields
            },
        }
        for path in declared_fields:
            scalar: object = value
            for part in path:
                scalar = getattr(scalar, part)
            self._add_scalar((attr, *path), scalar, scope, attr, value._modello_bound_fields[source_fields[path]])
