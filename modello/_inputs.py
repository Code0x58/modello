"""Capture scalar and component observations in a problem's existing scope."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from sympy import Expr, Symbol, SympifyError, simplify

from modello._model import _BoundField
from modello._paths import _path_label, _path_value
from modello._problem import ConstraintKind, FieldPath, ModelConstraint
from modello.errors import ModelInputError
from modello.vectors import _vector_input

if TYPE_CHECKING:
    from modello import Modello, ScalarValue, VectorValue
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

    def capture(self, values: Mapping[str, ScalarValue | VectorValue | Modello]) -> tuple[ModelConstraint, ...]:
        """Append observations in constructor order without changing the source."""
        for attr, value in values.items():
            if attr in self.model._modello_namespace.dummies:
                self._add_scalar((attr,), value, self.bindings, attr)
            elif attr in self.model._modello_namespace.vectors:
                vector = self.model._modello_namespace.vectors[attr]
                entries = _vector_input(value, len(vector), attr)
                for i, entry in enumerate(entries):
                    if entry is not None:
                        self._add_scalar((attr, str(i)), entry, self.bindings, attr)
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
            raise ModelInputError(f"{_path_label(path)} must be a SymPy-compatible scalar") from error
        if not isinstance(value, Expr):
            raise ModelInputError(f"{_path_label(path)} must be a SymPy-compatible scalar")
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
                raise ModelInputError(f"Child instance is missing {_path_label((attr, *path))}")
        _validate_vector_values(child_model, value, (attr,))
        scope = {
            **self.bindings,
            **{dummy: self.fields[prefix + path] for path, dummy in declared_fields.items()},
            **{
                value._modello_bound_fields[source_fields[path]].symbol: self.fields[prefix + path]
                for path in declared_fields
            },
        }
        for path in declared_fields:
            scalar = _path_value(value, path)
            self._add_scalar((attr, *path), scalar, scope, attr, value._modello_bound_fields[source_fields[path]])


def _validate_vector_values(model: type[Modello], instance: Modello, prefix: FieldPath = ()) -> None:
    """Reject overwritten vector shapes before copying any of their coordinates."""
    for name, vector in model._modello_namespace.vectors.items():
        _vector_input(getattr(instance, name), len(vector), _path_label((*prefix, name)))
    for name, child in model._modello_namespace.nested_models.items():
        _validate_vector_values(child._modello_model, getattr(instance, name), (*prefix, name))
