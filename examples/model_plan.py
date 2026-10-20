"""Compile real-valued Modello cases while preserving their declared model tree."""

from __future__ import annotations

from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, Literal, TypeVar, cast

from sympy import Expr, Float, Symbol

from examples.compute.named_map import NamedMap
from examples.compute.scalar_plan import ScalarPlan
from examples.compute.series import Series, Time
from modello import FieldPath, Modello, SymbolicScalar

Model = TypeVar("Model", bound=Modello)
Value = TypeVar("Value", covariant=True)


@dataclass(frozen=True)
class Known(Generic[Value]):
    """Bind observed data to a declared field of a model case."""

    field: SymbolicScalar
    value: Value


@dataclass(frozen=True)
class ModelField:
    """A declared field, its resolved expression, and optionally its evaluated value."""

    path: FieldPath
    expression: SymbolicScalar
    role: Literal["input", "known", "derived"]
    value: float | None = None


def _get_field(model: Modello, path: FieldPath) -> Expr:
    value: object = model
    for name in path:
        value = getattr(value, name)
    if not isinstance(value, Expr):
        raise TypeError(f"{'.'.join(path)} must be a scalar expression")
    return value


def _check_column(symbol: Expr, values: tuple[float, ...]) -> None:
    """Check real numeric columns without materializing a model per observation."""
    if symbol.is_positive:
        invalid = any(value <= 0 for value in values)
    elif symbol.is_nonnegative:
        invalid = any(value < 0 for value in values)
    elif symbol.is_negative:
        invalid = any(value >= 0 for value in values)
    elif symbol.is_nonpositive:
        invalid = any(value > 0 for value in values)
    else:
        invalid = False
    if symbol.is_nonzero and any(value == 0 for value in values):
        invalid = True
    if symbol.is_integer and any(not value.is_integer() for value in values):
        invalid = True
    if invalid:
        raise ValueError("A column violates its model assumptions")


def _copy_model(model: Model, values: dict[FieldPath, Expr]) -> Model:
    """Populate a copy without solving again or changing the source case.

    Like the named-member adapter, this is isolated access to core metadata.
    Model classes, component types, provider bindings, and symbol scopes survive.
    Cached properties must be recomputed for the copied field values.
    """

    def populate(source: Modello, prefix: FieldPath) -> Modello:
        result = copy(source)
        result._modello_instance_dummies = dict(source._modello_instance_dummies)
        for name in tuple(vars(result)):
            if isinstance(getattr(type(source), name, None), cached_property):
                delattr(result, name)
        namespace = type(source)._modello_namespace
        for name in namespace.dummies:
            setattr(result, name, values[prefix + (name,)])
        for name in namespace.nested_models:
            setattr(result, name, populate(getattr(source, name), (*prefix, name)))
        return result

    return cast(Model, populate(model, ()))


class ModelPlan(Generic[Model]):
    """A numerical view whose result retains the concrete Modello model type.

    Independent symbolic fields are the default inputs. Bind knowns by field
    reference; all scalar fields are outputs. Flat column IDs are private to
    the ScalarPlan backend, and declared field paths remain inspectable.
    """

    def __init__(
        self,
        model: Model,
        *,
        inputs: tuple[SymbolicScalar, ...] | None = None,
        properties: tuple[SymbolicScalar, ...] = (),
    ) -> None:
        expressions = {path: _get_field(model, path) for path in type(model)._modello_fields}
        self.source = _copy_model(model, expressions)
        self._paths = tuple(expressions)
        self._expressions = tuple(expressions.values())
        if any(not isinstance(expression, Expr) for expression in properties):
            raise TypeError("Additional properties must be model expressions")
        self._outputs = tuple(dict.fromkeys((*self._expressions, *cast(tuple[Expr, ...], properties))))
        inferred = tuple(dict.fromkeys(expr for expr in self._expressions if isinstance(expr, Symbol)))
        selected = inferred if inputs is None else inputs
        if any(not isinstance(expr, Symbol) or expr not in inferred for expr in selected):
            raise ValueError("Inputs must be independent symbolic fields of this model case")
        self._inputs = cast(tuple[Expr, ...], selected)
        self._input_columns = NamedMap((str(i), expr) for i, expr in enumerate(self._inputs))
        self._plan = ScalarPlan(
            self._input_columns,
            NamedMap((str(i), expr) for i, expr in enumerate(self._outputs)),
        )
        self._property_plans: dict[Expr, ScalarPlan] = {}

    def _property_plan(self, expression: Expr) -> ScalarPlan:
        if expression not in self._property_plans:
            self._property_plans[expression] = ScalarPlan(self._input_columns, NamedMap({"value": expression}))
        return self._property_plans[expression]

    def _align(self, knowns: tuple[Known[Value], ...]) -> NamedMap[Value]:
        supplied: dict[Expr, Value] = {}
        for known in knowns:
            if not isinstance(known, Known) or not isinstance(known.field, Expr) or known.field not in self._inputs:
                raise ValueError("Supply Known values for this view's input fields")
            if known.field in supplied:
                raise ValueError("An input field was supplied more than once")
            supplied[known.field] = known.value
        if set(supplied) != set(self._inputs):
            raise ValueError("Supply every input field exactly once")
        return NamedMap((name, supplied[expr]) for name, expr in self._input_columns.items())

    def _materialize(self, values: tuple[float, ...]) -> Model:
        evaluated = dict(zip(self._outputs, map(Float, values), strict=True))
        fields = {path: evaluated[expression] for path, expression in zip(self._paths, self._expressions, strict=True)}
        for path, value in fields.items():
            dummy = type(self.source)._modello_fields[path]
            if any(
                getattr(value, f"is_{assumption}", None) not in (None, expected)
                for assumption, expected in dummy.assumptions0.items()
            ):
                raise ValueError(f"Evaluated field {'.'.join(path)} violates its model assumptions")
        return _copy_model(self.source, fields)

    def evaluate(self, *knowns: Known[float]) -> Model:
        """Evaluate once and return a fresh numeric case with the same component types."""
        inputs = self._align(knowns)
        try:
            values = self._plan.evaluate(inputs)
        except ArithmeticError as error:
            raise ValueError("Values violate model assumptions or leave expressions undefined") from error
        return self._materialize(tuple(values.values()))

    def over(self, times: tuple[Time, ...], *knowns: Known[Series[Time]]) -> ModelPath[Model, Time]:
        """Evaluate aligned columns once; materialize model cases only when requested."""
        inputs = self._align(knowns)
        for name, symbol in self._input_columns.items():
            _check_column(symbol, inputs[name].values)
        try:
            outputs = self._plan.over(times, inputs)
        except ArithmeticError as error:
            raise ValueError("Values violate model assumptions or leave expressions undefined") from error
        columns = dict(zip(self._outputs, outputs.values(), strict=True))
        for path, expression in zip(self._paths, self._expressions, strict=True):
            _check_column(type(self.source)._modello_fields[path], columns[expression].values)
        return ModelPath(self, times, inputs, tuple(outputs.values()))

    def walk(self, result: Model | None = None) -> Iterator[ModelField]:
        """Traverse actual declared paths with source expressions and evaluated values."""
        if result is not None and type(result) is not type(self.source):
            raise TypeError("Inspect a result with the same model type as the compiled view")
        for path, expression in zip(self._paths, self._expressions, strict=True):
            role: Literal["input", "known", "derived"] = (
                "input" if expression in self._inputs else "derived" if expression.free_symbols else "known"
            )
            value = float(_get_field(result, path)) if result is not None else None
            yield ModelField(path, expression, role, value)


@dataclass(frozen=True)
class ModelPath(Generic[Model, Time]):
    """A compiled model's columns and observation axis, retaining its source schema."""

    view: ModelPlan[Model]
    times: tuple[Time, ...]
    _inputs: NamedMap[Series[Time]]
    _columns: tuple[Series[Time], ...]

    def at(self, index: int) -> Model:
        """Materialize a typed model case without invoking the symbolic solver."""
        if not -len(self.times) <= index < len(self.times):
            raise IndexError("Model path observation index out of range")
        return self.view._materialize(tuple(column.values[index] for column in self._columns))

    @cached_property
    def _derived_columns(self) -> dict[Expr, Series[Time]]:
        return {}

    def column(self, field: SymbolicScalar) -> Series[Time]:
        """Select by model reference, also allowing derived model properties."""
        if not isinstance(field, Expr):
            raise TypeError("Select a model expression, not a string column name")
        for expression, column in zip(self.view._outputs, self._columns, strict=True):
            if expression == field:
                return column
        # Properties such as contextual holding weights reuse the same input
        # columns and expressions; no financial formula is reconstructed here.
        if field not in self._derived_columns:
            plan = self.view._property_plan(field)
            self._derived_columns[field] = plan.over(self.times, self._inputs)["value"]
        return self._derived_columns[field]
