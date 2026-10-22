"""Compile and inspect typed numerical views of symbolic models."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from copy import copy
from dataclasses import dataclass, replace
from functools import cache, cached_property
from math import isfinite
from typing import Generic, Literal, Protocol, TypeVar, cast, overload, runtime_checkable

from sympy import Expr, Float, Symbol, lambdify, sympify

from examples.data import NamedMap, Series, Time
from modello import ConstraintKind, FieldPath, Modello, SymbolicScalar, Vector
from modello._inputs import _validate_vector_values
from modello._paths import _path_value


class ScalarPlan:
    """Keep expression compilation separate from fetching data and reducing time."""

    def __init__(self, inputs: NamedMap[Expr], outputs: NamedMap[Expr]) -> None:
        """Compile independent symbolic inputs and completely bound outputs."""
        if not all(isinstance(value, Symbol) for value in inputs.values()) or len(set(inputs.values())) != len(inputs):
            raise ValueError("Plan inputs must be independent symbols")
        expressions = outputs.map(sympify)
        if set().union(*(expr.free_symbols for expr in expressions.values())) - set(inputs.values()):
            raise ValueError("Plan outputs contain unspecified inputs")
        self.inputs = inputs
        self.expressions = expressions
        self.outputs = tuple(outputs)
        self._calculate = cast(
            Callable[..., tuple[float, ...]],
            lambdify(tuple(inputs.values()), tuple(expressions.values()), modules="math", cse=True, docstring_limit=0),
        )

    def evaluate(self, inputs: NamedMap[float]) -> NamedMap[float]:
        """Align names and evaluate a single point."""
        values = self.inputs.combine(inputs, lambda symbol, value: float(value))
        if not all(isfinite(value) for value in values.values()):
            raise ValueError("Plan inputs must be finite")
        result = NamedMap(zip(self.outputs, map(float, self._calculate(*values.values())), strict=True))
        if not all(isfinite(value) for value in result.values()):
            raise ValueError("Plan outputs must be finite")
        return result

    def over(self, times: tuple[Time, ...], inputs: NamedMap[Series[Time]]) -> NamedMap[Series[Time]]:
        """Align names and axes once, then evaluate every row without solving."""
        columns = self.inputs.combine(inputs, lambda symbol, series: series)
        if any(column.times != times for column in columns.values()):
            raise ValueError("Plan inputs must have the same observation times")
        rows = tuple(self._calculate(*(column.values[i] for column in columns.values())) for i in range(len(times)))
        return NamedMap((name, Series(times, tuple(row[i] for row in rows))) for i, name in enumerate(self.outputs))


Model = TypeVar("Model", bound=Modello)


Value = TypeVar("Value", covariant=True)


InputModel = TypeVar("InputModel", bound=Modello, contravariant=True)


@dataclass(frozen=True)
class Known(Generic[Value]):
    """Bind observed data to a declared field of a model case."""

    field: SymbolicScalar
    value: Value


@runtime_checkable
class ModelData(Protocol[InputModel, Value]):
    """Typed observations that bind themselves to fields of a compatible model."""

    def knowns(self, model: InputModel) -> tuple[Known[Value], ...]:
        """Select model fields and validate application identities at the boundary."""
        ...


@runtime_checkable
class ModelSeries(ModelData[InputModel, Series[Time]], Protocol[InputModel, Time]):
    """Model observations carrying their explicit evaluation axis."""

    @property
    def times(self) -> tuple[Time, ...]: ...


@dataclass(frozen=True)
class ModelField:
    """A declared field, its resolved expression, and optionally its evaluated value."""

    path: FieldPath
    expression: SymbolicScalar
    role: Literal["input", "known", "derived"]
    value: float | None = None


def _get_field(model: Modello, path: FieldPath) -> Expr:
    value = _path_value(model, path)
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


def _copy_model(model: Model, values: dict[FieldPath, Expr], inputs: tuple[Expr, ...] = ()) -> Model:
    """Populate a copy without solving again or changing the source case.

    Like the named-member adapter, this is isolated access to core metadata.
    Model classes, component types, provider bindings, and symbol scopes survive.
    Cached properties must be recomputed for the copied field values.
    """

    def populate(source: Modello, prefix: FieldPath) -> Modello:
        result = copy(source)
        result._modello_bound_fields = {}
        for path, dummy in type(source)._modello_definition.fields.items():
            binding = source._modello_bound_fields[dummy]
            expression = _get_field(source, path)
            result._modello_bound_fields[dummy] = replace(
                binding,
                value=values[prefix + path],
                kind=ConstraintKind.INPUT if expression in inputs else binding.copied_kind(expression),
            )
        for name in tuple(vars(result)):
            if isinstance(getattr(type(source), name, None), cached_property):
                delattr(result, name)
        namespace = type(source)._modello_namespace
        for name in namespace.dummies:
            setattr(result, name, values[prefix + (name,)])
        for name, vector in namespace.vectors.items():
            setattr(result, name, Vector(*(values[prefix + (name, str(i))] for i in range(len(vector)))))
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
        _validate_vector_values(type(model), model)
        expressions = {path: _get_field(model, path) for path in type(model)._modello_definition.fields}
        self.source = _copy_model(model, expressions)
        self._paths = tuple(expressions)
        self._expressions = tuple(expressions.values())
        if any(not isinstance(expression, Expr) for expression in properties):
            raise TypeError("Additional properties must be model expressions")
        self._outputs = tuple(dict.fromkeys((*self._expressions, *cast(tuple[Expr, ...], properties))))
        inferred = tuple(dict.fromkeys(expr for expr in self._expressions if isinstance(expr, Symbol)))
        selected = inferred if inputs is None else inputs
        if len(set(selected)) != len(selected) or any(
            not isinstance(expr, Symbol) or expr not in inferred for expr in selected
        ):
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

    def _align(self, observations: tuple[Known[Value] | ModelData[Model, Value], ...]) -> NamedMap[Value]:
        supplied: dict[Expr, Value] = {}
        for observation in observations:
            knowns: tuple[Known[Value], ...]
            if isinstance(observation, Known):
                knowns = (observation,)
            elif isinstance(observation, ModelData):
                knowns = observation.knowns(self.source)
            else:
                raise TypeError("Supply Known values or typed ModelData observations")
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
            dummy = type(self.source)._modello_definition.fields[path]
            if any(
                getattr(value, f"is_{assumption}", None) not in (None, expected)
                for assumption, expected in dummy.assumptions0.items()
            ):
                raise ValueError(f"Evaluated field {'.'.join(path)} violates its model assumptions")
        return _copy_model(self.source, fields, self._inputs)

    def evaluate(self, *knowns: Known[float] | ModelData[Model, float]) -> Model:
        """Evaluate once and return a fresh numeric case with the same component types."""
        inputs = self._align(knowns)
        try:
            values = self._plan.evaluate(inputs)
        except ArithmeticError as error:
            raise ValueError("Values violate model assumptions or leave expressions undefined") from error
        return self._materialize(tuple(values.values()))

    @overload
    def over(self, data: ModelSeries[Model, Time], /) -> ModelPath[Model, Time]: ...

    @overload
    def over(
        self, times: tuple[Time, ...], *knowns: Known[Series[Time]] | ModelData[Model, Series[Time]]
    ) -> ModelPath[Model, Time]: ...

    def over(
        self,
        times: tuple[Time, ...] | ModelSeries[Model, Time],
        *knowns: Known[Series[Time]] | ModelData[Model, Series[Time]],
    ) -> ModelPath[Model, Time]:
        """Evaluate aligned columns once; materialize model cases only when requested."""
        if isinstance(times, ModelSeries):
            if knowns:
                raise TypeError("Supply an observation record or an axis with field bindings")
            inputs = self._align((times,))
            times = times.times
        else:
            inputs = self._align(knowns)
        for name, symbol in self._input_columns.items():
            _check_column(symbol, inputs[name].values)
        try:
            outputs = self._plan.over(times, inputs)
        except ArithmeticError as error:
            raise ValueError("Values violate model assumptions or leave expressions undefined") from error
        columns = dict(zip(self._outputs, outputs.values(), strict=True))
        for path, expression in zip(self._paths, self._expressions, strict=True):
            _check_column(type(self.source)._modello_definition.fields[path], columns[expression].values)
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


_Symbolic = TypeVar("_Symbolic", bound="SymbolicModel")


class SymbolicModel(Modello):
    """Reuse one equation definition through model-preserving numerical views."""

    def compile(
        self: _Symbolic,
        *,
        inputs: tuple[SymbolicScalar, ...] | None = None,
        properties: tuple[SymbolicScalar, ...] = (),
    ) -> ModelPlan[_Symbolic]:
        """Compile this case's remaining independent fields and retain its model type."""
        return ModelPlan(self, inputs=inputs, properties=properties)

    @classmethod
    def plan(cls: type[_Symbolic]) -> ModelPlan[_Symbolic]:
        """Cache an unbound view; bound or inverse cases use their own compile method."""
        return cast(ModelPlan[_Symbolic], _model_plan(cls))


@cache
def _model_plan(model: type[SymbolicModel]) -> ModelPlan[SymbolicModel]:
    return model(model.__name__).compile()
