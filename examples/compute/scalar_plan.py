"""Compile named scalar expressions for point or aligned-series execution."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import cast

from sympy import Expr, Symbol, lambdify, sympify

from examples.compute.named_map import NamedMap
from examples.compute.series import Series, Time


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
