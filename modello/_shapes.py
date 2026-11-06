"""Resolve symbolic finite axes before compiling scalar coordinates."""

from __future__ import annotations

from collections.abc import Mapping

from sympy import Basic, Eq, Integer, Symbol, check_assumptions, solve, sympify

from modello._collections import _ModelVectorSpec
from modello._evaluation import _substitute
from modello._operators import _scalar_expression
from modello._reconstruction import _OPERATORS
from modello.errors import ModelInputError
from modello.vectors import Vector, _vector_input


def _dimensions(
    vectors: Mapping[str, Vector | _ModelVectorSpec],
    fields: Mapping[str, Symbol],
    definitions: Mapping[str, Basic],
    values: Mapping[str, object],
    component_values: Mapping[Symbol, Basic] | None = None,
) -> dict[Symbol, Integer]:
    """Use supplied lengths and scalar observations as simultaneous shape equations."""
    axes = {name: sympify(vector.shape[0]) for name, vector in vectors.items()}
    symbols = set().union(*(axis.free_symbols for axis in axes.values()))
    if not symbols:
        return {}
    if symbols - set(fields.values()) or any(symbol.is_integer is not True for symbol in symbols):
        raise ModelInputError("Symbolic vector sizes must use declared integer model fields")
    equations = []
    for name, axis in axes.items():
        if name in values and values[name] is not None:
            if isinstance(vectors[name], _ModelVectorSpec) and isinstance(values[name], Mapping):
                continue
            equations.append(Eq(axis, len(_vector_input(values[name], None, name))))
    for name, symbol in fields.items():
        if symbol in symbols:
            value = values.get(name, definitions.get(name, symbol))
            if value != symbol:
                equations.append(Eq(symbol, sympify(value)))
    observations = {
        name: _scalar_expression(value, f"{name} must be a scalar expression")
        for name, value in values.items()
        if name in fields and fields[name] not in symbols and value is not None
    }
    known = {**(component_values or {}), **{fields[name]: value for name, value in observations.items()}}
    for name, value in observations.items():
        # A total or another observed output can determine the window length.
        # Native operations and state queries need the prepared model's resolver.
        residual = _substitute(definitions.get(name, fields[name]) - value, known)
        if residual.free_symbols and residual.free_symbols <= symbols and not residual.has(Vector, *_OPERATORS):
            equations.append(Eq(residual, 0))
    candidates = solve(equations, sorted(symbols, key=str), dict=True) if equations else []
    complete = [
        candidate
        for candidate in candidates
        if all(isinstance(candidate.get(symbol), Integer) for symbol in symbols)
        and all(axis.subs(candidate).is_nonnegative is True for axis in axes.values())
        and all(check_assumptions(candidate[symbol], against=symbol) is not False for symbol in symbols)
    ]
    if len(complete) != 1:
        names = ", ".join(name for name, field in fields.items() if field in symbols)
        if equations and not candidates:
            raise ModelInputError(f"Vector lengths and size observations conflict for {names}")
        raise ModelInputError(f"Supply compatible vector observations or integer sizes to determine {names}")
    return complete[0]
