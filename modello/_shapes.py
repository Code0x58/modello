"""Resolve symbolic finite axes before compiling scalar coordinates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sympy import Basic, Eq, Integer, Symbol, Tuple, solve, sympify

from modello.errors import ModelInputError
from modello.vectors import Vector


def _dimensions(
    vectors: Mapping[str, Vector],
    fields: Mapping[str, Symbol],
    definitions: Mapping[str, Basic],
    values: Mapping[str, object],
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
            value = values[name]
            if not isinstance(value, (Vector, Sequence, Tuple)) or isinstance(value, (str, bytes)):
                raise ModelInputError(f"{name} must be a vector or a sequence of scalar observations")
            equations.append(Eq(axis, len(value)))
    for name, symbol in fields.items():
        if symbol in symbols:
            value = values.get(name, definitions.get(name, symbol))
            if value != symbol:
                equations.append(Eq(symbol, sympify(value)))
    candidates = solve(equations, sorted(symbols, key=str), dict=True) if equations else []
    complete = [
        candidate
        for candidate in candidates
        if all(isinstance(candidate.get(symbol), Integer) for symbol in symbols)
        and all(axis.subs(candidate).is_nonnegative is True for axis in axes.values())
        and all(
            getattr(candidate[symbol], f"is_{key}", None) in (None, expected)
            for symbol in symbols
            for key, expected in symbol.assumptions0.items()
        )
    ]
    if len(complete) != 1:
        names = ", ".join(name for name, field in fields.items() if field in symbols)
        if equations and not candidates:
            raise ModelInputError(f"Vector lengths and size observations conflict for {names}")
        raise ModelInputError(f"Supply compatible vector observations or integer sizes to determine {names}")
    return complete[0]
