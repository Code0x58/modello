"""Aggregate coordinate views without changing their observations or algebra."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Mapping

from sympy import Expr, Symbol

from modello.vectors import Vector

if TYPE_CHECKING:
    from modello._gui.presentation import Layout
    from modello.gui import _Snapshot


def vector_view(
    path: str,
    layout: Layout,
    context: _Snapshot,
    values: Mapping[Symbol, Expr],
    rows: Mapping[str, dict[str, Any]],
    raw: bool,
) -> dict[str, Any]:
    """Summarize a vector while retaining each entry's role and edit history."""
    vector = layout.vectors[path]
    info = layout.metadata[path]
    coordinates = [f"{path}[{i}]" for i in range(len(vector))]
    visible = [name for name in coordinates if name not in layout.hidden]
    elements = Vector(
        *(
            symbol
            if coordinates[i] in layout.hidden
            else context.definitions.get(symbol, symbol)
            if raw
            else values[symbol]
            for i, symbol in enumerate(vector)
        )
    )
    roles = Counter(rows[name]["kind"] for name in visible)
    kind = next(iter(roles)) if len(roles) == 1 else "mixed" if roles else "derived"
    role_labels = {"input": "observed", "free": "free", "derived": "calculated", "fixed": "recorded"}
    provenance = " · ".join([f"{len(vector)} entries", *(f"{count} {role_labels[k]}" for k, count in roles.items())])
    equation = layout.vector_equations.get(path)
    related = [{"kind": "definition", **equation}] if equation is not None else []
    for name in visible:
        for constraint in rows[name]["constraints"]:
            if constraint not in related:
                related.append(constraint)
    current_input = _input_text(coordinates, rows, "input") if len(visible) == len(vector) else ""
    initial_input = _input_text(coordinates, rows, "initial_input") if len(visible) == len(vector) else ""
    input_state = (
        "hidden"
        if len(visible) != len(vector)
        else "initial"
        if current_input == initial_input and initial_input
        else "unspecified"
        if current_input == initial_input
        else "dropped"
        if not current_input
        else "added"
        if not initial_input
        else "overridden"
    )
    scaled = elements * 100 if info.number_format == "percent" else elements
    aliases = [name for name, other in layout.vectors.items() if other == vector and name not in layout.hidden]
    observation_paths = list(dict.fromkeys([*aliases, *(alias for name in visible for alias in rows[name]["aliases"])]))
    return {
        "path": path,
        "parent": path.rpartition(".")[0],
        "name": path.rpartition(".")[2],
        "depth": path.count("."),
        "title": info.title or path.rpartition(".")[2].replace("_", " "),
        "description": info.description,
        "info": info.info,
        "symbol": info.symbol or path,
        "shape": [len(vector)],
        "entries": visible,
        "kind": kind,
        "provenance": provenance,
        "unit": info.unit,
        "number_format": info.number_format,
        "value": context.problem.format(elements),
        "value_state": "numeric" if all(entry.is_number for entry in elements) else "symbolic",
        "display_value": "",
        "approximate": False,
        "value_mathml": context.printer.values(scaled),
        "exact_mathml": context.printer.values(scaled),
        "symbol_mathml": context.printer.vector_symbol(path, info.symbol or path),
        "input": current_input,
        "initial_input": initial_input,
        "input_state": input_state,
        "editable": path in layout.editable_vectors,
        "assumptions": [f"{len(vector)} scalar entries"],
        "aliases": aliases,
        "observation_paths": observation_paths,
        "constraints": related,
        "depends_on": [
            {"path": name, "title": rows[name]["title"]} for name in visible if rows[name]["kind"] == "free"
        ],
    }


def _input_text(coordinates: list[str], rows: Mapping[str, dict[str, Any]], key: str) -> str:
    if not any(rows[name][key] for name in coordinates):
        return ""
    return "[" + ", ".join(rows[name][key] or "?" for name in coordinates) + "]"
