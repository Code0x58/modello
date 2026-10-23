"""Nested model summaries and their editable observation records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from modello._paths import _path_label
from modello._problem import ConstraintKind

if TYPE_CHECKING:
    from modello._gui.presentation import Layout
    from modello.diagnostics import ModelProblem
    from modello.gui import _Snapshot


def _records(
    root: str,
    size: int,
    rows: Mapping[str, dict[str, Any]],
    observed: set[str],
    hidden: frozenset[str],
) -> list[dict[str, str | None]]:
    return [
        {
            name[len(f"{root}[{i}].") :]: field["input"] if name in observed else None
            for name, field in rows.items()
            if name.startswith(f"{root}[{i}].") and name not in hidden
        }
        for i in range(size)
    ]


def component_views(
    layout: Layout,
    context: _Snapshot,
    rows: Mapping[str, dict[str, Any]],
    *,
    input_axes: frozenset[str],
    dimensions: frozenset[str],
) -> list[dict[str, Any]]:
    """Retain typed member nesting and the origins of their observed cells."""
    observed = {_path_label(c.path) for c in context.problem.constraints if c.kind is ConstraintKind.INPUT}
    result = []
    for path in layout.components:
        if path in layout.hidden:
            continue
        info = layout.metadata[path]
        parent = layout.parents[path]
        row: dict[str, Any] = {
            "path": path,
            "parent": parent,
            "title": info.title or path.split(".")[-1],
            "description": info.description,
            "collapsed": path in layout.folded,
        }
        if path in layout.collections:
            size = int(layout.collections[path]._modello_collection_size)
            member = layout._member_styles.get((path, ""))
            row.update(
                collection=True,
                shape=[size],
                symbol_mathml=context.printer.vector_symbol("", info.symbol or path),
                resizable=path in layout.editable_collections and (path in input_axes or not input_axes),
                member_title=member.title if member is not None and member.title else "Model",
                dimensions=list(dimensions),
                observations=_records(path, size, rows, observed, layout.hidden),
            )
        elif parent in layout.collections and path.endswith("]"):
            position = int(path.rpartition("[")[2][:-1])
            row.update(
                index=position,
                symbol_mathml=context.printer.indexed_symbol(
                    "",
                    info.symbol or layout.metadata[parent].symbol or parent,
                    position,
                ),
            )
        result.append(row)
    return result


def update_collection_observations(
    components: list[dict[str, Any]],
    rows: Mapping[str, dict[str, Any]],
    problem: ModelProblem,
    hidden: frozenset[str],
) -> None:
    """Refresh editor text without turning inferred or aliased values into inputs."""
    observed = {_path_label(c.path) for c in problem.constraints if c.kind is ConstraintKind.INPUT}
    for component in components:
        if component.get("collection"):
            component["observations"] = _records(component["path"], component["shape"][0], rows, observed, hidden)
