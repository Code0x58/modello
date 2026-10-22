"""Shared field traversal and readable paths, including vector coordinates."""

from __future__ import annotations

from typing import Any


def _path_label(path: tuple[str, ...]) -> str:
    """Keep component dots and zero-based vector subscripts unambiguous."""
    result = ""
    for part in path:
        result += f"[{part}]" if part.isdecimal() else ("." if result else "") + part
    return result


def _path_value(value: Any, path: tuple[str, ...]) -> Any:
    """Traverse the same schema on a reference, a populated model, or a vector."""
    for part in path:
        value = value[int(part)] if part.isdecimal() else getattr(value, part)
    return value


def _within_path(path: str, parent: str) -> bool:
    return not parent or path == parent or path.startswith(parent + ".") or path.startswith(parent + "[")
