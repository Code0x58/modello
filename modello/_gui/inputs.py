"""A bounded exact-arithmetic language for untrusted browser input."""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TypeAlias

from sympy import E, Expr, I, Rational, pi, sqrt

from modello._values import ScalarValue
from modello.errors import ModelInputError
from modello.vectors import VectorValue

ModelVectorInput: TypeAlias = Sequence[Mapping[str, ScalarValue | None] | None]
ExplorerValue: TypeAlias = ScalarValue | VectorValue | ModelVectorInput

_MAX_SCALAR_LENGTH = 160


def parse_model_vector(text: str, size: int | None) -> tuple[dict[str, Expr | None], ...]:
    """Read model records using the existing bounded scalar language per cell."""
    return tuple(
        {name: None if value is None else parse_scalar(value) for name, value in record.items()}
        for record in model_vector_entries(text, size)
    )


def model_vector_entries(text: str, size: int | None) -> tuple[dict[str, str | None], ...]:
    """Read bounded observation records while retaining their arithmetic text."""
    if len(text) > max(16_000, (size or 0) * 2048):
        raise ModelInputError("The model vector observation is too large")
    try:
        records = json.loads(text)
    except (ValueError, TypeError) as error:
        raise ModelInputError("Supply an ordered list of model observations") from error
    if not isinstance(records, list) or (size is not None and len(records) != size):
        raise ModelInputError("Model observations must agree with the vector length")
    result = []
    for record in records:
        if record is None:
            record = {}
        if not isinstance(record, dict) or len(record) > 256:
            raise ModelInputError("Each model observation must contain named scalar fields")
        values = {}
        for name, value in record.items():
            if not isinstance(name, str) or len(name) > 256:
                raise ModelInputError("Model observations need valid field names")
            if value is not None and not isinstance(value, str):
                raise ModelInputError("Enter model field values as text")
            values[name] = None if value is None or not value.strip() else value.strip()
        result.append(values)
    return tuple(result)


def parse_scalar(text: str) -> Expr:
    """Read a deliberately small arithmetic language, never eval/sympify text."""
    if not text.strip() or len(text) > _MAX_SCALAR_LENGTH:
        raise ModelInputError("Enter a number or a short exact arithmetic expression.")
    percent = text.rstrip().endswith("%")
    text = text.strip()[:-1].strip() if percent else text.strip()
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise ModelInputError("Use numbers, +, -, *, /, **, sqrt, pi, E, or I.") from error
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ModelInputError("The arithmetic expression is too long.")

    def read(node: ast.AST) -> Expr:
        result = read_node(node)
        if result.is_Rational and max(int(result.p).bit_length(), int(result.q).bit_length()) > 4096:
            raise ModelInputError("The arithmetic result is too large.")
        return result

    def read_node(node: ast.AST) -> Expr:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            token = ast.get_source_segment(text.strip(), node) or ""
            if len(token) > 32 or not math.isfinite(node.value) or abs(node.value) > 10**100:
                raise ModelInputError("Numeric literals must be finite and at most 1e100.")
            exponent = re.search(r"[eE]([+-]?[0-9_]+)$", token)
            if exponent and abs(int(exponent[1].replace("_", ""))) > 1000:
                raise ModelInputError("Decimal exponents must be between -1000 and 1000.")
            return Rational(token.replace("_", ""))
        if isinstance(node, ast.Name) and node.id in ("pi", "E", "I"):
            return {"pi": pi, "E": E, "I": I}[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = read(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = read(node.left), read(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and right.is_Rational and abs(right) <= 100:
                return left**right
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and not node.keywords
        ):
            return sqrt(read(node.args[0]))
        raise ModelInputError("Use numbers, +, -, *, /, ** (power at most 100), sqrt, pi, E, or I.")

    try:
        result = read(tree.body)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
        raise ModelInputError(str(error)) from error
    if result.is_finite is not True:
        raise ModelInputError("Inputs must be finite scalar values.")
    return result / 100 if percent else result


def parse_vector(text: str, size: int | None) -> tuple[Expr | None, ...]:
    """Parse exact scalar entries; '?' explicitly leaves a coordinate unobserved."""
    values: list[Expr | None] = []
    for i, entry in enumerate(vector_entries(text, size)):
        try:
            values.append(None if entry is None else parse_scalar(entry))
        except ModelInputError as error:
            raise ModelInputError(f"Entry {i}: {error}") from error
    return tuple(values)


def vector_entries(text: str, size: int | None) -> tuple[str | None, ...]:
    """Keep accepted arithmetic text when a whole-vector edit is flattened."""
    text = text.strip()
    limit = max(16_000, (size or 0) * (_MAX_SCALAR_LENGTH + 2) + 2)
    if not text.startswith("[") or not text.endswith("]") or len(text) > limit:
        raise ModelInputError("Enter a vector such as [1, ?, 3/2]; ? leaves an entry unobserved.")
    body = text[1:-1].strip()
    entries = body.split(",") if body else []
    if size is not None and len(entries) != size:
        raise ModelInputError(f"Enter exactly {size} vector entries; received {len(entries)}.")
    return tuple(None if entry.strip() == "?" else entry.strip() for entry in entries)


def binding_texts(
    bindings: Mapping[str, str | None],
    sizes: Mapping[str, int],
    *,
    collections: Mapping[str, int] | None = None,
    preserve: frozenset[str] = frozenset(),
) -> dict[str, str | None]:
    """Normalize paths while preserving the caller's bounded scalar syntax."""
    result: dict[str, str | None] = {}
    for path, text in bindings.items():
        if path in preserve:
            result[path] = None if text is None or not text.strip() else text.strip()
            continue
        if path in sizes:
            entries = (None,) * sizes[path] if text is None or not text.strip() else vector_entries(text, sizes[path])
            result.update((f"{path}[{i}]", entry) for i, entry in enumerate(entries))
        elif collections is not None and path in collections and text:
            for i, record in enumerate(model_vector_entries(text, collections[path])):
                result.update((f"{path}[{i}].{name}", value) for name, value in record.items())
        else:
            result[path] = None if text is None or not text.strip() else text.strip()
    return result


def request_limit(paths: Iterable[str]) -> int:
    """Budget a bounded scalar per editable path, including JSON escaping.

    A supplementary Unicode character can take twelve bytes as a JSON escape.
    The minimum also leaves room for a small request's envelope and diagnostics.
    """
    return max(16_384, 64 + sum(len(json.dumps(path)) + 12 * _MAX_SCALAR_LENGTH + 16 for path in paths))
