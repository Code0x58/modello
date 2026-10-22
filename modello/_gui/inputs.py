"""A bounded exact-arithmetic language for untrusted browser input."""

from __future__ import annotations

import ast
import math
import re

from sympy import E, Expr, I, Rational, pi, sqrt

from modello.errors import ModelInputError


def parse_scalar(text: str) -> Expr:
    """Read a deliberately small arithmetic language, never eval/sympify text."""
    if not text.strip() or len(text) > 160:
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
