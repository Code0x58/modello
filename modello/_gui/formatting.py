"""Readable numeric summaries with exact expressions retained separately."""

from __future__ import annotations

import math

from sympy import Expr, Rational


def display_number(value: Expr, *, percent: bool = False) -> tuple[str, bool]:
    """Format finite real scalars; mark rounding without changing the algebra."""
    if value.is_number is not True or value.is_real is not True or value.is_finite is not True:
        return "", False
    number = float(value.evalf(16))
    if not math.isfinite(number) or (number == 0 and value.is_zero is False):
        return str(value.evalf(4)) + ("%" if percent else ""), True
    if number and (abs(number) < 0.01 or abs(number) >= 1e12):
        text = format(number, ".3g")
    else:
        text = format(number, ",.2f").rstrip("0").rstrip(".")
    rounded = (value - Rational(text.replace(",", ""))).is_zero is not True
    return text + ("%" if percent else ""), rounded
