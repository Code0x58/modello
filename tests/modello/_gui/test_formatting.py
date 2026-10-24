"""Numeric display retains exact values and marks rounded summaries."""

from sympy import Rational, Symbol, sqrt


def test_numeric_formatting_retains_exact_values_and_marks_rounding():
    from modello._gui.formatting import display_number

    assert display_number(Rational(6500)) == ("6,500", False)
    assert display_number(Rational(13, 2), percent=True) == ("6.5%", False)
    assert display_number(sqrt(793) / 2, percent=True) == ("14.08%", True)
    assert display_number(Rational(1, 10000)) == ("0.0001", False)
    assert display_number(Symbol("x")) == ("", False)
    assert display_number(Rational(10) ** 1000)[1]
    tiny, rounded = display_number(Rational(10) ** -1000)
    assert tiny != "0" and "e-1000" in tiny and rounded
    assert display_number(Rational(0)) == ("0", False)
