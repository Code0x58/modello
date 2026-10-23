"""Browser scalar input uses bounded exact arithmetic."""

import pytest
from sympy import Rational, pi, sqrt

from modello._gui.inputs import parse_scalar, parse_vector
from modello.errors import ModelInputError


@pytest.mark.parametrize(
    "text, value",
    [
        ("1.25", Rational(5, 4)),
        ("70%", Rational(7, 10)),
        (" 6 % ", Rational(3, 50)),
        ("1/2%", Rational(1, 200)),
        ("3/2", Rational(3, 2)),
        ("sqrt(2)", sqrt(2)),
        ("pi", pi),
        ("2**3 - 1", 7),
        ("-2 + +3", 1),
    ],
)
def test_exact_arithmetic(text, value):
    assert parse_scalar(text) == value


@pytest.mark.parametrize(
    "text",
    [
        "%",
        "10%%",
        "1%2",
        "True",
        "1/0",
        "float('inf')",
        "1e9999",
        "1e-999999999",
        "2**101",
        "(2**100)**100",
        "[1,2]",
        "x",
        "x.width",
        "__import__('os').getcwd()",
        "sqrt(x=2)",
        "sqrt(1, 2)",
        "1 +",
        "",
        "2" * 161,
    ],
)
def test_parser_rejects_code_and_unbounded_arithmetic(text):
    with pytest.raises(ModelInputError):
        parse_scalar(text)


@pytest.mark.parametrize(
    "raw",
    [
        "[1,2]",
        "[1,2,3,4]",
        "[1,,3]",
        "[x,2,3]",
        "[1,True,3]",
        "[1, __import__('os'), 3]",
        "1,2,3",
        "[1, (2**100)**100, 3]",
    ],
)
def test_vector_parser_reuses_scalar_limits_and_rejects_code(raw):
    with pytest.raises(ModelInputError):
        parse_vector(raw, 3)


def test_vector_parser_supports_exact_percentages_and_explicit_missing_entries():
    assert parse_vector("[10%, ?, 3/2]", 3) == (Rational(1, 10), None, Rational(3, 2))
    assert parse_vector("[]", 0) == ()


def test_vector_text_budget_scales_with_its_trusted_shape():
    literal = "12345678901234567890123456789012"
    text = "[" + ", ".join([literal] * 600) + "]"
    assert len(text) > 16_000
    values = parse_vector(text, 600)
    assert len(values) == 600
    assert values[0] == values[-1] == int(literal)
