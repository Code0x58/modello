"""Native assumption keys retain their scalar-domain meaning."""

import pytest
from sympy import Q, S, Symbol

from modello._operators import _SignPredicate, _value_condition


@pytest.mark.parametrize(
    "predicate,truth_at_negative_zero_positive",
    (
        (_SignPredicate.POSITIVE, (False, False, True)),
        (_SignPredicate.NONNEGATIVE, (False, True, True)),
        (_SignPredicate.NEGATIVE, (True, False, False)),
        (_SignPredicate.NONPOSITIVE, (True, True, False)),
        (_SignPredicate.ZERO, (False, True, False)),
        (_SignPredicate.NONZERO, (True, False, True)),
    ),
)
def test_real_sign_assumptions_preserve_relational_guards(predicate, truth_at_negative_zero_positive):
    value = Symbol("value", real=True)
    # These keys come from SymPy's string-keyed assumption mappings.
    positive = _value_condition({predicate.value: True}, value)
    negative = _value_condition({predicate.value: False}, value)
    for point, truth in zip((-1, 0, 1), truth_at_negative_zero_positive):
        assert positive.subs(value, point) is (S.true if truth else S.false)
        assert negative.subs(value, point) is (S.false if truth else S.true)


def test_other_assumptions_retain_the_native_sympy_predicate():
    value = Symbol("value")
    assert _value_condition({"finite": True}, value) == Q.finite(value)
    assert _value_condition({"finite": False}, value) == ~Q.finite(value)


def test_sign_assumptions_do_not_order_an_unestablished_real_value():
    value = Symbol("value", complex=True)
    assert _value_condition({"positive": True}, value) == Q.positive(value)
