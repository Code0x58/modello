"""Native and Python 3.10 string enums keep the same boundary behaviour."""

import json

import pytest

from modello._operators import _SignPredicate
from modello._problem import ConstraintKind


@pytest.mark.parametrize("member", (ConstraintKind.INPUT, _SignPredicate.POSITIVE))
def test_string_enums_interoperate_with_strings(member):
    text = member.value
    assert isinstance(member, str)
    assert member == text and {member: 1}[text] == 1
    assert str(member) == text
    assert f"{member:>12}" == f"{text:>12}"
    assert json.dumps(member) == json.dumps(text)
