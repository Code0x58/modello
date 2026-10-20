"""Named-domain ownership, projection, and exact name alignment."""

import pytest

from examples.compute.named_map import NamedMap


def test_membership_is_owned_and_read_only():
    source = {"left": 1, "right": 2}
    members = NamedMap(source)
    source["left"] = 99
    source["new"] = 3
    assert members == {"left": 1, "right": 2}
    with pytest.raises(TypeError):
        members["left"] = 10


def test_projection_keeps_identity_and_declaration_order():
    members = NamedMap({"second": 2, "first": 1})
    projected = members.map(lambda value: str(value * 10))
    assert tuple(projected) == ("second", "first")
    assert projected == {"second": "20", "first": "10"}


def test_combinations_align_by_name_not_position():
    left = NamedMap({"a": 10, "b": 20})
    right = NamedMap({"b": 2, "a": 1})
    combined = left.combine(right, lambda first, second: first - second)
    assert tuple(combined.items()) == (("a", 9), ("b", 18))
    assert combined == {"b": 18, "a": 9}


@pytest.mark.parametrize("other", [{"a": 1}, {"a": 1, "b": 2, "c": 3}, {"a": 1, "c": 3}])
def test_combinations_require_the_same_domain(other):
    with pytest.raises(ValueError, match="Member names differ"):
        NamedMap({"a": 1, "b": 2}).combine(other, lambda a, b: a + b)


@pytest.mark.parametrize("name", ["", 1, None])
def test_member_names_are_nonempty_strings(name):
    with pytest.raises(ValueError, match="nonempty strings"):
        NamedMap([(name, 1)])


def test_duplicate_names_are_not_silently_overwritten():
    with pytest.raises(ValueError, match="Duplicate member name 'same'"):
        NamedMap([("same", 1), ("same", 2)])
