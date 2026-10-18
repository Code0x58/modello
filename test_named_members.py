# mypy: warn-unused-ignores
"""Named-domain semantics, independent of the financial example."""

import pytest

from examples.named_members import NamedMap, component_members
from examples.named_portfolio import EquityMembers, portfolio_model
from examples.portfolio import ApplePosition, EquityPosition, MicrosoftPosition, USDCash
from modello import Modello, component


def test_membership_is_owned_and_read_only():
    source = {"left": 1, "right": 2}
    members = NamedMap(source)
    source["left"] = 99
    source["new"] = 3
    assert members == {"left": 1, "right": 2}
    assert dict(members.items()) == {"left": 1, "right": 2}
    with pytest.raises(TypeError):
        members["left"] = 10
    with pytest.raises(KeyError):
        members["missing"]


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


def test_empty_domain_has_an_empty_projection_and_zero_sum():
    empty = NamedMap({})
    assert len(empty) == 0
    assert empty.map(str) == {}
    assert empty.combine({}, lambda a, b: (a, b)) == {}
    assert sum(empty.values()) == 0


@pytest.mark.parametrize("name", ["", 1, None])
def test_member_names_are_nonempty_strings(name):
    with pytest.raises(ValueError, match="nonempty strings"):
        NamedMap([(name, 1)])


def test_duplicate_names_are_not_silently_overwritten():
    with pytest.raises(ValueError, match="Duplicate member name 'same'"):
        NamedMap([("same", 1), ("same", 2)])


def test_component_view_retains_concrete_instances():
    source = EquityMembers(
        "holdings", aapl=ApplePosition("apple", quantity=10), msft=MicrosoftPosition("microsoft", quantity=20)
    )
    view = component_members(source, EquityPosition)
    assert tuple(view) == ("aapl", "msft")
    assert view["aapl"] is source.aapl
    assert isinstance(view["aapl"], ApplePosition)
    assert view.map(lambda position: position.quantity) == {"aapl": 10, "msft": 20}


def test_component_view_checks_its_common_value_type():
    class Mixed(Modello):
        apple = component(ApplePosition)
        cash = component(USDCash)

    with pytest.raises(TypeError, match="cash must be a EquityPosition component"):
        component_members(Mixed("mixed"), EquityPosition)
    with pytest.raises(TypeError, match="only model components"):
        component_members(ApplePosition("apple"), Modello)


def _typing_contract(position: EquityPosition) -> None:
    """Keep projections typed and prevent accidentally introducing Any."""
    members = NamedMap({"position": position})
    typed_position: EquityPosition = members["position"]
    tickers: NamedMap[str] = members.map(lambda member: member.instrument.ticker)
    numbers: NamedMap[int] = tickers.map(len)
    labels: NamedMap[str] = tickers.combine(numbers, lambda text, number: text * number)
    assert typed_position is position and labels["position"]
    _wrong_member: EquityPosition = labels["position"]  # type: ignore[assignment]
    _wrong_projection: NamedMap[EquityPosition] = tickers  # type: ignore[assignment]
    members["position"] = position  # type: ignore[index]
    portfolio = portfolio_model(EquityMembers)("portfolio")
    apple: ApplePosition = portfolio.equities.aapl
    assert apple.instrument.ticker == "AAPL"
    portfolio.equities.aapl = USDCash("cash")  # type: ignore[assignment]
