# mypy: warn-unused-ignores
"""Equity providers, quote identity, and configured acquisition parameters."""

from pathlib import Path

import pytest

from examples.compute.named_map import NamedMap
from examples.compute.providers import Provider, load_providers
from examples.portfolio.models import AAPL, Equity
from examples.portfolio.providers import CSVEquityProvider, DummyEquityProvider, EquityProvider
from examples.portfolio.sources import PriceHistory

from ._data import DATA


def test_dummy_provider_returns_typed_fixture_data():
    provider = DummyEquityProvider(symbol="AAPL")
    assert provider.instrument == AAPL
    assert provider.load().prices.values[0] == 100


def test_provider_symbols_are_independent_of_holding_names_and_shared():
    histories = load_providers(
        NamedMap({"growth": DummyEquityProvider(symbol="AAPL"), "income": DummyEquityProvider(symbol="AAPL")})
    )
    assert tuple(histories) == ("growth", "income")
    assert histories["growth"] is histories["income"]
    assert histories["growth"].equity == AAPL


def test_requests_include_all_parameters_not_only_the_instrument_id(tmp_path):
    original = DATA / "market.csv"
    alternate = tmp_path / "alternate.csv"
    alternate.write_text(original.read_text().replace(",100,1000", ",200,1000"))
    one = CSVEquityProvider(symbol="AAPL", path=original, namespace="synthetic")
    two = CSVEquityProvider(symbol="AAPL", path=alternate, namespace="synthetic")
    assert one.instrument == two.instrument
    histories = load_providers(NamedMap({"one": one, "two": two}))
    assert histories["one"].prices.values[0] == 100
    assert histories["two"].prices.values[0] == 200
    # A new load observes changed sources instead of a persistent cache.
    alternate.write_text(original.read_text())
    assert two.load().prices.values[0] == 100


def test_provider_symbols_and_quote_currencies_remain_distinct(tmp_path):
    original = DATA / "market.csv"
    with pytest.raises(ValueError, match="No market data for other:AAPL"):
        CSVEquityProvider(symbol="AAPL", path=original, namespace="other").load()
    with pytest.raises(ValueError, match="Market currency.*must be EUR"):
        DummyEquityProvider(symbol="AAPL", currency="EUR").load()
    path = tmp_path / "providers.csv"
    content = original.read_text()
    extra = content.replace("synthetic", "other").splitlines()[1:]
    path.write_text(content + "\n".join(extra) + "\n")
    histories = load_providers(
        NamedMap(
            {
                "one": CSVEquityProvider(symbol="AAPL", path=path, namespace="synthetic"),
                "two": CSVEquityProvider(symbol="AAPL", path=path, namespace="other"),
            }
        )
    )
    assert histories["one"].prices == histories["two"].prices
    assert histories["one"].equity != histories["two"].equity


def test_loading_validates_the_feed_schema_and_requested_symbol(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("member,price\naapl,100\n")
    with pytest.raises(ValueError, match="unique columns"):
        CSVEquityProvider(symbol="AAPL", path=bad, namespace="synthetic").load()
    bad.write_text((DATA / "market.csv").read_text().replace(",AAPL,", ",MSFT,"))
    with pytest.raises(ValueError, match="No market data for synthetic:AAPL"):
        CSVEquityProvider(symbol="AAPL", path=bad, namespace="synthetic").load()


def test_provider_construction_and_symbolic_use_do_not_fetch_data(tmp_path):
    from examples.portfolio.basket import basket_model
    from examples.portfolio.named import EquityMembers, portfolio_model

    provider = CSVEquityProvider(symbol="AAPL", path=tmp_path / "absent.csv", namespace="test")
    assert provider.instrument == Equity("AAPL", "test")
    Basket = basket_model(EquityMembers, NamedMap({"aapl": provider, "msft": provider}))
    portfolio = portfolio_model(Basket)("symbolic")
    assert portfolio.basket.positions["aapl"].provider is provider
    assert portfolio.nav.free_symbols
    with pytest.raises(FileNotFoundError):
        load_providers(portfolio.providers)


@pytest.mark.parametrize("field", ["symbol", "namespace", "currency"])
def test_provider_parameters_validate_identity_without_reading_a_file(field):
    values = {"symbol": "AAPL", "namespace": "test", "currency": "USD"}
    values[field] = ""
    with pytest.raises(ValueError, match="symbol, provider, and quote currency"):
        CSVEquityProvider(path=Path("unused.csv"), **values)


def _typing_contract() -> None:
    provider: EquityProvider = DummyEquityProvider(symbol="AAPL")
    generic: Provider[PriceHistory] = provider
    history: PriceHistory = generic.load()
    identity: Equity = provider.instrument
    assert history.equity == identity
    _wrong_provider: Provider[Equity] = provider  # type: ignore[assignment]
