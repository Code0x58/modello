# mypy: warn-unused-ignores

"""Named holdings and their shared valuation plans."""

from datetime import date

import pytest
from sympy import Rational

from examples.compute.named_map import NamedMap
from examples.portfolio.basket import EquityBasket, EquityHolding, EquityHoldings, equity
from examples.portfolio.explicit import MarketData, Portfolio
from examples.portfolio.models import Equity, EquityPosition, Quote, USDCash
from examples.portfolio.named import (
    Basket,
    EquityMembers,
    NamedMarketData,
    NamedPortfolio,
    NamedPortfolioPricer,
    ThreeEquities,
    ThreeEquityBasket,
    example_named_market_data,
    example_named_portfolio,
)
from examples.portfolio.providers import DummyEquityProvider, EquityProvider
from examples.portfolio.sources import PriceHistory
from modello import BoundInstanceDummy


def two_equities():
    return NamedPortfolio.define(Basket)(
        "two",
        basket=Basket(
            "basket",
            members=EquityMembers(
                "equities", aapl=EquityPosition("apple", quantity=10), msft=EquityPosition("microsoft", quantity=20)
            ),
        ),
        cash=USDCash("cash", amount=100),
    )


def test_named_calculation_reproduces_original_portfolio():
    data = MarketData.example()
    reference = Portfolio.example().pricer().over(data.named())
    actual = NamedPortfolioPricer(two_equities()).over(
        NamedMarketData(data.aapl.close.times, NamedMap({"msft": data.msft, "aapl": data.aapl}))
    )
    assert tuple(actual.positions) == ("aapl", "msft")
    assert actual.nav == reference.nav
    for name, expected_position in (("aapl", reference.positions["aapl"]), ("msft", reference.positions["msft"])):
        for field in ("value", "weight", "returns", "return_contribution"):
            observed = getattr(actual.positions[name], field)
            expected = getattr(expected_position, field)
            assert observed.times == expected.times
            assert observed.values == pytest.approx(expected.values, abs=1e-15)
    assert actual.cash_value == reference.cash_value
    result = actual.analytics(periods_per_year=252)
    expected = reference.analytics(periods_per_year=252)
    for attr in ("total_return", "mean_period_return", "annualized_volatility", "maximum_drawdown"):
        assert getattr(result.performance, attr) == pytest.approx(getattr(expected.performance, attr))
    cov = result.annualized_covariance
    assert cov["aapl"]["msft"] == pytest.approx(expected.annualized_covariance["aapl"]["msft"])
    risk = result.latest_weights_risk
    assert risk.volatility == pytest.approx(expected.latest_weights_risk.volatility)
    assert risk.contributions["aapl"] == pytest.approx(expected.latest_weights_risk.contributions["aapl"])


def test_new_member_flows_through_aggregates_and_all_projections():
    portfolio = example_named_portfolio()
    path = NamedPortfolioPricer(portfolio).over(example_named_market_data())
    assert path.nav.values == (3100, 3105, 3030, 3190, 3230)
    assert (
        tuple(portfolio.positions) == tuple(portfolio.values) == tuple(portfolio.weights) == ("aapl", "msft", "googl")
    )
    assert tuple(path.positions) == tuple(path.values) == tuple(path.weights) == ("aapl", "msft", "googl")
    assert path.values["googl"].values == (1000, 1005, 990, 1010, 1020)
    for index, nav in enumerate(path.nav.values):
        assert nav == sum(value.values[index] for value in path.values.values()) + path.cash_value.values[index]
        total_weight = sum(weight.values[index] for weight in path.weights.values()) + path.cash_weight.values[index]
        assert total_weight == pytest.approx(1)
    returns = path.nav.simple_returns()
    for index, actual in enumerate(returns.values):
        assert sum(position.return_contribution.values[index] for position in path.positions.values()) == pytest.approx(
            actual, abs=1e-15
        )


def test_map_reduction_keeps_inverse_valuation():
    source = ThreeEquities(
        "equities",
        aapl=EquityPosition("apple", instrument=Quote("quote", price=100)),
        msft=EquityPosition("microsoft", quantity=20, instrument=Quote("quote", price=50)),
        googl=EquityPosition("alphabet", quantity=1, instrument=Quote("quote", price=1000)),
    )
    portfolio = NamedPortfolio.define(ThreeEquityBasket)(
        "target", basket=ThreeEquityBasket("basket", members=source), cash=USDCash("cash", amount=100), nav=3100
    )
    assert portfolio.positions["aapl"].quantity == 10
    assert portfolio.weights == {"aapl": Rational(10, 31), "msft": Rational(10, 31), "googl": Rational(10, 31)}
    assert isinstance(source.aapl.quantity, BoundInstanceDummy)


def test_empty_equity_domain_is_a_cash_portfolio():
    class NoEquities(EquityHoldings):
        pass

    portfolio = NamedPortfolio.define(EquityBasket.define(NoEquities))("cash", cash=USDCash("cash", amount=100))
    dates = example_named_market_data().times
    path = NamedPortfolioPricer(portfolio).over(NamedMarketData(dates, NamedMap({})))
    assert portfolio.positions == portfolio.values == portfolio.weights == {}
    assert path.positions == path.values == path.weights == {}
    assert path.nav.values == (100,) * len(dates)
    assert path.cash_weight.values == (1,) * len(dates)
    result = path.analytics(periods_per_year=252)
    assert result.annualized_covariance == result.latest_weights_risk.contributions == {}
    assert (
        result.performance.total_return
        == result.performance.annualized_volatility
        == result.latest_weights_risk.volatility
        == 0
    )


@pytest.mark.parametrize("names", [("aapl", "msft"), ("aapl", "msft", "googl", "extra")])
def test_price_domain_must_match_the_compiled_model(names):
    data = example_named_market_data()
    prices = NamedMap((name, data.prices[name] if name in data.prices else data.prices["aapl"]) for name in names)
    with pytest.raises(ValueError, match="Member names differ"):
        NamedPortfolioPricer(example_named_portfolio()).over(NamedMarketData(data.times, prices))


def test_names_and_instrument_types_are_distinct_checks():
    data = example_named_market_data()
    prices = NamedMap({"aapl": data.prices["msft"], "msft": data.prices["aapl"], "googl": data.prices["googl"]})
    with pytest.raises(ValueError, match="Price instrument must be AAPL"):
        NamedPortfolioPricer(example_named_portfolio()).over(NamedMarketData(data.times, prices))


def test_every_price_vector_uses_the_declared_date_axis():
    data = example_named_market_data()
    dates = tuple(date(2026, 2, day) for day in range(2, 7))
    with pytest.raises(ValueError, match="match the observation dates"):
        NamedMarketData(dates, data.prices)
    for invalid in ((), (dates[0],), (dates[1], dates[0]), (dates[0], dates[0])):
        with pytest.raises(ValueError, match="at least two strictly increasing"):
            NamedMarketData(invalid, NamedMap({}))


def test_invalid_compilation_inputs():
    with pytest.raises(ValueError, match="position quantities"):
        NamedPortfolioPricer(NamedPortfolio.define(Basket)("missing"))
    portfolio = NamedPortfolio.define(Basket)(
        "known prices",
        basket=Basket(
            "basket",
            members=EquityMembers(
                "equities",
                aapl=EquityPosition("apple", quantity=10, instrument=Quote("quote", price=100)),
                msft=EquityPosition("microsoft", quantity=20, instrument=Quote("quote", price=50)),
            ),
        ),
        cash=USDCash("cash", amount=100),
    )
    with pytest.raises(ValueError, match="independent symbolic input"):
        NamedPortfolioPricer(portfolio)


def test_currency_needs_an_explicit_conversion_model():
    class EuroMembers(EquityHoldings):
        euro = equity(DummyEquityProvider(symbol="EXAMPLE", currency="EUR"))

    EuroBasket = EquityBasket.define(EuroMembers)
    portfolio = NamedPortfolio.define(EuroBasket)(
        "mixed currencies",
        basket=EuroBasket("basket", members=EuroMembers("members", euro=EquityPosition("euro", quantity=1))),
        cash=USDCash("cash", amount=1),
    )
    with pytest.raises(ValueError, match="requires USD"):
        NamedPortfolioPricer(portfolio)


def _typing_contract() -> None:
    portfolio = NamedPortfolio.define(Basket)("portfolio")
    apple: EquityPosition = portfolio.basket.members.aapl
    assert apple.instrument.price.free_symbols
    identity: Equity = portfolio.basket.instruments["aapl"]
    assert identity.symbol == "AAPL"
    holding: EquityHolding = portfolio.positions["aapl"]
    provider: EquityProvider = holding.provider
    history: PriceHistory = provider.load()
    assert history.equity == identity
    assert isinstance(provider, DummyEquityProvider)
    instrument: Equity = holding.instrument
    assert instrument.symbol == "AAPL" and holding.weight.free_symbols
    portfolio.basket.members.aapl = USDCash("cash")  # type: ignore[assignment]
    _wrong_result: Equity = provider.load()  # type: ignore[assignment]
    declared: EquityPosition = equity(provider)
    assert declared
    _wrong_holding: USDCash = equity(provider)  # type: ignore[assignment]
    equity(equity)  # type: ignore[arg-type]
