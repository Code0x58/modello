# mypy: warn-unused-ignores

"""Reusable valuation, exact domains, and typed portfolio views."""

from datetime import date, timedelta

import pytest

from examples.compute.named_map import NamedMap
from examples.portfolio.analytics import NamedPortfolioPath
from examples.portfolio.basket import EquityBasket, EquityHolding, EquityHoldings, equity
from examples.portfolio.explicit import Basket, EquityMembers, MarketData, Portfolio
from examples.portfolio.models import Equity, EquityPosition, Quote, USDCash
from examples.portfolio.named import ThreeEquityPortfolio
from examples.portfolio.providers import DummyEquityProvider, EquityProvider
from examples.portfolio.sources import PriceHistory, timestamp
from examples.portfolio.valuation import MarketSnapshot, NamedMarketData, NamedPortfolio


def test_identified_quotes_evaluate_to_the_original_typed_portfolio():
    book = Portfolio.example()
    pricer = book.pricer()
    at = timestamp("2026-01-05T16:01:00Z")
    snapshot = MarketSnapshot.sample(book.load_market(), at=at, max_price_age=timedelta(days=4))
    assert snapshot.at == at and snapshot.prices["aapl"].equity == book.instruments["aapl"]
    value = pricer.evaluate(snapshot)
    assert type(value) is Portfolio
    assert type(value.basket) is Basket and type(value.basket.members) is EquityMembers
    assert float(value.nav) == 2100
    assert float(value.basket.members.aapl.market_value) == 1000
    assert float(value.basket.positions["aapl"].weight) == pytest.approx(10 / 21)
    assert float(value.basket.holding(value.basket.members.aapl).weight) == pytest.approx(10 / 21)
    with pytest.raises(ValueError, match="this basket"):
        value.basket.holding(book.basket.members.aapl)
    assert value.providers == book.providers
    assert book.basket.members.aapl.instrument.price.free_symbols
    book.basket.members.aapl.instrument.price = 999
    assert float(pricer.evaluate(snapshot).nav) == 2100
    with pytest.raises(ValueError, match="Member names differ"):
        pricer.evaluate(MarketSnapshot(at, NamedMap({})))
    with pytest.raises(ValueError, match="instrument must"):
        pricer.evaluate(
            MarketSnapshot(at, NamedMap({"aapl": snapshot.prices["msft"], "msft": snapshot.prices["aapl"]}))
        )


def test_vector_evaluation_keeps_the_model_schema_for_points_and_columns():
    book = ThreeEquityPortfolio.example()
    path = book.pricer().over(ThreeEquityPortfolio.example_data())
    assert path.column(book.nav).values == (3100, 3105, 3030, 3190, 3230)
    point = path.at(4)
    assert type(point) is ThreeEquityPortfolio
    assert float(point.basket.members.googl.market_value) == 1020
    assert float(point.basket.positions["googl"].weight) == pytest.approx(1020 / 3230)


def test_member_names_share_no_namespace_with_portfolio_totals():
    class Members(EquityHoldings):
        nav = equity.component(DummyEquityProvider(symbol="AAPL"))
        cash = equity.component(DummyEquityProvider(symbol="MSFT"))

    basket = EquityBasket.define(Members)
    portfolio = NamedPortfolio.define(basket)(
        "names",
        basket=basket(
            "basket",
            members=Members("members", nav=equity("apple", quantity=10), cash=equity("microsoft", quantity=20)),
        ),
        cash=USDCash("cash", amount=100),
    )
    value = portfolio.pricer().evaluate(
        MarketSnapshot.sample(
            portfolio.load_market(), at=timestamp("2026-01-05T16:01:00Z"), max_price_age=timedelta(days=4)
        )
    )
    assert float(value.nav) == 2100
    assert float(value.basket.members.nav.market_value) == float(value.basket.members.cash.market_value) == 1000


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


def test_runtime_composition_reuses_definitions_with_independent_case_symbols():
    basket = EquityBasket.define(EquityMembers)
    portfolio = NamedPortfolio.define(basket)
    assert basket is EquityBasket.define(EquityMembers)
    assert portfolio is NamedPortfolio.define(basket)
    assert portfolio.plan() is NamedPortfolio.define(basket).plan()
    first, second = portfolio("same"), portfolio("same")
    assert first.basket.members.aapl.quantity != second.basket.members.aapl.quantity
    assert first.basket.members.aapl.instrument.price != second.basket.members.aapl.instrument.price

    class Empty(EquityHoldings):
        pass

    assert EquityBasket.define(Empty) is not basket


def test_named_calculation_reproduces_original_portfolio():
    data = MarketData.example()
    reference = NamedPortfolioPath.from_model(Portfolio.example().pricer().over(data.named()))
    actual = NamedPortfolioPath.from_model(
        two_equities()
        .pricer()
        .over(NamedMarketData(data.aapl.close.times, NamedMap({"msft": data.msft, "aapl": data.aapl})))
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


def test_empty_equity_domain_is_a_cash_portfolio():
    class NoEquities(EquityHoldings):
        pass

    portfolio = NamedPortfolio.define(EquityBasket.define(NoEquities))("cash", cash=USDCash("cash", amount=100))
    dates = ThreeEquityPortfolio.example_data().times
    path = NamedPortfolioPath.from_model(portfolio.pricer().over(NamedMarketData(dates, NamedMap({}))))
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
    data = ThreeEquityPortfolio.example_data()
    prices = NamedMap((name, data.prices[name] if name in data.prices else data.prices["aapl"]) for name in names)
    with pytest.raises(ValueError, match="Member names differ"):
        NamedPortfolioPath.from_model(ThreeEquityPortfolio.example().pricer().over(NamedMarketData(data.times, prices)))


def test_names_and_instrument_types_are_distinct_checks():
    data = ThreeEquityPortfolio.example_data()
    prices = NamedMap({"aapl": data.prices["msft"], "msft": data.prices["aapl"], "googl": data.prices["googl"]})
    with pytest.raises(ValueError, match="Price instrument must be AAPL"):
        NamedPortfolioPath.from_model(ThreeEquityPortfolio.example().pricer().over(NamedMarketData(data.times, prices)))


def test_every_price_vector_uses_the_declared_date_axis():
    data = ThreeEquityPortfolio.example_data()
    dates = tuple(date(2026, 2, day) for day in range(2, 7))
    with pytest.raises(ValueError, match="match the observation dates"):
        NamedMarketData(dates, data.prices)
    for invalid in ((), (dates[0],), (dates[1], dates[0]), (dates[0], dates[0])):
        with pytest.raises(ValueError, match="at least two strictly increasing"):
            NamedMarketData(invalid, NamedMap({}))


def test_invalid_compilation_inputs():
    with pytest.raises(ValueError, match="unspecified inputs"):
        NamedPortfolio.define(Basket)("missing").pricer()
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
    with pytest.raises(ValueError, match="independent symbolic fields"):
        portfolio.pricer()


def test_currency_needs_an_explicit_conversion_model():
    class EuroMembers(EquityHoldings):
        euro = equity.component(DummyEquityProvider(symbol="EXAMPLE", currency="EUR"))

    EuroBasket = EquityBasket.define(EuroMembers)
    portfolio = NamedPortfolio.define(EuroBasket)(
        "mixed currencies",
        basket=EuroBasket("basket", members=EuroMembers("members", euro=EquityPosition("euro", quantity=1))),
        cash=USDCash("cash", amount=1),
    )
    with pytest.raises(ValueError, match="requires USD"):
        portfolio.pricer()


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
    declared: EquityPosition = equity.component(provider)
    assert declared
    _wrong_holding: USDCash = equity.component(provider)  # type: ignore[assignment]
    equity.component(equity)  # type: ignore[arg-type]
    pricer = Portfolio.example().pricer()
    evaluated: Portfolio = pricer.evaluate(MarketSnapshot(timestamp("2026-01-05T16:01:00Z"), NamedMap({})))
    member: EquityPosition = evaluated.basket.members.aapl
    assert member
    extended: ThreeEquityPortfolio = (
        ThreeEquityPortfolio.example().pricer().over(ThreeEquityPortfolio.example_data()).at(0)
    )
    assert extended.basket.members.googl
    pricer.evaluate(NamedMap({}))  # type: ignore[arg-type]
    evaluated.basket.holding(USDCash("cash"))  # type: ignore[arg-type]
    _wrong_model: EquityPosition = pricer.evaluate(  # type: ignore[assignment]
        MarketSnapshot(timestamp("2026-01-05T16:01:00Z"), NamedMap({}))
    )
