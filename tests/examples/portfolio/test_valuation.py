"""Typed portfolios, bound holdings, named valuation, and instrument identities."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pytest
from sympy import Rational

from examples.data import NamedMap
from examples.named_members import bound_component
from examples.portfolio.analytics import NamedPortfolioPath
from examples.portfolio.demo import Basket, EquityMembers, Portfolio
from examples.portfolio.models import AAPL, MSFT, EquityPosition, Quote, USDCash
from examples.portfolio.sources import DummyEquityProvider, timestamp
from examples.portfolio.valuation import (
    EquityBasket,
    EquityHoldings,
    MarketSnapshot,
    NamedMarketData,
    NamedPortfolio,
    equity,
)
from modello import BoundInstanceDummy, InstanceDummy, component

from ._data import ThreeEquityPortfolio, fixed_portfolio, market_data


def test_identified_quotes_evaluate_to_the_original_typed_portfolio():
    book = fixed_portfolio()
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
    with pytest.raises(ValueError, match="this model case"):
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


def test_basket_owns_value_and_weights_without_per_member_formulas():
    members = EquityMembers(
        "members",
        aapl=EquityPosition("apple", instrument=Quote("quote", price=100)),
        msft=EquityPosition("microsoft", quantity=20, instrument=Quote("quote", price=50)),
    )
    source = Basket("basket", members=members)
    portfolio = Portfolio("target", basket=source, cash=USDCash("cash", amount=100), nav=2100)
    assert portfolio.basket.members.aapl.quantity == 10
    assert isinstance(source.members.aapl.quantity, BoundInstanceDummy)
    assert portfolio.basket.instruments == {"aapl": AAPL, "msft": MSFT}
    assert type(portfolio.basket.members.aapl) is type(portfolio.basket.members.msft) is EquityPosition
    assert portfolio.basket.market_value == 2000
    assert portfolio.basket.capital == portfolio.nav == 2100
    assert portfolio.basket.weight == Rational(20, 21)
    assert portfolio.basket.weights == {"aapl": Rational(10, 21), "msft": Rational(10, 21)}
    assert portfolio.basket.weight + portfolio.cash_weight == 1
    assert sum(portfolio.basket.weights.values()) == portfolio.basket.weight
    apple = portfolio.basket.positions["aapl"]
    assert apple.model is portfolio.basket.members.aapl
    assert apple.instrument.symbol == "AAPL"
    assert apple.instrument.provider == "synthetic"
    assert apple.provider == DummyEquityProvider(symbol="AAPL")
    assert apple.quantity == 10 and apple.price == 100
    assert apple.weight == Rational(10, 21)


def test_basket_weight_can_constrain_an_inverse_solve():
    basket = Basket(
        "allocation",
        capital=2000,
        weight=Rational(1, 2),
        members=EquityMembers(
            "members",
            aapl=EquityPosition("apple", instrument=Quote("quote", price=100)),
            msft=EquityPosition("microsoft", quantity=0),
        ),
    )
    assert basket.members.aapl.quantity == 10
    assert basket.market_value == 1000
    assert basket.weights == {"aapl": Rational(1, 2), "msft": 0}


def test_binding_requires_the_declared_position_domain_and_type():
    with pytest.raises(TypeError, match="EquityPosition component"):

        class Mixed(EquityHoldings):
            balance = bound_component(USDCash, DummyEquityProvider("AAPL"))

    with pytest.raises(TypeError, match="only model components"):

        class Scalars(EquityHoldings):
            value = InstanceDummy("value")

    with pytest.raises(TypeError, match="EquityProvider"):
        equity.component(AAPL)
    with pytest.raises(TypeError, match="EquityProvider bindings"):

        class WrongBinding(EquityHoldings):
            aapl = bound_component(EquityPosition, AAPL)

    with pytest.raises(ValueError, match="Member names differ"):

        class Missing(EquityHoldings):
            aapl = component(EquityPosition)

    with pytest.raises(TypeError, match="derive from the holding declarations"):

        class SeparateBinding(EquityBasket[EquityMembers]):
            members = component(EquityMembers)
            providers = NamedMap({"aapl": DummyEquityProvider("MSFT"), "msft": DummyEquityProvider("AAPL")})
            market_value = EquityBasket.total(members)


def test_same_instrument_can_have_separate_holdings_and_weights():
    class Sleeves(EquityHoldings):
        growth = equity.component(DummyEquityProvider("AAPL"))
        income = equity.component(DummyEquityProvider("AAPL"))

    AppleBasket = EquityBasket.define(Sleeves)
    portfolio = NamedPortfolio.define(AppleBasket)(
        "two sleeves",
        basket=AppleBasket(
            "basket",
            members=Sleeves(
                "members", growth=EquityPosition("growth", quantity=1), income=EquityPosition("income", quantity=3)
            ),
        ),
        cash=USDCash("cash", amount=100),
    )
    data = market_data().prices["aapl"]
    path = NamedPortfolioPath.from_model(
        portfolio.pricer().over(NamedMarketData(data.close.times, NamedMap({"income": data, "growth": data})))
    )
    assert path.nav.values[0] == 500
    assert path.positions["growth"].weight.values[0] == pytest.approx(0.2)
    assert path.positions["income"].weight.values[0] == pytest.approx(0.6)
    assert path.positions["growth"].value.values != path.positions["income"].value.values


def test_provider_and_currency_are_part_of_valuation_identity():
    data = market_data()
    portfolio = Portfolio(
        "holdings",
        basket=Basket(
            "basket",
            members=EquityMembers(
                "members", aapl=EquityPosition("a", quantity=1), msft=EquityPosition("m", quantity=1)
            ),
        ),
        cash=USDCash("cash", amount=100),
    )
    for identity in (replace(AAPL, provider="other"), replace(AAPL, currency="EUR")):
        with pytest.raises(ValueError, match="Price instrument"):
            NamedPortfolioPath.from_model(
                portfolio.pricer().over(
                    NamedMarketData(
                        data.prices["aapl"].close.times,
                        NamedMap({"aapl": replace(data.prices["aapl"], equity=identity), "msft": data.prices["msft"]}),
                    )
                )
            )
