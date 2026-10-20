"""Basket structure, instrument bindings, and allocation in a shared capital base."""

from dataclasses import replace

import pytest
from sympy import Rational, simplify

from examples.compute.named_map import NamedMap
from examples.named_members import bound_component
from examples.portfolio.analytics import NamedPortfolioPath
from examples.portfolio.basket import EquityBasket, EquityHoldings, equity
from examples.portfolio.explicit import Basket, EquityMembers, MarketData, Portfolio
from examples.portfolio.models import AAPL, MSFT, Equity, EquityPosition, Quote, USDCash
from examples.portfolio.providers import DummyEquityProvider
from examples.portfolio.valuation import NamedMarketData, NamedPortfolio
from modello import BoundInstanceDummy, InstanceDummy, component


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


def test_symbolic_weights_share_the_portfolio_nav():
    portfolio = Portfolio("symbolic")
    assert portfolio.basket.capital == portfolio.nav
    assert simplify(sum(portfolio.weights.values()) + portfolio.cash_weight) == 1
    assert simplify(sum(portfolio.basket.weights.values()) - portfolio.basket.weight) == 0


def test_binding_order_follows_the_component_declaration():
    class Reordered(EquityHoldings):
        msft = equity.component(DummyEquityProvider("MSFT"))
        aapl = equity.component(DummyEquityProvider("AAPL"))

    ReorderedBasket = EquityBasket.define(Reordered)
    basket = ReorderedBasket("basket")
    assert tuple(basket.positions) == tuple(basket.providers) == tuple(basket.instruments) == ("msft", "aapl")
    assert basket.positions["aapl"].instrument == AAPL


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


def test_inherited_holding_changes_its_provider_and_component_together():
    class Extended(EquityMembers):
        msft = equity.component(DummyEquityProvider("AAPL"))
        extra = equity.component(DummyEquityProvider("AAPL"))

    basket = EquityBasket.define(Extended)("basket")
    assert tuple(basket.positions) == tuple(basket.providers) == ("aapl", "msft", "extra")
    assert basket.instruments == {"aapl": AAPL, "msft": AAPL, "extra": AAPL}
    assert Basket("original").instruments["msft"] == MSFT
    portfolio = NamedPortfolio.define(type(basket))("portfolio")
    histories = portfolio.load_market()
    assert histories["aapl"] is histories["msft"] is histories["extra"]


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
    data = MarketData.example().aapl
    path = NamedPortfolioPath.from_model(
        portfolio.pricer().over(NamedMarketData(data.close.times, NamedMap({"income": data, "growth": data})))
    )
    assert path.nav.values[0] == 500
    assert path.positions["growth"].weight.values[0] == pytest.approx(0.2)
    assert path.positions["income"].weight.values[0] == pytest.approx(0.6)
    assert path.positions["growth"].value.values != path.positions["income"].value.values


def test_provider_and_currency_are_part_of_valuation_identity():
    data = MarketData.example()
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
                        data.aapl.close.times,
                        NamedMap({"aapl": replace(data.aapl, equity=identity), "msft": data.msft}),
                    )
                )
            )


@pytest.mark.parametrize("field", ["symbol", "provider", "currency"])
@pytest.mark.parametrize("value", ["", "   ", None])
def test_instrument_identity_needs_nonempty_strings(field, value):
    with pytest.raises(ValueError, match="symbol, provider, and quote currency"):
        replace(Equity("AAPL", "synthetic"), **{field: value})
