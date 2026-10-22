"""Explore a two-asset portfolio with metadata beside its model declarations."""

from sympy import Expr, Rational, sqrt

from modello import InstanceDummy, Modello, component
from modello.gui import ModelInfo, Scenario, create_app, gui


class Equities(Modello):
    """Annual return and risk assumptions for equities."""

    expected_return = gui(
        InstanceDummy("return", real=True),
        title="Equity return",
        description="Assumed annual equity return.",
        symbol="μₑ",
        number_format="percent",
        unit="annual",
    )
    volatility = gui(
        InstanceDummy("volatility", positive=True),
        title="Equity volatility",
        description="Equity return standard deviation.",
        symbol="σₑ",
        number_format="percent",
        unit="annual",
    )


class Bonds(Modello):
    """Annual return and risk assumptions for bonds."""

    expected_return = gui(
        InstanceDummy("return", real=True),
        title="Bond return",
        description="Assumed annual bond return.",
        symbol="μᵦ",
        number_format="percent",
        unit="annual",
    )
    volatility = gui(
        InstanceDummy("volatility", positive=True),
        title="Bond volatility",
        description="Bond return standard deviation.",
        symbol="σᵦ",
        number_format="percent",
        unit="annual",
    )


@ModelInfo(title="Market assumptions")
class Market(Modello):
    """The stock and bond assumptions used by the portfolio."""

    equity = gui(component(Equities), title="Equities", description="A higher-return, higher-volatility asset class.")
    bonds = gui(component(Bonds), title="Bonds", description="A lower-return, lower-volatility asset class.")
    correlation = gui(
        InstanceDummy("correlation", real=True),
        title="Return correlation",
        description="The correlation between equity and bond returns.",
        symbol="ρ",
        hidden=True,
    )
    risk_free_rate = gui(
        InstanceDummy("cash_rate", real=True),
        title="Risk-free rate",
        description="The cash return subtracted in the Sharpe ratio.",
        symbol="rᶠ",
        hidden=True,
        number_format="percent",
    )


@ModelInfo(title="Portfolio allocation")
class Allocation(Modello):
    """Long-only weights that must sum to 100%."""

    equity = gui(
        InstanceDummy("equity_weight", nonnegative=True),
        title="Equity allocation",
        description="The fraction invested in equities. Choose 0% to 100%.",
        info="Leave this free to explore the formulas, or observe a portfolio return to infer the mix.",
        symbol="w",
        number_format="percent",
    )
    bonds: Expr = gui(
        InstanceDummy("bond_weight", nonnegative=True),
        title="Bond allocation",
        description="The rest is invested in bonds, so the two weights sum to 100%.",
        info="Its non-negative domain also prevents an equity allocation above 100%.",
        symbol="b",
        number_format="percent",
    )
    bonds = 1 - equity


@ModelInfo(
    title="Portfolio lab",
    description="What happens to return and risk when you change the stock–bond mix?",
    info=(
        "A one-year mean–variance model with illustrative inputs. "
        "The background inputs are zero return correlation "
        "and a 2% risk-free rate. They stay bound when you release an allocation. "
        "This model explores the relationships; "
        "it does not optimise a portfolio or estimate returns from market data."
    ),
    initial_field="allocation.equity",
    highlights=("expected_return", "volatility", "expected_gain"),
    scenarios=(
        Scenario(
            title="Explore freely",
            description="Keep the equity allocation free and see how it drives the results.",
            select="allocation.equity",
        ),
        Scenario(
            title="70 / 30 split",
            description="Supply a 70% equity allocation and calculate the return, risk, and expected gain.",
            bindings={"allocation.equity": Rational(7, 10)},
            select="volatility",
        ),
        Scenario(
            title="Target 6% return",
            description="Observe a 6% expected return and let the model infer the required allocation.",
            bindings={"expected_return": Rational(6, 100)},
            select="allocation.equity",
        ),
        Scenario(
            title="Find a missing return",
            description="Keep a 70/30 mix, drop the assumed equity return, and infer it from a 6% portfolio return.",
            bindings={
                "allocation.equity": Rational(7, 10),
                "market.equity.expected_return": None,
                "expected_return": Rational(6, 100),
            },
            select="market.equity.expected_return",
        ),
    ),
)
class Portfolio(Modello):
    """What happens to return and risk when you change the stock–bond mix?"""

    allocation = component(Allocation)
    market = component(Market)
    capital = gui(
        InstanceDummy("capital", positive=True),
        title="Starting capital",
        description="The amount invested for the one-year horizon.",
        symbol="C",
        unit="USD",
    )
    expected_return = gui(
        allocation.equity * market.equity.expected_return + allocation.bonds * market.bonds.expected_return,
        title="Expected return",
        description="The weighted mean of the equity and bond returns.",
        info="You can supply this as a target instead of supplying the allocation.",
        symbol="μₚ",
        number_format="percent",
        unit="annual",
    )
    volatility = gui(
        sqrt(
            (allocation.equity * market.equity.volatility) ** 2
            + (allocation.bonds * market.bonds.volatility) ** 2
            + 2
            * allocation.equity
            * allocation.bonds
            * market.correlation
            * market.equity.volatility
            * market.bonds.volatility
        ),
        title="Portfolio volatility",
        description="The standard deviation of portfolio return, including diversification.",
        info="Risk is combined through variances and covariance, rather than averaging the two volatilities.",
        symbol="σₚ",
        number_format="percent",
        unit="annual",
    )
    expected_gain = gui(
        capital * expected_return,
        title="Expected gain",
        description="Starting capital multiplied by expected annual return.",
        symbol="G",
        unit="USD",
    )
    sharpe = gui(
        (expected_return - market.risk_free_rate) / volatility,
        title="Sharpe ratio",
        description="Expected return above the cash rate, per unit of portfolio volatility.",
        symbol="S",
    )


DEMO_INPUTS = {
    "capital": 100_000,
    "market.equity.expected_return": Rational(8, 100),
    "market.equity.volatility": Rational(20, 100),
    "market.bonds.expected_return": Rational(3, 100),
    "market.bonds.volatility": Rational(5, 100),
    "market.correlation": 0,
    "market.risk_free_rate": Rational(2, 100),
}


def main() -> None:
    """Start the demo at http://127.0.0.1:5058."""
    create_app(Portfolio, "Two-asset portfolio · one-year horizon", inputs=DEMO_INPUTS).run(
        host="127.0.0.1", port=5058, debug=False
    )


if __name__ == "__main__":
    main()
