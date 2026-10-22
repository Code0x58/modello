"""Explore a two-asset mean–variance portfolio in the optional browser GUI."""

from sympy import Expr, Rational, sqrt

from modello import InstanceDummy, Modello, component
from modello.gui import FieldInfo, ModelInfo, Scenario, create_app


class Asset(Modello):
    """Annual return and risk assumptions for one asset class."""

    expected_return = InstanceDummy("return", real=True)
    volatility = InstanceDummy("volatility", positive=True)
    __gui__ = ModelInfo(
        fields={
            "expected_return": FieldInfo(
                title="Expected return",
                description="The assumed mean annual arithmetic return.",
                number_format="percent",
                unit="annual",
            ),
            "volatility": FieldInfo(
                title="Volatility",
                description="Annual return standard deviation; a measure of risk.",
                number_format="percent",
                unit="annual",
            ),
        }
    )


class Market(Modello):
    """The stock and bond assumptions used by the portfolio."""

    equity = component(Asset)
    bonds = component(Asset)
    correlation = InstanceDummy("correlation", real=True)
    risk_free_rate = InstanceDummy("cash_rate", real=True)
    __gui__ = ModelInfo(
        title="Market assumptions",
        fields={
            "equity": FieldInfo(title="Equities", description="A higher-return, higher-volatility asset class."),
            "bonds": FieldInfo(title="Bonds", description="A lower-return, lower-volatility asset class."),
            "equity.expected_return": FieldInfo(
                title="Equity return",
                description="Assumed annual equity return.",
                symbol="μₑ",
                number_format="percent",
                unit="annual",
            ),
            "equity.volatility": FieldInfo(
                title="Equity volatility",
                description="Equity return standard deviation.",
                symbol="σₑ",
                number_format="percent",
                unit="annual",
            ),
            "bonds.expected_return": FieldInfo(
                title="Bond return",
                description="Assumed annual bond return.",
                symbol="μᵦ",
                number_format="percent",
                unit="annual",
            ),
            "bonds.volatility": FieldInfo(
                title="Bond volatility",
                description="Bond return standard deviation.",
                symbol="σᵦ",
                number_format="percent",
                unit="annual",
            ),
            "correlation": FieldInfo(
                title="Return correlation",
                description="The correlation between equity and bond returns.",
                symbol="ρ",
                hidden=True,
            ),
            "risk_free_rate": FieldInfo(
                title="Risk-free rate",
                description="The cash return subtracted in the Sharpe ratio.",
                symbol="rᶠ",
                hidden=True,
                number_format="percent",
            ),
        },
    )


class Allocation(Modello):
    """Long-only weights that must sum to 100%."""

    equity = InstanceDummy("equity_weight", nonnegative=True)
    bonds: Expr = InstanceDummy("bond_weight", nonnegative=True)
    bonds = 1 - equity
    __gui__ = ModelInfo(
        title="Portfolio allocation",
        fields={
            "equity": FieldInfo(
                title="Equity allocation",
                description="The fraction invested in equities. Choose 0% to 100%.",
                info="Leave this free to explore the formulas, or observe a portfolio return to infer the mix.",
                symbol="w",
                number_format="percent",
            ),
            "bonds": FieldInfo(
                title="Bond allocation",
                description="The rest is invested in bonds, so the two weights sum to 100%.",
                info="Its non-negative domain also prevents an equity allocation above 100%.",
                symbol="b",
                number_format="percent",
            ),
        },
    )


class Portfolio(Modello):
    """What happens to return and risk when you change the stock–bond mix?"""

    allocation = component(Allocation)
    market = component(Market)
    capital = InstanceDummy("capital", positive=True)
    expected_return = (
        allocation.equity * market.equity.expected_return + allocation.bonds * market.bonds.expected_return
    )
    volatility = sqrt(
        (allocation.equity * market.equity.volatility) ** 2
        + (allocation.bonds * market.bonds.volatility) ** 2
        + 2
        * allocation.equity
        * allocation.bonds
        * market.correlation
        * market.equity.volatility
        * market.bonds.volatility
    )
    expected_gain = capital * expected_return
    sharpe = (expected_return - market.risk_free_rate) / volatility
    __gui__ = ModelInfo(
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
                "Explore freely",
                "Keep the equity allocation free and see how it drives the results.",
                select="allocation.equity",
            ),
            Scenario(
                "70 / 30 split",
                "Supply a 70% equity allocation and calculate the return, risk, and expected gain.",
                {"allocation.equity": Rational(7, 10)},
                select="volatility",
            ),
            Scenario(
                "Target 6% return",
                "Observe a 6% expected return and let the model infer the required allocation.",
                {"expected_return": Rational(6, 100)},
                select="allocation.equity",
            ),
        ),
        fields={
            "capital": FieldInfo(
                title="Starting capital",
                description="The amount invested for the one-year horizon.",
                symbol="C",
                unit="USD",
            ),
            "expected_return": FieldInfo(
                title="Expected return",
                description="The weighted mean of the equity and bond returns.",
                info="You can supply this as a target instead of supplying the allocation.",
                symbol="μₚ",
                number_format="percent",
                unit="annual",
            ),
            "volatility": FieldInfo(
                title="Portfolio volatility",
                description="The standard deviation of portfolio return, including diversification.",
                info="Risk is combined through variances and covariance, rather than averaging the two volatilities.",
                symbol="σₚ",
                number_format="percent",
                unit="annual",
            ),
            "expected_gain": FieldInfo(
                title="Expected gain",
                description="Starting capital multiplied by expected annual return.",
                symbol="G",
                unit="USD",
            ),
            "sharpe": FieldInfo(
                title="Sharpe ratio",
                description="Expected return above the cash rate, per unit of portfolio volatility.",
                symbol="S",
            ),
        },
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
