"""Explore indexed paths with ``uv run --extra gui python -m examples.gui_timeseries``."""

from __future__ import annotations

import argparse
from typing import Mapping

from sympy import Rational

from examples.relative_series import GeometricDiffusion, MeanReversion, ObservedAutoregression
from examples.timeseries import LinearExposure, inventory_path, linear_trend, return_path, rolling_window
from modello import Modello, ScalarValue, VectorValue
from modello.gui import FieldInfo, ModelInfo, Scenario, create_app


@ModelInfo(
    title="Inventory over time",
    description="Follow signed fills from an opening state to closing inventory and cash.",
    info=(
        "Four ordered executions, each with a signed quantity and a positive price. "
        "The inventory vector includes the opening boundary and all four closing boundaries. "
        "The third fill starts unknown. Observe it directly, or infer it from closing inventory or cash. "
        "This illustrative settlement model excludes fees and income."
    ),
    initial_field="fills",
    highlights=("quantity", "cash", "turnover"),
    fields={
        "before": FieldInfo(title="Opening state"),
        "after": FieldInfo(title="Closing state"),
        "before.quantity": FieldInfo(title="Opening inventory", symbol="q₀", unit="shares"),
        "before.cash": FieldInfo(title="Opening cash", symbol="C₀", unit="USD"),
        "fills": FieldInfo(
            title="Signed fills",
            symbol="f",
            unit="shares",
            description="One signed quantity per execution. Positive buys; negative sells.",
        ),
        "prices": FieldInfo(title="Execution prices", symbol="p", unit="USD/share"),
        "quantities": FieldInfo(
            title="Inventory path",
            symbol="q",
            unit="shares",
            description="Five event boundaries: before the first fill, then after each fill.",
        ),
        "quantity": FieldInfo(title="Closing inventory", symbol="q₄", unit="shares"),
        "cash": FieldInfo(title="Closing cash", symbol="C₄", unit="USD"),
        "notional": FieldInfo(
            title="Signed trade value",
            symbol="N",
            unit="USD",
            description="The dot product of signed fills and execution prices.",
        ),
        "turnover": FieldInfo(title="Shares traded", symbol="T", unit="shares"),
    },
    scenarios=(
        Scenario(
            title="Explore the path",
            description="Keep the third fill open and inspect its effect on every later boundary.",
            select="quantities",
        ),
        Scenario(
            title="Supply every fill",
            description="Observe a third fill of three shares and calculate the closing state.",
            bindings={"fills[2]": 3},
            select="quantities",
        ),
        Scenario(
            title="Infer from inventory",
            description="Observe seven closing shares and infer the missing fill.",
            bindings={"quantity": 7},
            select="fills[2]",
        ),
        Scenario(
            title="Infer from cash",
            description="Observe USD 835 closing cash and infer the same missing fill.",
            bindings={"cash": 835},
            select="fills[2]",
        ),
    ),
)
class InventoryLab(inventory_path(4)):  # type: ignore[misc]  # Runtime-selected finite shape.
    pass


@ModelInfo(
    title="Compounding over time",
    description="See products build a wealth path, or infer a missing period return.",
    initial_field="returns",
    highlights=("after", "total_return"),
    fields={
        "before": FieldInfo(title="Opening capital", symbol="W₀", unit="USD"),
        "after": FieldInfo(title="Closing capital", symbol="W₄", unit="USD"),
        "returns": FieldInfo(title="Period returns", symbol="r", number_format="percent"),
        "factors": FieldInfo(
            title="Gross growth factors",
            symbol="g",
            description="Every factor must be positive; a return of −100% is outside this model.",
        ),
        "wealth": FieldInfo(title="Wealth path", symbol="W", unit="USD"),
        "total_return": FieldInfo(title="Compounded return", symbol="R", number_format="percent"),
        "growth": FieldInfo(title="Total growth factor", symbol="G"),
    },
    scenarios=(
        Scenario(
            title="Explore compounding",
            description="Keep one period return open and inspect the product.",
            select="wealth",
        ),
        Scenario(
            title="A flat period",
            description="Supply a zero return for the third period.",
            bindings={"returns[2]": 0},
            select="wealth",
        ),
        Scenario(
            title="Infer from closing capital",
            description="Observe USD 104.0094 at the end and infer the missing return.",
            bindings={"after": Rational(520047, 5000)},
            select="returns[2]",
        ),
    ),
)
class ReturnLab(return_path(4)):  # type: ignore[misc]  # Runtime-selected finite shape.
    pass


@ModelInfo(
    title="Rolling observations",
    description="Three complete trailing windows over five observations.",
    initial_field="samples",
    highlights=("means",),
    fields={
        "samples": FieldInfo(title="Observations", symbol="x", description="The third observation starts open."),
        "means": FieldInfo(
            title="Three-point means",
            symbol="m",
            description=(
                "Entry zero is the mean ending at observation two. Every window uses three available observations."
            ),
        ),
    },
    scenarios=(
        Scenario(
            title="Explore the windows",
            description="See which trailing means depend on the missing observation.",
            select="means",
        ),
        Scenario(
            title="Supply the observation",
            description="Supply three for the missing observation.",
            bindings={"samples[2]": 3},
            select="means",
        ),
        Scenario(
            title="Infer from a mean",
            description="Observe a middle window mean of three and infer the observation.",
            bindings={"means[1]": 3},
            select="samples[2]",
        ),
    ),
)
class WindowLab(rolling_window(5, 3)):  # type: ignore[misc]  # Runtime-selected finite shape.
    pass


@ModelInfo(
    title="Continuous exposure",
    description="Integrate an explicit linear interpolation between two levels.",
    initial_field="exposure",
    highlights=("exposure",),
    fields={
        "opening": FieldInfo(title="Opening level", symbol="a", unit="units"),
        "closing": FieldInfo(title="Closing level", symbol="b", unit="units"),
        "duration": FieldInfo(title="Duration", symbol="T", unit="days"),
        "exposure": FieldInfo(title="Integrated exposure", symbol="A", unit="unit-days"),
    },
    scenarios=(
        Scenario(title="Inspect the integral", description="Keep the closing level open and inspect the integral."),
        Scenario(
            title="Supply the closing level",
            description="Supply four units at the closing boundary.",
            bindings={"closing": 4},
            select="exposure",
        ),
        Scenario(
            title="Infer from exposure",
            description="Observe nine unit-days and infer a closing level of four units.",
            bindings={"exposure": 9},
            select="closing",
        ),
    ),
)
class ExposureLab(LinearExposure):
    pass


@ModelInfo(
    title="Calibrating a linear trend",
    description="Inspect vector residuals and fit a line by observing both objective gradients as zero.",
    info=(
        "Four equally weighted observations with explicitly supplied times. "
        "The squared-error objective is convex; distinct times determine a unique stationary line. "
        "The model retains the objective and its derivatives, so it can also describe other lines. "
        "This example supplies no stochastic model or uncertainty estimate."
    ),
    initial_field="squared_error",
    highlights=("slope", "intercept", "squared_error"),
    fields={
        "times": FieldInfo(title="Observation times", symbol="t"),
        "observations": FieldInfo(title="Observed levels", symbol="y"),
        "slope": FieldInfo(title="Slope", symbol="a"),
        "intercept": FieldInfo(title="Intercept", symbol="b"),
        "fitted": FieldInfo(title="Fitted levels", symbol="ŷ"),
        "residuals": FieldInfo(title="Residuals", symbol="e", description="Observed minus fitted level at each time."),
        "squared_error": FieldInfo(title="Sum of squared errors", symbol="L"),
        "slope_gradient": FieldInfo(title="Slope gradient", symbol="Lₐ"),
        "intercept_gradient": FieldInfo(title="Intercept gradient", symbol="Lᵦ"),
    },
    scenarios=(
        Scenario(
            title="Inspect the objective",
            description="Keep slope and intercept open and inspect the sum of squared vector residuals.",
        ),
        Scenario(
            title="Choose a line",
            description="Supply slope one and intercept one, then inspect the residuals and objective.",
            bindings={"slope": 1, "intercept": 1},
            select="residuals",
        ),
        Scenario(
            title="Fit the stationary line",
            description="Observe both gradients as zero to infer slope and intercept of nine tenths.",
            bindings={"slope_gradient": 0, "intercept_gradient": 0},
            select="slope_gradient",
        ),
    ),
)
class TrendLab(linear_trend(4)):  # type: ignore[misc]  # Runtime-selected finite shape.
    pass


@ModelInfo(
    title="Relative state equations",
    description="Follow Tₙ from Tₙ₋₁ and infer an earlier innovation from a later boundary.",
    info=(
        "The recurrence is written once in relative coordinates. Paste any nonempty state vector: "
        "its length determines "
        "N, the number of transitions. The opening state and innovations are inferred from those observations. "
        "Supplying innovations as well constrains the same path, so inconsistent observations are rejected. "
        "no probability distribution is assigned to them."
    ),
    initial_field="levels",
    highlights=("N", "closing"),
    fields={
        "opening": FieldInfo(title="Opening state", symbol="a"),
        "retention": FieldInfo(title="Retention coefficient", symbol="ρ"),
        "innovations": FieldInfo(title="Innovations", symbol="ε"),
        "levels": FieldInfo(
            title="State observations",
            symbol="v",
            description=(
                "Paste a state vector of any length. Its N + 1 entries determine the horizon N. "
                "Use ? for an unobserved state."
            ),
        ),
        "N": FieldInfo(title="Transitions in the observed window", symbol="N"),
        "T": FieldInfo(symbol="T"),
        "law": FieldInfo(title="Relative recurrence and boundary"),
        "closing": FieldInfo(title="Closing state", symbol="c"),
    },
    scenarios=(
        Scenario(
            title="Infer innovations from states",
            description="Observe five states and infer the opening boundary and four innovations.",
            select="levels",
        ),
        Scenario(
            title="A longer observation window",
            description="Seven observed states determine N = 6 without changing the recurrence.",
            bindings={"levels": [2, 3, 5, 8, 13, 21, 34]},
            select="N",
        ),
        Scenario(
            title="Drive the path with innovations",
            description="Release the state observations, then supply the opening boundary and innovations.",
            bindings={"levels": None, "opening": 2, "innovations": [1, 0, 2, 0]},
            select="levels",
        ),
        Scenario(
            title="Infer from the later boundary",
            description="Release the state observations and infer one missing innovation from the closing boundary.",
            bindings={"levels": None, "opening": 2, "innovations": [1, 0, None, 0], "closing": Rational(5, 4)},
            select="innovations[2]",
        ),
    ),
)
class RelativeLab(ObservedAutoregression):
    pass


@ModelInfo(
    title="An Itô diffusion",
    description="Derive log-process coefficients and conditional moments from one geometric Brownian law.",
    info=(
        "The law uses one standard Wiener driver. Duration uses the model's chosen time unit; "
        "the drift and diffusion coefficients use matching units. The supplied Wiener increment is a realization, "
        "not an expectation. Zero increment gives a conditional path that differs from the conditional mean."
    ),
    initial_field="log_drift",
    highlights=("growth", "volatility", "mean"),
    fields={
        "growth": FieldInfo(title="Level growth coefficient", symbol="μ"),
        "volatility": FieldInfo(title="Diffusion coefficient", symbol="σ"),
        "opening": FieldInfo(title="Opening level", symbol="S₀"),
        "duration": FieldInfo(title="Elapsed time", symbol="τ"),
        "wiener_increment": FieldInfo(title="Realized Wiener increment", symbol="ΔW"),
        "log_drift": FieldInfo(title="Log-process drift", symbol="a"),
        "log_diffusion": FieldInfo(title="Log-process diffusion", symbol="b"),
        "closing": FieldInfo(title="Conditional closing level", symbol="S"),
        "mean": FieldInfo(title="Conditional mean", symbol="m"),
        "variance": FieldInfo(title="Conditional variance", symbol="v"),
        "process": FieldInfo(title="Itô process equation"),
    },
    scenarios=(
        Scenario(
            title="Inspect the Itô correction",
            description="The log drift includes minus one half of diffusion squared.",
            select="log_drift",
        ),
        Scenario(
            title="Infer from transformed coefficients",
            description="Release both coefficients and infer them from log drift and diffusion.",
            bindings={
                "growth": None,
                "volatility": None,
                "log_drift": Rational(2, 25),
                "log_diffusion": Rational(1, 5),
            },
            select="growth",
        ),
        Scenario(
            title="An empty time interval",
            description="Zero elapsed time and zero Wiener increment retain the opening state.",
            bindings={"duration": 0, "wiener_increment": 0},
            select="closing",
        ),
    ),
)
class DiffusionLab(GeometricDiffusion):
    pass


@ModelInfo(
    title="Mean-reverting diffusion moments",
    description="Explore exact conditional moments of an affine additive Itô process.",
    initial_field="mean",
    highlights=("mean", "variance"),
    fields={
        "rate": FieldInfo(title="Mean-reversion rate", symbol="κ"),
        "equilibrium": FieldInfo(title="Equilibrium level", symbol="θ"),
        "volatility": FieldInfo(title="Diffusion coefficient", symbol="σ"),
        "opening": FieldInfo(title="Opening state", symbol="X₀"),
        "duration": FieldInfo(title="Elapsed time", symbol="τ"),
        "mean": FieldInfo(title="Conditional mean", symbol="m"),
        "variance": FieldInfo(title="Conditional variance", symbol="v"),
        "process": FieldInfo(title="Mean-reverting process equation"),
    },
    scenarios=(
        Scenario(
            title="Inspect the conditional moments",
            description="The opening state approaches the equilibrium level in conditional expectation.",
        ),
        Scenario(
            title="Release mean reversion",
            description="Set the rate to zero and recover additive Brownian moments.",
            bindings={"rate": 0},
            select="variance",
        ),
    ),
)
class ReversionLab(MeanReversion):
    pass


DEMOS: Mapping[str, tuple[type[Modello], Mapping[str, ScalarValue | VectorValue]]] = {
    "inventory": (
        InventoryLab,
        {"before.quantity": 5, "before.cash": 1000, "fills": [2, -1, None, -2], "prices": [100, 110, 105, 120]},
    ),
    "returns": (ReturnLab, {"before": 100, "returns": [Rational(1, 50), Rational(-1, 100), None, Rational(3, 100)]}),
    "rolling": (WindowLab, {"samples": [1, 2, None, 4, 5]}),
    "exposure": (ExposureLab, {"opening": 2, "duration": 3}),
    "trend": (TrendLab, {"times": [0, 1, 2, 3], "observations": [1, 2, 2, 4]}),
    "relative": (RelativeLab, {"retention": Rational(1, 2), "levels": [2, 2, 1, Rational(5, 2), Rational(5, 4)]}),
    "diffusion": (
        DiffusionLab,
        {"growth": Rational(1, 10), "volatility": Rational(1, 5), "opening": 100, "duration": 1, "wiener_increment": 0},
    ),
    "reversion": (ReversionLab, {"rate": 1, "equilibrium": 4, "volatility": 2, "opening": 1, "duration": 1}),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=DEMOS, default="inventory")
    parser.add_argument("--port", type=int, default=5059)
    args = parser.parse_args()
    model, inputs = DEMOS[args.example]
    create_app(model, inputs=inputs).run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
