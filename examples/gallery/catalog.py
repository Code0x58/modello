"""Presentation and starting observations, separate from the model equations."""

from dataclasses import dataclass
from typing import Mapping

from sympy import Rational

from examples.gallery.models import (
    AccountSequence,
    GeometricDiffusion,
    LinearExposure,
    LinearTrend,
    ObservedAutoregression,
    VectorSummary,
)
from examples.geometry import RightAngleTriangle
from examples.nested_geometry import Box
from modello import Modello
from modello.gui import ExplorerValue, FieldInfo, ModelInfo, Scenario


@dataclass(frozen=True)
class Example:
    model: type[Modello]
    inputs: Mapping[str, ExplorerValue]
    info: ModelInfo


TUTORIAL = {
    "simple": Example(
        RightAngleTriangle,
        {"a": 3, "b": 4},
        ModelInfo(
            title="Scalar models",
            description="Name quantities, declare their domains, and solve the same equations in either direction.",
            info="A right-angle triangle: all sides are positive. Omitted observations remain unknown.",
            initial_field="c",
            highlights=("a", "b", "c"),
            fields={
                "a": FieldInfo(title="First leg", symbol="a"),
                "b": FieldInfo(title="Second leg", symbol="b"),
                "c": FieldInfo(title="Hypotenuse", symbol="c"),
            },
            scenarios=(
                Scenario(
                    title="Solve in reverse",
                    description="Supply the hypotenuse instead of the first leg.",
                    bindings={"a": None, "c": 5},
                    select="a",
                ),
                Scenario(
                    title="Leave a quantity unknown",
                    description="Release a leg to see the remaining symbolic relationship.",
                    bindings={"b": None},
                    select="c",
                ),
            ),
        ),
    ),
    "components": Example(
        Box,
        {"base.a": 3, "base.b": 4, "height": 12},
        ModelInfo(
            title="Component models",
            description="Compose typed child models and solve their quantities together with the parent.",
            info="A box reuses the triangle as its base. Parent observations can determine child quantities.",
            initial_field="diagonal",
            highlights=("base.c", "diagonal"),
            fields={
                "base": FieldInfo(title="Triangle component"),
                "base.a": FieldInfo(title="Base width", symbol="a"),
                "base.b": FieldInfo(title="Base depth", symbol="b"),
                "base.c": FieldInfo(title="Base diagonal", symbol="c"),
                "height": FieldInfo(title="Height", symbol="h"),
                "diagonal": FieldInfo(title="Box diagonal", symbol="d"),
            },
            scenarios=(
                Scenario(
                    title="Solve through a component",
                    description="Observe the box diagonal to infer a quantity inside its base.",
                    bindings={"base.a": None, "diagonal": 13},
                    select="base.a",
                ),
            ),
        ),
    ),
    "trend": Example(
        LinearTrend,
        {"times": [0, 1, 2, 3], "observations": [1, 2, 2, 4]},
        ModelInfo(
            title="Vector models",
            description="Let observations set vector sizes; combine indexed formulas, reductions, and derivatives.",
            info="Equal weights and distinct times give a unique least-squares line. No noise distribution is assumed.",
            initial_field="squared_error",
            highlights=("slope", "intercept", "squared_error"),
            fields={
                "times": FieldInfo(title="Observation times", symbol="t"),
                "observations": FieldInfo(title="Observed levels", symbol="y"),
                "slope": FieldInfo(title="Slope", symbol="a"),
                "intercept": FieldInfo(title="Intercept", symbol="b"),
                "fitted": FieldInfo(title="Fitted levels", symbol="ŷ"),
                "residuals": FieldInfo(title="Residuals", symbol="e"),
                "squared_error": FieldInfo(title="Sum of squared errors", symbol="L"),
                "slope_gradient": FieldInfo(title="Slope gradient", symbol="Lₐ"),
                "intercept_gradient": FieldInfo(title="Intercept gradient", symbol="Lᵦ"),
            },
            scenarios=(
                Scenario(
                    title="Fit the stationary line",
                    description="Observe both objective gradients as zero.",
                    bindings={"slope_gradient": 0, "intercept_gradient": 0},
                    select="slope",
                ),
            ),
        ),
    ),
    "sequence": Example(
        AccountSequence,
        {
            "states[0].quantity": 3,
            "states[0].cash": 1000,
            "fills": [{"quantity": 2, "price": 100, "fee": 1}, {"quantity": -1, "fee": 1}],
        },
        ModelInfo(
            title="Recurrent models",
            description="Relate typed model states with relative indices and solve from boundary observations.",
            info="Add or remove executions to change N. The terminal account is the last state.",
            initial_field="terminal.cash",
            highlights=("terminal.quantity", "terminal.cash", "total_fees"),
            fields={
                "fills": FieldInfo(title="Executions", symbol="F"),
                "fills[]": FieldInfo(title="Execution", symbol="F"),
                "fills[].quantity": FieldInfo(title="Signed quantity", symbol="u", unit="shares"),
                "fills[].price": FieldInfo(title="Price", symbol="p", unit="USD/share"),
                "fills[].fee": FieldInfo(title="Fee", symbol="c", unit="USD"),
                "fills[].notional": FieldInfo(title="Trade value", symbol="v", unit="USD"),
                "states": FieldInfo(title="Account states", symbol="A"),
                "states[]": FieldInfo(title="Account state", symbol="A"),
                "states[].quantity": FieldInfo(title="Inventory", symbol="Q", unit="shares"),
                "states[].cash": FieldInfo(title="Cash", symbol="C", unit="USD"),
                "terminal": FieldInfo(title="Terminal account"),
                "terminal.quantity": FieldInfo(title="Terminal inventory", unit="shares"),
                "terminal.cash": FieldInfo(title="Terminal cash", unit="USD"),
                "total_fees": FieldInfo(title="Total fees", symbol="K", unit="USD"),
                "law": FieldInfo(title="Account recurrence"),
            },
            scenarios=(
                Scenario(
                    title="Infer an execution price",
                    description="Terminal cash of USD 908 determines the sale price.",
                    bindings={"terminal.cash": 908},
                    select="fills[1].price",
                ),
                Scenario(
                    title="No executions",
                    description="An empty execution vector retains the initial account.",
                    bindings={"fills": []},
                    select="terminal.cash",
                ),
            ),
        ),
    ),
}

DEMOS = {
    "vectors": Example(
        VectorSummary,
        {"samples": [1, None, 3]},
        ModelInfo(
            title="Vector reductions",
            description="Sums, products, and dot products use the same scalar equations.",
            initial_field="samples",
            highlights=("total", "product", "squared_norm"),
            fields={
                "samples": FieldInfo(title="Observations", symbol="x"),
                "total": FieldInfo(title="Sum", symbol="s"),
                "product": FieldInfo(title="Product", symbol="p"),
                "squared_norm": FieldInfo(title="Sum of squares", symbol="q"),
            },
            scenarios=(
                Scenario(
                    title="Infer a missing observation",
                    description="A sum of six determines the middle observation.",
                    bindings={"total": 6},
                    select="samples[1]",
                ),
            ),
        ),
    ),
    "relative": Example(
        ObservedAutoregression,
        {"retention": Rational(1, 2), "levels": [2, 2, 1, Rational(5, 2), Rational(5, 4)]},
        ModelInfo(
            title="Scalar series",
            description="One recurrence relates each state to its predecessor and an innovation.",
            info="Observation length determines N. Innovations here are unknown inputs without a distribution.",
            initial_field="levels",
            highlights=("closing",),
            fields={
                "opening": FieldInfo(title="Initial state", symbol="T₀"),
                "retention": FieldInfo(title="Retention", symbol="ρ"),
                "innovations": FieldInfo(title="Innovations", symbol="ε"),
                "levels": FieldInfo(title="State observations", symbol="T"),
                "closing": FieldInfo(title="Terminal state", symbol="T"),
                "law": FieldInfo(title="Relative recurrence and boundary"),
            },
            scenarios=(
                Scenario(
                    title="Drive the sequence",
                    description="Replace state observations with innovations.",
                    bindings={"levels": None, "opening": 2, "innovations": [1, 0, 2, 0]},
                    select="levels",
                ),
            ),
        ),
    ),
    "exposure": Example(
        LinearExposure,
        {"opening": 2, "duration": 3},
        ModelInfo(
            title="Continuous calculus",
            description="Use native SymPy integrals in ordinary forward and inverse problems.",
            initial_field="exposure",
            highlights=("exposure",),
            fields={
                "opening": FieldInfo(title="Opening level", symbol="a"),
                "closing": FieldInfo(title="Closing level", symbol="b"),
                "duration": FieldInfo(title="Duration", symbol="T", unit="days"),
                "exposure": FieldInfo(title="Integrated exposure", symbol="A", unit="unit-days"),
            },
            scenarios=(
                Scenario(
                    title="Infer the closing level",
                    description="Nine unit-days of exposure imply a closing level of four.",
                    bindings={"exposure": 9},
                    select="closing",
                ),
            ),
        ),
    ),
    "diffusion": Example(
        GeometricDiffusion,
        {"growth": Rational(1, 10), "volatility": Rational(1, 5), "opening": 100, "duration": 1, "wiener_increment": 0},
        ModelInfo(
            title="Itô diffusion",
            description="Use the stochastic contrib extension for local calculus and conditional moments.",
            info=(
                "One standard Wiener driver. A realised increment of zero gives a path value, "
                "distinct from the conditional mean."
            ),
            initial_field="log_drift",
            highlights=("growth", "volatility", "mean"),
            fields={
                "growth": FieldInfo(title="Growth coefficient", symbol="μ"),
                "volatility": FieldInfo(title="Diffusion coefficient", symbol="σ"),
                "opening": FieldInfo(title="Opening level", symbol="S₀"),
                "duration": FieldInfo(title="Elapsed time", symbol="τ"),
                "wiener_increment": FieldInfo(title="Realised Wiener increment", symbol="ΔW"),
                "log_drift": FieldInfo(title="Log-process drift", symbol="a"),
                "log_diffusion": FieldInfo(title="Log-process diffusion", symbol="b"),
                "closing": FieldInfo(title="Conditional closing level", symbol="S"),
                "mean": FieldInfo(title="Conditional mean", symbol="m"),
                "variance": FieldInfo(title="Conditional variance", symbol="v"),
                "process": FieldInfo(title="Itô process equation"),
            },
            scenarios=(
                Scenario(
                    title="Infer the process coefficients",
                    description="Infer growth and volatility from the transformed coefficients.",
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
                    description="Zero duration and increment retain the opening state.",
                    bindings={"duration": 0, "wiener_increment": 0},
                    select="closing",
                ),
            ),
        ),
    ),
}

EXAMPLES = {**TUTORIAL, **DEMOS}

# Attach display metadata once; the model module remains independent of the UI.
for example in EXAMPLES.values():
    example.info(example.model)
