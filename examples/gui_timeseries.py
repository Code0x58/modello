"""Explore indexed paths with ``uv run --extra gui python -m examples.gui_timeseries``."""

from __future__ import annotations

import argparse
from typing import Mapping

from sympy import Rational

from examples.timeseries import LinearExposure, inventory_path, return_path, rolling_window
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


DEMOS: Mapping[str, tuple[type[Modello], Mapping[str, ScalarValue | VectorValue]]] = {
    "inventory": (
        InventoryLab,
        {"before.quantity": 5, "before.cash": 1000, "fills": [2, -1, None, -2], "prices": [100, 110, 105, 120]},
    ),
    "returns": (ReturnLab, {"before": 100, "returns": [Rational(1, 50), Rational(-1, 100), None, Rational(3, 100)]}),
    "rolling": (WindowLab, {"samples": [1, 2, None, 4, 5]}),
    "exposure": (ExposureLab, {"opening": 2, "duration": 3}),
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
