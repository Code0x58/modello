"""The P&L equations shared by account valuation and attribution reports."""

from datetime import datetime
from typing import ClassVar

from examples.compute.named_map import NamedMap
from examples.compute.series import Series
from examples.symbolic import SymbolicModel
from modello import InstanceDummy

PROFIT_FIELDS = ("holding", "trading", "income", "fees", "gross", "net")


class Profit(SymbolicModel):
    """Currency P&L shared by instrument and account reporting models."""

    holding = InstanceDummy("holding", real=True)
    trading = InstanceDummy("trading", real=True)
    income = InstanceDummy("income", real=True)
    fees = InstanceDummy("fees", nonnegative=True)
    gross = holding + trading + income
    net = gross - fees

    inputs: ClassVar[tuple[str, ...]] = ("holding", "trading", "income", "fees")
    outputs: ClassVar[tuple[str, ...]] = PROFIT_FIELDS

    @classmethod
    def sum(
        cls, times: tuple[datetime, ...], paths: tuple[NamedMap[Series[datetime]], ...]
    ) -> NamedMap[Series[datetime]]:
        """Reduce aligned report columns; their formulas remain on the model."""
        if any(
            set(path) != set(PROFIT_FIELDS) or any(series.times != times for series in path.values()) for path in paths
        ):
            raise ValueError("Profit reports must share named columns and interval times")
        return NamedMap(
            (name, Series(times, tuple(sum(path[name].values[i] for path in paths) for i in range(len(times)))))
            for name in PROFIT_FIELDS
        )


class InstrumentPnL(Profit):
    """Exact interval mark-to-market profit, including signed virtual sleeves."""

    opening_quantity = InstanceDummy("opening_quantity", real=True)
    opening_price = InstanceDummy("opening_price", positive=True)
    closing_price = InstanceDummy("closing_price", positive=True)
    fill_quantity = InstanceDummy("fill_quantity", real=True)
    fill_notional = InstanceDummy("fill_notional", real=True)
    holding = opening_quantity * (closing_price - opening_price)
    trading = fill_quantity * closing_price - fill_notional

    inputs: ClassVar[tuple[str, ...]] = (
        "opening_quantity",
        "opening_price",
        "closing_price",
        "fill_quantity",
        "fill_notional",
        "income",
        "fees",
    )


class AccountPnL(Profit):
    """Relate investment profit and external capital to account NAV changes."""

    opening_nav = InstanceDummy("opening_nav", positive=True)
    closing_nav = InstanceDummy("closing_nav", positive=True)
    external_flows = InstanceDummy("external_flows", real=True)
    nav_change = closing_nav - opening_nav
    residual = nav_change - Profit.net - external_flows

    inputs: ClassVar[tuple[str, ...]] = (
        "holding",
        "trading",
        "income",
        "fees",
        "opening_nav",
        "closing_nav",
        "external_flows",
    )
    outputs: ClassVar[tuple[str, ...]] = (*PROFIT_FIELDS, "nav_change", "external_flows", "residual")
