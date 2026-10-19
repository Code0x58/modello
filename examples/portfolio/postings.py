"""Balance relationships for externally supplied fills and capital movements."""

from typing import ClassVar

from examples.symbolic import SymbolicModel
from modello import InstanceDummy


class FillPosting(SymbolicModel):
    """A signed fill changes held shares and spends its notional plus fees."""

    opening_quantity = InstanceDummy("opening_quantity", real=True)
    opening_cash = InstanceDummy("opening_cash", real=True)
    quantity = InstanceDummy("quantity", real=True)
    price = InstanceDummy("price", positive=True)
    fee = InstanceDummy("fee", nonnegative=True)
    closing_quantity = opening_quantity + quantity
    closing_cash = opening_cash - quantity * price - fee

    inputs: ClassVar[tuple[str, ...]] = ("opening_quantity", "opening_cash", "quantity", "price", "fee")
    outputs: ClassVar[tuple[str, ...]] = ("closing_quantity", "closing_cash")


class CapitalFlow(SymbolicModel):
    """External capital changes cash and units at the pre-flow unit price."""

    opening_cash = InstanceDummy("opening_cash", real=True)
    opening_nav = InstanceDummy("opening_nav", positive=True)
    opening_units = InstanceDummy("opening_units", positive=True)
    amount = InstanceDummy("amount", real=True)
    unit_price = opening_nav / opening_units
    issued_units = amount / unit_price
    closing_units = opening_units + issued_units
    closing_cash = opening_cash + amount

    inputs: ClassVar[tuple[str, ...]] = ("opening_cash", "opening_nav", "opening_units", "amount")
    outputs: ClassVar[tuple[str, ...]] = ("closing_cash", "closing_units")
