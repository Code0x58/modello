"""Small composed models for billing, production planning, and circuit checks.

Run ``python -m examples.applications`` for forward and inverse examples.
All prices and rates are synthetic; quantities use the units in each docstring.
"""

from sympy import Rational

from examples.reconcile import DECIMAL_SHIFTS, Reconciliation, Scale, Suspect
from modello import InstanceDummy, Modello, component, inspect_model

###
# Invoice example
###


class InvoiceLine(Modello):
    """Quantity times price, in one currency, before tax."""

    quantity = InstanceDummy("quantity", positive=True)
    unit_price = InstanceDummy("unit_price", nonnegative=True)
    amount = quantity * unit_price


class InvoiceItems(Modello):
    """Separately identifiable parts and labour charges."""

    parts = component(InvoiceLine)
    labour = component(InvoiceLine)
    subtotal = parts.amount + labour.amount


class Invoice(Modello):
    """An invoice with one synthetic tax rate on items and shipping."""

    items = component(InvoiceItems)
    shipping = InstanceDummy("shipping", nonnegative=True)
    tax_rate = InstanceDummy("tax_rate", nonnegative=True)
    tax = (items.subtotal + shipping) * tax_rate
    total = items.subtotal + shipping + tax


def main_invoice() -> None:
    """Calculate an invoice, infer a price, and check common entry mistakes."""
    labour = InvoiceLine("labour", quantity=2, unit_price=30)
    billed = Invoice(
        "billed",
        items=InvoiceItems("items", parts=InvoiceLine("parts", quantity=10, unit_price=12), labour=labour),
        shipping=5,
        tax_rate=Rational(1, 5),
    )
    reconciled = Invoice(
        "reconciled",
        items=InvoiceItems("items", parts=InvoiceLine("parts", quantity=10), labour=labour),
        shipping=5,
        tax_rate=Rational(1, 5),
        total=222,
    )
    print("Invoice total:", billed.total)
    print("Unit price implied by the invoice total:", reconciled.items.parts.unit_price)

    # Different cause labels can share the same numerical correction.
    price_check = inspect_model(InvoiceLine, "price units", quantity=10, unit_price=1200, amount=120)
    price_study = Reconciliation(
        price_check,
        (Suspect(price_check.field("unit_price"), (*DECIMAL_SHIFTS, Scale("GBX quoted as GBP", Rational(1, 100)))),),
    )
    for match in price_study.scan().matches:
        print("Price correction:", match.original, "->", match.proposed)
        print("Possible explanations:", "; ".join(mutation.label for mutation in match.mutations))
    lot_check = inspect_model(InvoiceLine, "lot size", quantity=2, unit_price=3, amount=150)
    lot_study = Reconciliation(
        lot_check, (Suspect(lot_check.field("quantity"), (*DECIMAL_SHIFTS, Scale("25 items per lot", 25))),)
    )
    for match in lot_study.scan().matches:
        print("Quantity correction for a 25-item lot:", match.original, "->", match.proposed)


###
# Production example
###


class Batch(Modello):
    """A production batch in kg; saleable output and waste conserve mass."""

    feed_mass = InstanceDummy("feed_mass", positive=True)
    yield_fraction = InstanceDummy("yield_fraction", positive=True)
    good_mass = feed_mass * yield_fraction
    waste_mass = InstanceDummy("waste_mass", nonnegative=True)
    waste_mass = feed_mass - good_mass


class Packaging(Modello):
    """An exact number of identical packs, with mass in kg."""

    units = InstanceDummy("units", positive=True, integer=True)
    mass_per_unit = InstanceDummy("mass_per_unit", positive=True)
    packed_mass = units * mass_per_unit


class ProductionOrder(Modello):
    """Connect batch output to packaging, with no leftover saleable material."""

    batch = component(Batch)
    packaging = component(Packaging)
    packed_mass = packaging.packed_mass
    packed_mass = batch.good_mass


def main_production() -> None:
    """Plan feed requirements or infer yield from measured output."""
    packs = Packaging("packs", units=180, mass_per_unit=Rational(1, 2))
    planned = ProductionOrder("planned", batch=Batch("batch", yield_fraction=Rational(9, 10)), packaging=packs)
    measured = ProductionOrder("measured", batch=Batch("batch", feed_mass=100), packaging=packs)
    print("Required feed (kg):", planned.batch.feed_mass)
    print("Yield implied by packed output:", measured.batch.yield_fraction)


###
# Circuit example
###


class Resistor(Modello):
    """Ideal steady DC resistor: ohms, amperes, volts, and watts."""

    resistance = InstanceDummy("resistance", positive=True)
    current = InstanceDummy("current", real=True)
    voltage = current * resistance
    power = current**2 * resistance


class SeriesCircuit(Modello):
    """Two resistors share a current; the source supplies both voltage drops."""

    first = component(Resistor)
    second = component(Resistor)
    current = first.current
    current = second.current
    voltage = first.voltage + second.voltage
    power = first.power + second.power


def main_circuit() -> None:
    """Calculate current or infer resistance from circuit measurements."""
    designed = SeriesCircuit(
        "designed", first=Resistor("first", resistance=100), second=Resistor("second", resistance=200), voltage=12
    )
    tested = SeriesCircuit("tested", first=Resistor("first", resistance=100), voltage=12, current=Rational(1, 25))
    print("Series current (A):", designed.current)
    print("Second resistance implied by the measurements (ohms):", tested.second.resistance)


def main() -> None:
    """Run all three applications."""
    main_invoice()
    main_production()
    main_circuit()


if __name__ == "__main__":
    main()
