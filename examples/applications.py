"""Small composed models for billing, production planning, and circuit checks.

Run ``python -m examples.applications`` for forward and inverse examples.
All prices and rates are synthetic; quantities use the units in each docstring.
"""

from sympy import Rational

from modello import InstanceDummy, Modello, component


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


def main() -> None:
    """Use each set of equations with different observed quantities."""
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

    packs = Packaging("packs", units=180, mass_per_unit=Rational(1, 2))
    planned = ProductionOrder("planned", batch=Batch("batch", yield_fraction=Rational(9, 10)), packaging=packs)
    measured = ProductionOrder("measured", batch=Batch("batch", feed_mass=100), packaging=packs)
    print("Required feed (kg):", planned.batch.feed_mass)
    print("Yield implied by packed output:", measured.batch.yield_fraction)

    designed = SeriesCircuit(
        "designed", first=Resistor("first", resistance=100), second=Resistor("second", resistance=200), voltage=12
    )
    tested = SeriesCircuit("tested", first=Resistor("first", resistance=100), voltage=12, current=Rational(1, 25))
    print("Series current (A):", designed.current)
    print("Second resistance implied by the measurements (ohms):", tested.second.resistance)


if __name__ == "__main__":
    main()
