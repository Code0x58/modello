"""Practical questions must work in both directions through the same components."""

import pytest
from sympy import Rational

from examples.applications import (
    Batch,
    Invoice,
    InvoiceItems,
    InvoiceLine,
    Packaging,
    ProductionOrder,
    Resistor,
    SeriesCircuit,
)
from modello import ModelSolveError


def test_invoice_total_and_missing_line_price():
    labour = InvoiceLine("labour", quantity=2, unit_price=30)
    partial = InvoiceLine("parts", quantity=10)
    reconciled = Invoice(
        "reconciled",
        items=InvoiceItems("items", parts=partial, labour=labour),
        shipping=5,
        tax_rate=Rational(1, 5),
        total=222,
    )
    assert isinstance(reconciled.items.parts, InvoiceLine)
    assert reconciled.items.parts.unit_price == 12
    assert reconciled.tax == 37
    assert partial.unit_price.free_symbols  # Completing the copy preserves the source.
    billed = Invoice(
        "billed",
        items=InvoiceItems("items", parts=InvoiceLine("parts", quantity=10, unit_price=12), labour=labour),
        shipping=5,
        tax_rate=Rational(1, 5),
    )
    assert billed.total == reconciled.total


def test_production_feed_and_observed_yield():
    packs = Packaging("packs", units=180, mass_per_unit=Rational(1, 2))
    planned = ProductionOrder("planned", batch=Batch("batch", yield_fraction=Rational(9, 10)), packaging=packs)
    measured = ProductionOrder("measured", batch=Batch("batch", feed_mass=100), packaging=packs)
    assert planned.batch.feed_mass == 100
    assert planned.batch.waste_mass == 10
    assert measured.batch.yield_fraction == Rational(9, 10)
    assert measured.batch.good_mass == measured.packaging.packed_mass == 90


def test_production_rejects_output_exceeding_feed():
    with pytest.raises(ModelSolveError) as caught:
        ProductionOrder(
            "impossible",
            batch=Batch("batch", feed_mass=80),
            packaging=Packaging("packs", units=180, mass_per_unit=Rational(1, 2)),
        )
    assert caught.value.problem.analyse().status == "inconsistent"


def test_packaging_requires_a_whole_number_of_units():
    with pytest.raises(ModelSolveError):
        ProductionOrder(
            "fractional pack",
            batch=Batch("batch", feed_mass=1, yield_fraction=1),
            packaging=Packaging("packs", mass_per_unit=Rational(3, 5)),
        )


def test_circuit_current_and_missing_resistance():
    designed = SeriesCircuit(
        "designed", first=Resistor("first", resistance=100), second=Resistor("second", resistance=200), voltage=12
    )
    assert designed.first.current == designed.second.current == Rational(1, 25)
    assert (designed.first.voltage, designed.second.voltage) == (4, 8)
    assert designed.power == Rational(12, 25)
    tested = SeriesCircuit("tested", first=Resistor("first", resistance=100), voltage=12, current=Rational(1, 25))
    assert tested.second.resistance == 200


def test_circuit_measurements_cannot_imply_a_negative_resistor():
    with pytest.raises(ModelSolveError):
        SeriesCircuit("impossible", first=Resistor("first", resistance=100), voltage=2, current=Rational(1, 25))
