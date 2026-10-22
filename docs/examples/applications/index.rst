Practical component models
==========================

Write each relationship once, then use the same composed model to calculate
outputs or infer missing inputs from observations. Named components retain
their Python model types throughout the hierarchy. Run these applications with:

.. code-block:: console

   $ python -m examples.applications

Each section keeps its models and runnable example together. Call
``main_invoice()``, ``main_production()``, or ``main_circuit()`` from
``examples.applications`` to run just that application.

Invoice reconciliation
----------------------

Calculate an invoice total from line prices, or recover a missing nested price
from a confirmed total. The invoice groups parts and labour into named components:

.. literalinclude:: ../../../examples/applications.py
   :start-at: class InvoiceLine
   :end-before: def main_invoice
   :language: python

All amounts use one currency. The synthetic tax rate applies to items and
shipping, with no rounding or jurisdiction-specific rules.

>>> from sympy import Rational
>>> from examples.applications import Invoice, InvoiceItems, InvoiceLine
>>> labour = InvoiceLine("labour", quantity=2, unit_price=30)
>>> billed = Invoice(
...     "billed",
...     items=InvoiceItems("items", parts=InvoiceLine("parts", quantity=10, unit_price=12), labour=labour),
...     shipping=5, tax_rate=Rational(1, 5),
... )
>>> billed.items.subtotal, billed.tax, billed.total
(180, 37, 222)

Trust the total and omit a suspect price. The constraint reaches through all
three levels, while the original partial component stays symbolic:

>>> partial = InvoiceLine("parts", quantity=10)
>>> reconciled = Invoice(
...     "reconciled", items=InvoiceItems("items", parts=partial, labour=labour),
...     shipping=5, tax_rate=Rational(1, 5), total=222,
... )
>>> reconciled.items.parts.unit_price
12
>>> isinstance(reconciled.items.parts, InvoiceLine)
True
>>> bool(partial.unit_price.free_symbols)
True

The :doc:`reconciliation framework <../reconciliation/framework>` first tries
common entry errors. Its examples check pence/pounds (GBX/GBP) price units and a 25-item lot
size. If both quantity and price are suspect, a total alone may admit several
explanations.

Material requirements and yield
-------------------------------

Work backwards from a packaging target to the feed required, or use measured
feed to infer yield. A batch owns mass balance; packaging owns pack count and
size. The parent equates saleable output with packed mass:

.. literalinclude:: ../../../examples/applications.py
   :start-at: class Batch
   :end-before: def main_production
   :language: python

The first ``packed_mass`` assignment aliases the packaging output; the second
equates it with batch output. There is no leftover saleable material.

>>> from examples.applications import Batch, Packaging, ProductionOrder
>>> packs = Packaging("packs", units=180, mass_per_unit=Rational(1, 2))
>>> planned = ProductionOrder(
...     "planned", batch=Batch("batch", yield_fraction=Rational(9, 10)), packaging=packs,
... )
>>> planned.batch.feed_mass, planned.batch.waste_mass
(100, 10)
>>> measured = ProductionOrder("measured", batch=Batch("batch", feed_mass=100), packaging=packs)
>>> measured.batch.yield_fraction
9/10

Masses are in kg. Nonnegative waste limits yield to at most one; packaging
requires an integer number of units. Output exceeding feed or a fractional
pack count fails the model. Leftovers or procurement minimums need additional
relationships.

Circuit measurements
--------------------

Predict current from known resistances, or infer a missing resistance from
voltage and current measurements. Each resistor keeps its Ohm's law and power
equations; the parent supplies a shared current and sums voltage drops:

.. literalinclude:: ../../../examples/applications.py
   :start-at: class Resistor
   :end-before: def main_circuit
   :language: python

>>> from examples.applications import Resistor, SeriesCircuit
>>> designed = SeriesCircuit(
...     "designed", first=Resistor("first", resistance=100),
...     second=Resistor("second", resistance=200), voltage=12,
... )
>>> designed.current, designed.first.voltage, designed.second.voltage, designed.power
(1/25, 4, 8, 12/25)
>>> measured = SeriesCircuit(
...     "measured", first=Resistor("known", resistance=100), voltage=12, current=Rational(1, 25),
... )
>>> measured.second.resistance
200

Units are amperes, volts, ohms, and watts under steady DC assumptions.
:doc:`Solution families </solutions>` can retain bounded measurements;
statistical calibration needs an explicit error model.

Choosing model boundaries
-------------------------

The same pattern suits stock balances, metered energy, and loan-payment
reconciliation: define the parts' equations, then connect them in a parent.
Quantities need explicit connections; reusing an instance copies its values
into independent scopes. All components are solved jointly, so a larger
hierarchy can increase symbolic solver cost. See :doc:`/nesting`.

Related observations from one source also make a useful component boundary.
Swap that component to compare alternative source assumptions against the same
trusted output, as in the :ref:`CDS source comparison <cds-red-data>`.

Declare units, signs, rounding, and external data conventions explicitly.
Construct a new problem for changed observations; assigning an attribute does
not recalculate the model. Copied inferred results remain constraints too.
Start from original observations with symbolic dependent values when those
results should be recalculated, as in the :doc:`CDS example <../reconciliation/index>`.
