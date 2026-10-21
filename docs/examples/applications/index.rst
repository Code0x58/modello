Practical component models
===========================

Component models are useful when the same relationships need to answer several
questions: calculate a total, infer a missing input, check a measurement, or
explain a discrepancy. The hierarchy supplies ownership and readable paths;
the equations connect the parts. The observations determine the solving
direction.

The runnable examples below use only Modello and SymPy:

.. code-block:: console

   $ python -m examples.applications

Choosing a useful boundary
--------------------------

A component should own a recognisable piece of mathematics: an invoice line,
a production batch, or a resistor. A parent owns the connections and the
combined result. These boundaries make both the original formulas and the
location of an inferred value understandable.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Application
     - Components and connections
     - Useful questions
   * - Billing and procurement
     - Lines, charges, and an invoice total
     - What should the bill be? Which price would explain the received total?
   * - Manufacturing
     - Batch output connected to packaged mass
     - How much feed is needed? What yield does measured output imply?
   * - Electrical checks
     - Resistors connected by shared current and summed voltage
     - What current is expected? What resistance explains the measurements?
   * - Inventory reconciliation
     - Opening stock, receipts, shipments, and closing stock
     - What missing receipt or shipment explains a physical count?
   * - Energy accounting
     - Equipment demand, operating time, and metered supply
     - What operating time or efficiency explains the bill?
   * - Loan schedules
     - Balance, interest accrual, and payment periods
     - What payment, rate, or missing posting explains the closing balance?
   * - Trade reconciliation
     - Contract terms, market marks, cashflows, and PnL
     - Which input changes could explain a trusted result? See
       :doc:`../reconciliation/index`.

The first three applications are implemented here. The other rows suggest
extensions of the same pattern; each needs its own explicit conventions.

Invoice reconciliation through three levels
--------------------------------------------

An invoice contains an item group, which contains separately named parts and
labour lines. Each line owns its quantity-price relationship:

.. literalinclude:: ../../../examples/applications.py
   :pyobject: InvoiceLine
   :language: python

.. literalinclude:: ../../../examples/applications.py
   :pyobject: InvoiceItems
   :language: python

.. literalinclude:: ../../../examples/applications.py
   :pyobject: Invoice
   :language: python

All amounts use one currency. The synthetic tax applies to both items and
shipping, with no rounding. This is a modelling example, rather than a statement
of any jurisdiction's tax rules.

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

Now trust the received total and omit a suspect unit price. The constraint
travels from the invoice through the item group into the parts line:

>>> partial = InvoiceLine("parts", quantity=10)
>>> reconciled = Invoice(
...     "reconciled", items=InvoiceItems("items", parts=partial, labour=labour),
...     shipping=5, tax_rate=Rational(1, 5), total=222,
... )
>>> reconciled.items.parts.unit_price
12
>>> bool(partial.unit_price.free_symbols)
True

The inferred price is an explanation conditional on the other observations.
If both the quantity and price are suspect, the total alone generally cannot
identify which was wrong. The original component remains unchanged.

Before freely inferring a price, the :doc:`reconciliation framework
<../reconciliation/framework>` can try common decimal shifts and configured
quote-unit corrections. The runnable application also demonstrates a GBX/GBP
price hypothesis and a 25-item lot-size hypothesis for quantity.

Material requirements and measured yield
-----------------------------------------

Production and packaging can be separate models, connected by conservation of
saleable material. The nonnegative waste field constrains yield to at most one:

.. literalinclude:: ../../../examples/applications.py
   :pyobject: Batch
   :language: python

.. literalinclude:: ../../../examples/applications.py
   :pyobject: Packaging
   :language: python

.. literalinclude:: ../../../examples/applications.py
   :pyobject: ProductionOrder
   :language: python

The first ``packed_mass`` assignment aliases the packaging output. The second
equates it with the batch's good mass. There is no leftover material in this
model; add an explicit inventory component if there can be leftovers.

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

Masses are in kg. Packaging uses a positive integer count: an implied fractional
pack fails the model rather than being silently rounded. Output exceeding feed
also fails because it requires negative waste. Procurement minimums and
discrete purchasing choices would need additional constraints or an optimiser.

Circuit design and measurement checks
--------------------------------------

A resistor owns Ohm's law and its power equation. The circuit supplies the
series connection:

.. literalinclude:: ../../../examples/applications.py
   :pyobject: Resistor
   :language: python

.. literalinclude:: ../../../examples/applications.py
   :pyobject: SeriesCircuit
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

The units are amperes, volts, ohms, and watts. These are ideal, steady DC
relationships. Measurement uncertainty can be represented with bounded
observations using the :doc:`solution-family API </solutions>`; a statistical
calibration needs a defined error model and fitting criterion.

What the approach provides, and what to add
-------------------------------------------

The strongest fit is a modest, auditable system of algebraic relationships
whose observations change between tasks. Nested components retain their types,
names, equations, and symbolic dependencies. Shared quantities require explicit
equations: supplying the same instance twice copies its values into independent
component scopes. The :doc:`nesting guide </nesting>` explains those mechanics.

Component boundaries do not partition the solve into independent subproblems.
The whole hierarchy is solved jointly, so expanding it can increase symbolic
solver cost. Keep external data loading, calendars, and reporting separate from
the equations, as the :doc:`portfolio example <../portfolio/index>` does.

Declare domains deliberately, keep units and sign conventions explicit, and
use exact values such as ``Rational`` when exploring solution families. The
library does not automatically check dimensions, convert currencies, apply
business rounding, optimise an objective, or assign probabilities to possible
answers. Mutation also does not recalculate a model.

For reconciliation, distinguish original observations from copied results.
Supplying a solved component retains its inferred values as constraints.
Removing one original scalar from elsewhere does not remove those constraints.
Rebuild from the original observations, or expose scalar observation aliases on
a parent with otherwise symbolic children. The next example uses that second
approach to make individual release trials precise.
