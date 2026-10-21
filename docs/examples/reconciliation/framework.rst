A small reconciliation framework
================================

Try common entry mistakes against a trusted output, then ask which input would
make the equations fit. The experimental ``examples.reconcile`` helper reuses
any prepared ``ModelProblem``, from an invoice line to a nested pricing model:

* ``Reconciliation(problem, suspects)`` selects the observations that may change.
* ``scan()`` tests concrete entry-error hypotheses, one field at a time.
* ``infer(field)`` releases one suspect observation for inverse solving.

Try common mistakes first
-------------------------

``Suspect(field)`` uses ``DECIMAL_SHIFTS``: factors 10, 1/10, 100, and 1/100.
Supply labelled ``Scale`` hypotheses to configure a field's quote units, lot
size, or sign convention.

Here a price of 1200 may have been entered in GBX where GBP was expected:

>>> from sympy import Rational
>>> from examples.applications import InvoiceLine
>>> from examples.reconcile import DECIMAL_SHIFTS, Reconciliation, Scale, Suspect
>>> from modello import inspect_model
>>> problem = inspect_model(InvoiceLine, "price units", quantity=10, unit_price=1200, amount=120)
>>> price = problem.field("unit_price")
>>> study = Reconciliation(problem, [
...     Suspect(price, (*DECIMAL_SHIFTS, Scale("GBX quoted as GBP", Rational(1, 100)))),
... ])
>>> report = study.scan()
>>> report.baseline.status, len(report.trials), len(report.matches)
('inconsistent', 4, 1)
>>> match, = report.matches
>>> match.original, match.proposed
(1200, 12)
>>> [mutation.label for mutation in match.mutations]
['two extra zeros (/100)', 'GBX quoted as GBP']

Both labels describe the same correction, so they share one checked trial.
They are alternative explanations, without additional evidence for either cause.
Each match carries its trial problem, ready for inspection or selection as a
complete model instance:

>>> corrected = match.problem.solutions().one()
>>> corrected.unit_price, corrected.amount
(12, 120)

A configured lot size works the same way:

>>> problem = inspect_model(InvoiceLine, "lot size", quantity=2, unit_price=3, amount=150)
>>> quantity = problem.field("quantity")
>>> study = Reconciliation(problem, [Suspect(quantity, [Scale("25 items per lot", 25)])])
>>> match, = study.scan().matches
>>> match.original, match.proposed
(2, 50)

Factors must be exact, nonzero rationals different from one. Use ``Rational``
for decimals. ``Scale("opposite side", -1)`` explicitly enables sign reversal.
The caller supplies unit and lot conventions; the helper does not infer them.

Use inverse solving when needed
-------------------------------

If the configured checks find no match, ask for the value required by the
equations. This stage is explicit and can also retain multiple answers:

>>> problem = inspect_model(InvoiceLine, "unusual break", quantity=2, unit_price=3, amount=21)
>>> quantity = problem.field("quantity")
>>> study = Reconciliation(problem, [Suspect(quantity)])
>>> report = study.scan()
>>> len(report.matches), len(report.unresolved)
(0, 0)
>>> family = study.infer(quantity)
>>> family.values(quantity)
{7}

``family.one()`` returns a checked model instance when exactly one complete
numerical solution can be established. Otherwise it raises, retaining the
family for further inspection. See :doc:`/solutions`.

Read the evidence
-----------------

``report.baseline`` assesses the original observations. If a complete checked
solution already exists, ``scan()`` returns no trials. Otherwise
``report.trials`` retains each distinct changed value and its ``analysis``:

.. list-table::
   :header-rows: 1

   * - Status
     - Meaning
   * - ``consistent``
     - A complete set of values satisfies every constraint; included in ``report.matches``.
   * - ``inconsistent``
     - An exact contradiction or domain violation rejects the trial.
   * - ``unknown``
     - Neither a complete solution nor a contradiction was established; included in ``report.unresolved``.

A trial contains ``field``, ``original``, ``proposed``, ``mutations``, its modified
``problem``, and ``analysis``. ``trial.problem.solutions()`` can explore its
states. Several changes can fit the same output; the report keeps every match
and undecided trial in declaration order, with no likelihood ranking. Confirm
the cause against source records.
An undecided baseline or trial does not establish that a record is wrong.
See :doc:`/diagnostics` for the evidence guarantees.

Choose what may change
-----------------------

Select actual fields with one directly recorded, exact, finite real observation.
Configuration is captured once, including supplied lists or generators.
Full component paths select nested inputs: in the CDS example,
``problem.field("closing", "terms", "signed_notional")`` identifies the same
quantity as the top-level notional alias. Declare that quantity once.
Each trial changes only that scalar observation; ``infer(field)`` removes it.
Siblings, other inputs,
definitions, and domains remain fixed, including copied inferred results.
This differs from ``problem.without("component")``, which removes a whole input
argument group.

A copied valuation calculated from an incorrect input can therefore still
conflict after that input changes. Prepare from original observations with
symbolic dependent values, as in the :doc:`CDS example <index>`.
The source problem, component instances, and external records remain unchanged.

The scan uses exact equalities and tests individual changes. Zero inputs produce
no scale trials; several errors may need changes together. Tolerances,
combinations, and probabilities are outside this helper. Bounded observations
can be explored separately with solution families.
