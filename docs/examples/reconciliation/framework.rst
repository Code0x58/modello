A small reconciliation framework
================================

``examples.reconcile`` is an experimental helper around a ``ModelProblem``:

* ``Reconciliation(problem, suspects)`` selects the observations that may change.
* ``scan()`` tests concrete entry-error hypotheses, one field at a time.
* ``infer(field)`` releases one suspect observation for inverse solving.

Try common mistakes first
-------------------------

``Suspect(field)`` uses ``DECIMAL_SHIFTS``: factors 10, 1/10, 100, and 1/100.
Supply labelled ``Scale`` hypotheses to configure a field's quote units, lot
size, or sign convention. Configuration iterables are captured when constructed.

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

If the catalogue yields no confirmed correction, ask for the value required by
the equations. This stage is explicit and can also retain multiple answers:

>>> problem = inspect_model(InvoiceLine, "unusual break", quantity=2, unit_price=3, amount=21)
>>> quantity = problem.field("quantity")
>>> study = Reconciliation(problem, [Suspect(quantity)])
>>> report = study.scan()
>>> len(report.matches), len(report.unresolved)
(0, 0)
>>> family = study.infer(quantity)
>>> family.values(quantity)
{7}

``family.one()`` selects a unique checked model; it raises if uniqueness or
numerical materialisation cannot be established. See :doc:`/solutions`.

Read the evidence
-----------------

``report.baseline`` assesses the original observations. If already witnessed
consistent, ``scan()`` returns no trials. Otherwise ``report.trials`` retains
every distinct, nontrivial replacement and its diagnostic ``analysis``:

.. list-table::
   :header-rows: 1

   * - Status
     - Meaning
   * - ``consistent``
     - A complete checked witness exists; included in ``report.matches``.
   * - ``inconsistent``
     - An exact contradiction or domain violation rejects the trial.
   * - ``unknown``
     - Neither established; included in ``report.unresolved``.

A trial contains ``field``, ``original``, ``proposed``, ``mutations``, its modified
``problem``, and ``analysis``. ``trial.problem.solutions()`` can explore its
states. Matches prove feasibility, not uniqueness or cause. All matches and
undecided trials remain available in declaration order, without ranking.
An undecided baseline or trial does not establish that a record is wrong.
See :doc:`/diagnostics` for the evidence guarantees.

Observation boundaries
----------------------

Select actual fields with one directly recorded, exact, finite real observation.
Aliases are one quantity and cannot be declared twice. Each trial changes only
that scalar constraint; ``infer(field)`` removes it. Siblings, other inputs,
definitions, and domains remain fixed, including copied inferred results.
This differs from ``problem.without("component")``, which removes a whole input
argument group.

A copied PV inferred from the bad price can therefore block a correction.
Prepare from raw observations with symbolic dependent values, as in the
:doc:`CDS example <index>`. No source snapshot or external record is edited.

The scan uses exact equalities and tests individual changes. Zero inputs produce
no scale trials; several errors may need changes together. Tolerances,
combinations, and probabilities are outside this helper. Bounded observations
can be explored separately with solution families.
