A small reconciliation framework
==================================

``examples.reconcile`` supplies a reusable workflow around a recorded
``ModelProblem``. It has no trade-specific formulas or field-name heuristics.
The caller selects suspect fields and their permitted mutations. All other
observations, definitions, and domains stay fixed.

There are two explicit stages:

1. ``scan()`` tries concrete, common corrections individually and records the
   diagnostic evidence for every distinct trial.
2. ``infer(field)`` releases one declared suspect observation and returns a
   solution family, useful when the catalogue does not explain the break.

These helpers are experimental and live in the examples layer. They reuse the
existing diagnostic and solution-family APIs without changing core solving.

Declare field-specific hypotheses
----------------------------------

Use actual field records, including nested paths. The default ``DECIMAL_SHIFTS``
catalogue contains factors 10, 1/10, 100, and 1/100. Add a labelled ``Scale`` for
a known quote unit, lot size, sign convention, or other rational conversion.
Decimal numbers are represented exactly with ``Rational``.

For example, the recorded price is 1200 while ten units should cost 120. A
price quoted in GBX but treated as GBP is one possible explanation:

>>> from sympy import Rational
>>> from examples.applications import InvoiceLine
>>> from examples.reconcile import DECIMAL_SHIFTS, Reconciliation, Scale, Suspect
>>> from modello import inspect_model
>>> problem = inspect_model(InvoiceLine, "price units", quantity=10, unit_price=1200, amount=120)
>>> price = problem.field("unit_price")
>>> study = Reconciliation(problem, (
...     Suspect(price, (*DECIMAL_SHIFTS, Scale("GBX quoted as GBP", Rational(1, 100)))),
... ))
>>> report = study.scan()
>>> report.baseline.status
'inconsistent'
>>> len(report.trials), len(report.matches)
(4, 1)
>>> match, = report.matches
>>> match.original, match.proposed
(1200, 12)
>>> [mutation.label for mutation in match.mutations]
['two extra zeros (/100)', 'GBX quoted as GBP']

Two explanations produce the same price, so they share one checked trial.
Neither label establishes the cause: compare the quote units and source entry.
The two hypotheses do not count as independent evidence for the correction.

The corrected case is available explicitly from the trial snapshot:

>>> case = match.problem.solutions().one()
>>> case.quantity, case.unit_price, case.amount
(10, 12, 120)

The framework only attempts configured factors on configured fields. It does
not interpret instrument identifiers, fetch FX rates, or infer that every price
is quoted in minor currency units. A price unit conversion scales one number;
converting an entire trade currency can also require changing amounts and other
fields, which needs a separate model or explicitly designed hypothesis.

A custom lot size
------------------

For a quantity recorded in lots rather than individual items, provide the
instrument's actual lot size. Here it is 25:

>>> problem = inspect_model(InvoiceLine, "lot size", quantity=2, unit_price=3, amount=150)
>>> quantity = problem.field("quantity")
>>> study = Reconciliation(problem, (
...     Suspect(quantity, (*DECIMAL_SHIFTS, Scale("25 items per lot", 25))),
... ))
>>> match, = study.scan().matches
>>> match.original, match.proposed, match.mutations[0].label
(2, 50, '25 items per lot')

``Scale("opposite side", -1)`` can test a sign reversal when that is an allowed
interpretation of a particular field. Sign reversal is not in the default
catalogue. Factors must be exact, nonzero rationals different from one.

When common corrections do not fit
-----------------------------------

An implied value need not look like a decimal shift. In this example, none of
the four common factors explains the observed amount:

>>> problem = inspect_model(InvoiceLine, "unusual break", quantity=2, unit_price=3, amount=21)
>>> quantity = problem.field("quantity")
>>> study = Reconciliation(problem, (Suspect(quantity),))
>>> report = study.scan()
>>> len(report.matches), len(report.unresolved)
(0, 0)
>>> family = study.infer(quantity)
>>> family.values(quantity)
{7}
>>> family.one().quantity
7

In an application, the control flow can be:

.. code-block:: python

   report = study.scan()
   if report.baseline.status == "consistent":
       print("The recorded observations already admit a checked state.")
   elif report.matches:
       for match in report.matches:
           print(match.field.path, match.original, match.proposed, match.mutations)
   else:
       for trial in report.unresolved:
           print("Undecided:", trial.field.path, trial.proposed, trial.analysis.reason)
       for suspect in study.suspects:
           family = study.infer(suspect.field)
           print(suspect.field.path, family.values(suspect.field))

Inverse inference can leave no feasible value, several values, or an unresolved
family. It does not select a branch or manufacture a unique correction. See
the :doc:`solution-family guide </solutions>` before selecting states.

Reading a report
-----------------

``report.baseline`` records the diagnostic assessment of the original problem.
An already witnessed baseline produces no mutation trials. Otherwise each
unique replacement is checked against the original snapshot with just that
observation changed. A zero-valued input produces no scale trials, because
scaling zero would not change the record.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Trial outcome
     - Meaning
   * - ``consistent``
     - A complete numerical witness passes all retained equations and domains.
       The trial appears in ``report.matches``. It need not be the only state.
   * - ``inconsistent``
     - An exact contradiction or domain violation rejects this concrete trial.
   * - ``unknown``
     - Analysis established neither a witness nor a contradiction. The trial
       appears in ``report.unresolved`` and must not be called a rejected cause.

``report.trials`` includes all outcomes. Each trial retains ``field``,
``original``, ``proposed``, all matching ``mutations`` labels, the diagnostic
``analysis``, and its modified ``problem``. Declaration order controls display
order, without ranking likelihood or confidence. The :doc:`diagnostic reference
</diagnostics>` describes the evidence behind the three statuses.

Preserving the observation boundary
------------------------------------

Every suspect must identify exactly one directly recorded scalar observation
with an exact, finite real value. A derived but unobserved field is not eligible.
Aliases identify the same quantity, so declaring them twice is rejected.
Multiple input observations of the same quantity are rejected as ambiguous.

A nested scalar can be selected even when it belongs to a component argument
with several inputs. The trial changes only its original scalar constraint;
siblings in the same constructor argument stay fixed. ``infer(field)`` likewise
removes just that scalar observation. This is deliberately finer-grained than
``ModelProblem.without("component")``, which removes the entire argument group.

Copied inferred outputs also stay fixed. If a supplied component already
computed its value from the bad price, that old value can block both mutation
trials and inverse inference. Prepare from raw observations with symbolic
dependent terms, or expose observations through parent aliases as in the
:doc:`CDS example <index>`. The helper never silently discards inferred values
or other evidence to make a hypothesis fit.

Multiple suspect fields are tested individually against the original problem.
All matches are reported. If two fields need correction together, these trials
can all fail; there is no automatic combinatorial search. Exact equality is the
current scan criterion. Round-off tolerances and probabilistic ranking are not
implicit. Bounded observations can be explored separately with solution
families. Nothing here edits source records or an external system.
