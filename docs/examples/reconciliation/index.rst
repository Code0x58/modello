Investigate a trade PnL break
=============================

Keep a trusted output fixed, try common entry mistakes on selected fields, then
use inverse solving if those hypotheses do not explain the break. This example
uses a one-period CDS model; the reusable API is described in :doc:`framework`.
The :doc:`reference-data scenario <reference_data>` shows a stale parameter
bundle selected through a previous RED pair, with confirmed trade terms fixed.

.. code-block:: console

   $ python -m examples.reconciliation

.. toctree::
   :maxdepth: 1

   framework
   reference_data

The model and its conventions
-----------------------------

The remaining contract pays premium on survival and protection on default,
both at period end, with no accrual on default. All amounts use one currency:

.. math::

   V = N D\left[(1-R)(1-S) - \frac{C}{10000}\tau S\right].

Here :math:`N` is signed notional (positive buys protection), :math:`C` the annual
coupon in basis points, :math:`R` recovery, :math:`S` survival probability,
:math:`D` the discount factor, and :math:`\tau` the year fraction. The parent
connects identical terms at two marks and adds net cash received:

.. math::

   \mathrm{PnL}=V_{\mathrm{closing}}-V_{\mathrm{opening}}+\mathrm{cashflow}.

.. literalinclude:: ../../../examples/reconciliation.py
   :start-at: class CDSTerms
   :end-before: def sample_problem
   :language: python

Recovery plus loss fraction, and survival plus default probability, are each
nonnegative and sum to one. These domains remain constraints during trials.
Parent aliases expose each original trade observation separately.

``risky_annuity`` is :math:`A=D S\tau`. At fixed market inputs, increasing the
coupon by 1bp changes the buyer's value by :math:`-N A/10000`. For the corrected
10m trade below, the closing coupon sensitivity is -970 currency units/bp.
This is the premium-annuity calculation behind risky PV01; scaling conventions
vary by system (see the `RPV01 reference
<https://www.mathworks.com/help/finance/cdsrpv01.html>`_). Market CS01 and
interest-rate DV01 would require credit- and rate-curve dependencies too.

Production CDS pricing also needs schedules, accrual on default, curve
calibration, and settlement conventions. See the `ISDA specifications
<https://www.cdsmodel.com/documentation.html>`_. This example isolates the
reconciliation workflow using explicit simplified cashflows.

Find the break
--------------

The fixture records 1m notional, a 100bp coupon, 40% recovery, and a -10,000 cash
posting, with trusted PnL of 51,000. Survival changes from 98% to 97%; both marks
use discount factor 1 and a one-year remaining period. Equal year fractions
isolate a market change without time decay. The cash posting is a separate fee;
no premium or protection payment occurs between marks, and trade terms stay
unchanged.

Prepare the conflicting observations without invoking construction. Removing
the PnL observation first shows the recorded trade's calculated result:

>>> from examples.reconciliation import sample_problem
>>> problem = sample_problem()
>>> recorded = problem.without("pnl").solutions().one()
>>> recorded.opening.value, recorded.closing.value, recorded.pnl
(2200, 8300, -3900)
>>> recorded.pnl - 51_000
-54900

Try common mistakes
-------------------

Select notional and coupon as suspects. PnL, recovery, cashflow, and market
observations remain fixed. Each replacement is bound before analysis starts:

>>> from examples.reconcile import Reconciliation, Suspect
>>> study = Reconciliation(problem, [
...     Suspect(problem.field("signed_notional")),
...     Suspect(problem.field("coupon_bps")),
... ])
>>> report = study.scan()
>>> len(report.trials), len(report.matches), len(report.unresolved)
(8, 1, 0)
>>> match, = report.matches
>>> match.field.path, match.original, match.proposed
(('signed_notional',), 1000000, 10000000)
>>> match.mutations[0].label
'one missing zero (x10)'
>>> corrected = match.problem.solutions().one()
>>> corrected.opening.value, corrected.closing.value, corrected.pnl
(22000, 83000, 51000)

The notional correction has a checked witness. It suggests checking the source
entry; it does not establish the true cause or modify the record.

Infer inputs and add evidence
-----------------------------

If the catalogue has no confirmed match, ``study.infer(field)`` releases one
suspect observation and returns a solution family. To show why matching PnL
alone is insufficient, explicitly broaden the investigation here:

>>> names = ("signed_notional", "coupon_bps", "recovery", "cashflow")
>>> broader = Reconciliation(problem, [Suspect(problem.field(name)) for name in names])
>>> for name in names:
...     family = broader.infer(problem.field(name))
...     print(name, family.values(problem.field(name)))
signed_notional {10000000}
coupon_bps {55000}
recovery EmptySet
cashflow {44900}

A 550% annual coupon or a different cash posting also fits this PnL; recovery
alone cannot explain it within [0, 1]. Suppose an independent opening valuation
is known to be 22,000. That evidence distinguishes the three feasible trials:

>>> from modello import FieldValue
>>> opening_value = FieldValue(problem.field("opening", "value"), 22_000)
>>> for name in ("signed_notional", "coupon_bps", "cashflow"):
...     family = broader.infer(problem.field(name)).observe(opening_value)
...     print(name, family.is_empty)
signed_notional False
coupon_bps True
cashflow True

Only notional survives these individual-release trials. Compare the suggestion
with independent trade records before changing an entry. An inferred family
can also retain several or unresolved states; see :doc:`/solutions`.

Read the formula
----------------

The pricing expression comes from the same equations used in the marks:

>>> from examples.reconciliation import CDSMark
>>> from modello import inspect_model
>>> pricing = inspect_model(CDSMark, "formula")
>>> expression = pricing.solutions().expression(pricing.field("value"))
>>> print(pricing.format(expression))
-market.discount*market.survival*market.year_fraction*terms.coupon_bps*terms.signed_notional/10000 + market.discount*terms.signed_notional*(1 - market.survival)*(1 - terms.recovery)

``print(problem)`` shows nested equations and observations; ``sympy.latex`` can
typeset an expression. The :doc:`framework contract <framework>` covers exact
inputs, copied results, and undecided trials. For uncertain outputs, explore
:doc:`bounded observations <../solution_families/index>` separately.

A numerical pricer can supply intermediate factors, but the displayed algebra
then explains valuation conditional on those factors. Inferring a maturity,
date, or curve input requires modelling or recalculating its dependencies too.
