Investigate a trade PnL break
=============================

One pricing model can calculate PnL, test alternative source data, and infer
suspect inputs from a trusted output. Its symbolic equations also provide the
pricing formula for inspection. This one-period CDS example keeps those
equations fixed throughout; :doc:`framework` adds configurable entry-error checks.

.. code-block:: console

   $ python -m examples.reconciliation

.. toctree::
   :maxdepth: 1

   framework

The model and its conventions
-----------------------------

The remaining contract pays premium on survival and protection on default,
both at period end, with no accrual on default. All amounts use one currency:

.. math::

   V = N D\left[(1-R)(1-S) - \frac{C}{10000}\tau S\right].

Here :math:`N` is signed notional (positive buys protection), :math:`C` the annual
coupon in basis points, :math:`R` recovery, :math:`S` survival probability,
:math:`D` the discount factor, and :math:`\tau` the year fraction. The parent
connects identical trade terms and recovery assumptions at two marks, then adds
net cash received:

.. math::

   \mathrm{PnL}=V_{\mathrm{closing}}-V_{\mathrm{opening}}+\mathrm{cashflow}.

.. literalinclude:: ../../../examples/reconciliation.py
   :start-at: class CDSTerms
   :end-before: def sample_problem
   :language: python

Recovery plus loss fraction, and survival plus default probability, are each
nonnegative and sum to one. These domains remain constraints during trials.
Parent aliases expose notional, coupon, and recovery separately.

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

Read the formula
----------------

Derive the pricing expression from the model used for valuation and
reconciliation. A reviewer can read the formula that drives the checks:

>>> from examples.reconciliation import CDSMark
>>> from modello import inspect_model
>>> pricing = inspect_model(CDSMark, "formula")
>>> expression = pricing.solutions().expression(pricing.field("value"))
>>> print(pricing.format(expression))
-discount*red.survival*terms.coupon_bps*terms.signed_notional*year_fraction/10000 + discount*terms.signed_notional*(1 - red.recovery)*(1 - red.survival)

``pricing.format(expression)`` renders quantities as component paths;
``sympy.latex`` can typeset the expression. Printing a prepared problem shows
its nested equations and observations. The :doc:`framework contract <framework>`
covers exact inputs, copied results, and undecided trials.

A numerical pricer can supply intermediate factors, but the displayed algebra
then explains valuation conditional on those factors. Inferring a maturity,
date, or curve input requires modelling or recalculating its dependencies too.

Find the break
--------------

The fixture records 1m notional, a 100bp coupon, 40% recovery, and a -10,000 cash
posting, with trusted PnL of 51,000. Survival changes from 98% to 97%; both marks
use discount factor 1 and a one-year remaining period. Equal year fractions
isolate a market change without time decay. The cash posting is a separate fee;
no premium or protection payment occurs between marks, and trade terms stay
unchanged.

``sample_problem()`` prepares the conflicting observations for inspection.
Removing only the trusted PnL lets the same model calculate PnL from the recorded
inputs:

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

Changing notional to 10m satisfies all retained observations and domains.
Check this possible explanation against the original trade record.

Infer inputs and add evidence
-----------------------------

If the configured checks find no match, ``study.infer(field)`` releases one
suspect observation and returns the possible values as a solution family.
Here we deliberately broaden the investigation to compare explanations that
fit the same PnL:

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
is known to be 22,000. Add that observation to each family to distinguish the
three feasible explanations:

>>> from modello import FieldValue
>>> opening_value = FieldValue(problem.field("opening", "value"), 22_000)
>>> for name in ("signed_notional", "coupon_bps", "cashflow"):
...     family = broader.infer(problem.field(name)).observe(opening_value)
...     print(name, family.values(problem.field(name)))
signed_notional {10000000}
coupon_bps EmptySet
cashflow EmptySet

Only notional survives these individual-release trials. Compare the suggestion
with independent trade records before changing an entry. An inferred family
can also retain several or unresolved states; see :doc:`/solutions`.
For uncertain outputs, explore :doc:`bounded observations <../solution_families/index>`.

.. _cds-red-data:

Compare source components
--------------------------

Related observations from one source make a useful component. Here ``REDData``
bundles credit assumptions associated with a RED entity–obligation pair. Swap
the component to check whether the break comes from using the previous pair's
parameters, with the confirmed 10m notional and other observations fixed:

>>> from sympy import Rational
>>> from examples.reconciliation import CDSMark, CDSPnL, REDData
>>> from modello import inspect_model
>>> for label, data in (
...     ("previous", REDData("DEMO:PAIR:OLD", recovery=Rational(2, 5), survival=Rational(39, 40))),
...     ("current", REDData("DEMO:PAIR:CURRENT", recovery=Rational(2, 5), survival=Rational(97, 100))),
... ):
...     trial = inspect_model(
...         CDSPnL, "RED comparison",
...         opening=CDSMark(
...             "opening", red=REDData("opening", survival=Rational(98, 100)),
...             discount=1, year_fraction=1,
...         ),
...         closing=CDSMark("closing", red=data, discount=1, year_fraction=1),
...         signed_notional=10_000_000, coupon_bps=100, recovery=Rational(2, 5),
...         cashflow=-10_000, pnl=51_000,
...     )
...     calculated = trial.without("pnl").solutions().one()
...     print(label, calculated.pnl, trial.analyse().status)
previous 20500 inconsistent
current 51000 consistent

The current parameters fit the trusted PnL; the previous parameters do not.
Both evaluations use the same trade terms, market factors, cashflow, and
pricing equations. Only the closing RED data component changes.

All values are shown here and are synthetic. The ``DEMO:`` names label the
examples; they are not RED codes or symbolic constraints. Changing a label
alone cannot change the price, and matching PnL is a hypothesis to confirm
against independent records.

RED distinguishes entity codes from entity–obligation pair codes; a pair can
change while the entity stays the same (see `Markit's identifier description
<https://www.esma.europa.eu/sites/default/files/1-Markit_1.pdf>`_). An unchanged
code is not a complete contract description either: sovereign CDS can have
different 2003 and 2014 ISDA definitions (`ESRB report, section 3.3
<https://www.esrb.europa.eu/pub/pdf/reports/esrb.report250411_creditdefaultswaps.en.pdf>`_).
Check the contract and source date independently. This example does not model
those legal payoff differences or assume an automatic RED change.
