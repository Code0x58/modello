Investigate a trade PnL break
==============================

A trusted output can be an input to the investigation. Keep that output fixed,
release a suspect observation, and ask what value would satisfy the same
equations. Repeat for other observations. The useful result is a set of
conditional explanations that operations can compare with independent records.

This example uses a deliberately small CDS model with visible pricing
equations. It demonstrates the full workflow without requiring a production
pricing engine:

.. code-block:: console

   $ python -m examples.reconciliation

Pricing assumptions and signs
------------------------------

The contract has one remaining period. On survival, its premium is paid at the
period end; on default, protection is also paid at that end. There is no premium
accrual on default. All values and cashflows use one currency. Let:

* :math:`N` be signed notional, positive for buying protection and negative for
  selling protection;
* :math:`C` be the annual coupon in basis points, so :math:`c=C/10000`;
* :math:`R` be recovery, :math:`S` survival probability to the period end,
  :math:`D` its discount factor, and :math:`\tau` the year fraction.

The value of the remaining contract is:

.. math::

   V = N D \left[(1-R)(1-S) - \frac{C}{10000}\tau S\right].

The protection and premium decomposition follows the usual CDS structure;
the `MathWorks pricing reference
<https://www.mathworks.com/help/finance/cdsprice.html>`_ describes those legs.
The payment timing and one-period assumptions here are simplified. Production
CDS pricing needs schedules, day counts, curves, accrual on default, and
settlement conventions; the `ISDA CDS Standard Model documentation
<https://www.cdsmodel.com/documentation.html>`_ supplies the relevant specifications.

.. literalinclude:: ../../../examples/reconciliation.py
   :pyobject: CDSTerms
   :language: python

.. literalinclude:: ../../../examples/reconciliation.py
   :pyobject: CreditPeriod
   :language: python

.. literalinclude:: ../../../examples/reconciliation.py
   :pyobject: CDSMark
   :language: python

Nonnegative recovery and loss fraction together enforce :math:`0\leq R\leq1`;
survival and default probability similarly enforce :math:`0\leq S\leq1`.
These are model constraints, so a release trial cannot silently escape them.

Connect two valuations and cash
-------------------------------

An unchanged trade has identical terms at its opening and closing marks.
Positive ``cashflow`` means net trade cash received during the comparison
interval, including any relevant fee or settlement. It is separate from the
remaining contract values at the two marks:

.. math::

   \mathrm{PnL} = V_{\mathrm{closing}} - V_{\mathrm{opening}} + \mathrm{cashflow}.

.. literalinclude:: ../../../examples/reconciliation.py
   :pyobject: CDSPnL
   :language: python

The parent aliases notional, coupon, and recovery to the closing terms and
equates them to the opening terms. This both connects the marks and exposes
each original trade observation as a separate constructor input. Market
observations stay inside their own components.

The synthetic observations are:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Observation
     - Value
   * - Recorded signed notional
     - 1,000,000
   * - Recorded coupon
     - 100 bps
   * - Recorded recovery
     - 40%
   * - Recorded net cashflow
     - -10,000
   * - Opening / closing survival
     - 98% / 97%
   * - Discount factor / year fraction, both marks
     - 1 / 1 year
   * - Trusted PnL
     - 51,000

The equal year fractions isolate a change in market observations; this is a
comparison of snapshots without modelling time decay. The recorded cashflow
is a separate cash posting. No actual premium or protection payment occurs
between these synthetic marks. The trade is unchanged throughout, with no FX,
new trades, amortisation, or novation.

Find the break, then release observations
------------------------------------------

Prepare the observation snapshot without asking construction to solve its
conflicting inputs. Removing the trusted PnL observation shows the recorded
trade's calculated result:

>>> from examples.reconciliation import sample_problem
>>> problem = sample_problem()
>>> recorded = problem.without("pnl").solutions().one()
>>> recorded.opening.value, recorded.closing.value, recorded.pnl
(2200, 8300, -3900)
>>> recorded.pnl - 51_000
-54900
>>> problem.analyse().status
'inconsistent'

For the investigation, start every trial from the original problem, which
includes the trusted PnL. Release exactly one recorded observation:

>>> notional_trial = problem.without("signed_notional").solutions()
>>> notional_trial.values(problem.field("signed_notional"))
{10000000}
>>> corrected = notional_trial.one()
>>> corrected.opening.terms.signed_notional, corrected.closing.terms.signed_notional
(10000000, 10000000)
>>> corrected.opening.value, corrected.closing.value, corrected.pnl
(22000, 83000, 51000)

``one()`` checks the selected state against the retained equations and domains
and returns typed model components. The original problem remains unchanged.

Repeat the same question for other fields:

>>> for name in ("coupon_bps", "recovery", "cashflow"):
...     alternatives = problem.without(name).solutions()
...     print(name, alternatives.values(problem.field(name)))
coupon_bps {55000}
recovery EmptySet
cashflow {44900}

.. list-table::
   :header-rows: 1
   :widths: 23 27 50

   * - Released observation
     - Implied value
     - Interpretation, conditional on everything else
   * - Signed notional
     - 10,000,000
     - A factor-of-ten entry error would explain the break.
   * - Coupon
     - 55,000 bps
     - A 550% annual coupon fits mathematically; compare it with the contract.
   * - Recovery
     - No feasible value
     - Recovery alone cannot explain the break within [0, 1].
   * - Net cashflow
     - +44,900
     - A different cash posting also explains the PnL; check settlement records.

PnL alone does not prove the notional was wrong. It establishes the notional
required if the other observations and this pricing model are correct.
Plausibility rules, confidence in sources, and permitted corrections belong
to the reconciliation workflow. Keep the candidate, the original observation,
the retained assumptions, and the supporting independent evidence together.

Use another known-good observation
-----------------------------------

Suppose the shadow account also establishes an opening value of 22,000.
That observation discriminates between the alternatives above:

>>> from modello import FieldValue
>>> opening_value = FieldValue(problem.field("opening", "value"), 22_000)
>>> for name in ("signed_notional", "coupon_bps", "cashflow"):
...     alternatives = problem.without(name).solutions().observe(opening_value)
...     print(name, alternatives.is_empty)
signed_notional False
coupon_bps True
cashflow True
>>> checked = notional_trial.observe(opening_value).one()
>>> checked.signed_notional
10000000

Among these individual-release trials, the notional explanation survives both
observations. Independently sourced trade confirmation is still the evidence
for changing the record; these results do not assign probabilities or update
external systems.

Read the actual formula
------------------------

The formula can be obtained from the same model used to calculate the marks:

>>> from examples.reconciliation import CDSMark
>>> from modello import inspect_model
>>> pricing = inspect_model(CDSMark, "formula")
>>> expression = pricing.solutions().expression(pricing.field("value"))
>>> print(pricing.format(expression))
-market.discount*market.survival*market.year_fraction*terms.coupon_bps*terms.signed_notional/10000 + market.discount*terms.signed_notional*(1 - market.survival)*(1 - terms.recovery)

``print(problem)`` additionally shows the complete nested equations and the
observations attached to them. ``sympy.latex(expression)`` provides mathematical
typesetting; ``problem.format`` supplies readable component paths. The symbolic
expression retains the source fields for further substitution or sensitivity
analysis.

Applying the pattern to a real reconciliation
----------------------------------------------

Keep the observations at the granularity you may release. Here
``problem.without("coupon_bps")`` removes only that scalar constructor input.
``without("opening")`` would remove the entire opening component's copied
observations. A copied, fully priced trade also carries inferred values such
as its original PV; those can keep a trial overconstrained after removing a raw
input elsewhere. Rebuild such trials from original observations, or use this
example's parent aliases with symbolic trade terms inside the mark components.

Keep trusted outputs, model definitions, domains, and trusted market data fixed.
If market data itself is suspect, define a separate investigation that explicitly
releases it. If several fields were entered incorrectly, individual releases
may all fail. Releasing several observations can also leave a continuous family
or several discrete answers. Use :doc:`solution families </solutions>` to
retain those possibilities and add further evidence; do not arbitrarily select
one. A zero legacy solver count alone is not an inconsistency proof, and some
families remain conditional or unresolved.

For rounded or uncertain outputs, use exact bounds around the observation
instead of forcing an exact equality. Use ``Rational`` for decimal observations.
The :doc:`bounded calibration example <../solution_families/index>` demonstrates
that workflow. Financial rounding and reconciliation tolerances must be stated
explicitly.

Replacing the toy pricing component with a richer symbolic model preserves the
idea, provided the observations still identify the released quantities and the
solver can handle the resulting equations. A numerical pricing engine can
supply trusted intermediate factors, but then the inspected formula explains
the algebra conditional on those factors. Inferring a date, maturity, or curve
input that changes those factors requires modelling or recalculating those
dependencies too. This example does not invert an opaque external pricer.
