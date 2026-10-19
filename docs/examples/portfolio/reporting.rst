Symbolic reporting, signal attribution, and utility
===================================================

The :doc:`account example <accounts>` supplies positions and postings. Reporting
adds another model layer over those observations: it expresses P&L, reconciles
it to account value, and attributes it to named signal sleeves. SymPy also
derives local risk preferences from a supplied cash utility function.

`Explore the interactive account report <report.html>`_.
It contains account and instrument tables, signal-to-order lineage, and timeline
charts. Select an instrument, toggle chart legends, or hover observations to
inspect their timestamps and values. The source data remains the three-day
synthetic example described in :ref:`account-performance`.

Generate the same self-contained report locally:

.. code-block:: console

   $ uv run --locked python -m examples.portfolio.demo --html /tmp/account.html --json /tmp/account.json

The HTML runs offline. Both outputs use the same typed ``AccountReport``;
rendering introduces no further P&L calculations. The implementation lives in
``examples/portfolio/reporting.py``, ``examples/portfolio/utility.py``, and ``examples/portfolio/report_html.py``.

Reuse the renderer with your own account report and aligned benchmark
comparison through ``render_html(report, comparison, context="...")``.
Account, benchmark, and cash-reference names come from those typed inputs;
the optional context supplies text about the source data or methodology.
The demonstration passes its synthetic-data notice here. Observation counts,
measurement dates, and availability of annualized metrics follow the actual
sample, so the same renderer works for longer research histories.

Define reporting relationships as models
----------------------------------------

``InstrumentPnL`` and ``AccountPnL`` derive from a shared ``Profit`` model.
Holding, trading, income, and fees produce gross and net investment profit.
For each interval:

.. math::

   P\&L = q_0(P_1-P_0) + \Delta q\,P_1
          - \mathrm{signed\ fill\ notional} + income - fees.

Here signed fill notional is ``sum(fill.quantity * fill.price)``. Buys add
positive quantities and notionals; sells subtract them. The account model adds
external capital and relates the result to its opening and closing NAV.

These remain ordinary symbolic models, suitable for differentiation and inverse
questions as well as numerical reporting:

>>> from sympy import diff, simplify
>>> from examples.portfolio.reporting import InstrumentPnL, AccountPnL
>>> model = InstrumentPnL("instrument")
>>> simplify(diff(model.net, model.opening_quantity) - (model.closing_price - model.opening_price))
0
>>> diff(model.net, model.fees)
-1
>>> account = AccountPnL("account", opening_nav=1000, holding=10, trading=2, income=3, fees=1, external_flows=100, residual=0)
>>> account.net, account.closing_nav
(14, 1114)

The first derivative above is P&L sensitivity to an additional opening share.
It is distinct from allocating observed P&L to the signals responsible for
trades. The report compiles resolved model expressions through ``ScalarPlan``
and evaluates them on interval vectors. It retains the expressions for
inspection under ``instrument_plan`` and ``account_plan``.

Attribute executed decisions to named signals
---------------------------------------------

A reporting convention determines which P&L belongs to a signal. The example
uses an unchanged opening reference book plus signed incremental trading
sleeves. An ``OrderAttribution`` instance links an existing typed order to
nonnegative named signal weights summing to one:

>>> from examples.portfolio.demo import example_run, research_report
>>> from examples.compute.named_map import NamedMap
>>> from examples.portfolio.reporting import OrderAttribution
>>> scenario = example_run()
>>> order = scenario.orders["rebalance-1:aapl"]
>>> attribution = OrderAttribution(order, NamedMap({"momentum": 1.0}))
>>> attribution.weights["momentum"]
1.0

For a policy combining two signals, an order could carry
``NamedMap({"value": 0.4, "momentum": 0.6})``. Declare that allocation when
constructing the decision, alongside each signal's timestamped observations.
The reporting adapter checks that the signal was available at the decision and
that linked fills match the account, instrument, side, and order timing.

Only executed shares enter a sleeve. Its shares persist into later intervals;
fees follow the order allocation. Instrument income is split according to the
signed sleeve shares held at the income event. Trades without a declared signal
link and account-level cash income remain in an explicit ``unattributed`` row.
Income paid after a position is fully closed also goes there, since the current
holding supplies no allocation basis.

Selling an opening holding creates a negative signal sleeve against the
unchanged positive reference holding. Their shares add to the actual position,
including after it reaches zero. This expresses the incremental P&L of the sell
decision relative to continuing to hold. These are accounting sleeves; account
funding and position constraints apply to the actual combined portfolio.

>>> reporting = research_report(scenario)
>>> {source: round(sum(columns["net"].values), 2) for source, columns in reporting.sources.items()}
{'opening': 55.0, 'signal:momentum': 11.9, 'unattributed': 0.0}
>>> {member: round(sum(instrument.profit["net"].values), 2) for member, instrument in reporting.instruments.items()}
{'aapl': 90.9, 'msft': -24.0}
>>> tuple(round(x, 10) for x in reporting.account["residual"].values)
(0.0, 0.0, 0.0)

The momentum sleeve earns USD 15.90 on AAPL and loses USD 4.00 on MSFT, for
USD 11.90 in total. Its MSFT result includes the effect of selling ten shares
and the subsequent price change relative to the reference holding. The opening
reference book earns USD 55.00; together they reconcile to USD 66.90 investment
P&L. The USD 500 deposit is external capital and enters neither contribution.

For a fixed executed P&L stream, the reporting model can retain a symbolic
allocation coefficient too. Using AAPL's seven executed shares, total signed
notional of USD 735.60, allocated income of USD 3.50, and USD 1 fees:

>>> from sympy import Rational, Symbol
>>> weight = Symbol("signal_weight", nonnegative=True)
>>> allocated = InstrumentPnL("allocated", opening_quantity=0, opening_price=100, closing_price=107,
...     fill_quantity=7*weight, fill_notional=Rational("735.6")*weight,
...     income=Rational("3.5")*weight, fees=weight)
>>> allocated.net == Rational("15.9") * weight
True
>>> diff(allocated.net, weight)
159/10

That derivative measures sensitivity to the declared credit allocation while
the executed stream stays fixed. Changing the underlying signal or allocation
policy could instead change orders and fills, requiring a new scenario.

This is an additive decomposition under the declared allocation rule. A policy
with interacting signals needs its own choice of allocation, such as explicit
decision weights or separate counterfactual portfolios. The symbolic identities
make the chosen calculation inspectable; they do not identify a unique causal
effect from a transaction log.

Derive risk preferences from cash utility
-----------------------------------------

For increasing utility :math:`U(W)` of cash wealth :math:`W`, the Arrow–Pratt
measures are:

.. math::

   A(W)=-\frac{U''(W)}{U'(W)}, \qquad
   R(W)=-W\frac{U''(W)}{U'(W)}.

Absolute risk aversion has units of inverse currency; relative risk aversion is
dimensionless. CARA means constant **absolute** risk aversion, while CRRA means
constant **relative** risk aversion. See `MIT's risk-preference lecture
<https://ocw.mit.edu/courses/14-123-microeconomic-theory-iii-spring-2015/81e86df52f7911ba22af879e1dfd8ec0_MIT14_123S15_attitudes.pdf>`_
for these definitions.

``CashUtility`` accepts a typed wealth symbol and a SymPy expression. The helper
families use normalizations with convenient limits:

* ``cara(alpha)`` defines ``(1 - exp(-alpha * W)) / alpha``, giving
  ``A(W) = alpha`` and ``R(W) = alpha * W``. At zero it uses linear utility.
* ``crra(gamma)`` defines ``(W**(1-gamma) - 1) / (1-gamma)``, giving
  ``A(W) = gamma / W`` and ``R(W) = gamma``. At one it uses ``log(W)``.

>>> from examples.portfolio.utility import cara, crra, CashUtility
>>> exponential = cara(0.001)
>>> exponential.expressions["absolute_risk_aversion"]
1/1000
>>> power = crra(2)
>>> power.expressions["absolute_risk_aversion"] == 2 / power.wealth
True
>>> power.expressions["relative_risk_aversion"]
2

The expressions include utility, marginal utility, curvature, and both ratios.
They are derived using `SymPy differentiation
<https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html>`_ before
numerical execution. A custom expression follows the same route:

>>> from sympy import Symbol, log
>>> wealth = Symbol("wealth", positive=True)
>>> custom = CashUtility("Log wealth", wealth, log(wealth))
>>> custom.expressions["relative_risk_aversion"]
1

The report evaluates preferences at account NAV, **including deposits and
withdrawals**. Its illustrative CARA parameter is 0.001 per USD and its CRRA
parameter is 2. Thus CARA absolute risk aversion stays at 0.001 while relative
risk aversion changes with wealth; CRRA relative risk aversion stays at 2 while
absolute risk aversion changes with wealth.

>>> reporting.utilities["cara"].values["absolute_risk_aversion"].values
(0.001, 0.001, 0.001, 0.001)
>>> reporting.utilities["crra"].values["relative_risk_aversion"].values
(2.0, 2.0, 2.0, 2.0)

These parameters express supplied preferences. They are not estimates inferred
from the account's returns, and the utility chart is not a performance index.
Utility levels depend on normalization across families; positive affine changes
to a given utility preserve both derivative ratios. Evaluation requires positive
wealth and finite outputs with positive numerical marginal utility.

The account and instrument tables, per-source timelines, and utility timelines
are views over these same observations. The reporting layer supplies inputs
which a later optimisation layer could consume without coupling accounting to
an optimiser.
