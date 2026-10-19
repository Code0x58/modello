Portfolio analytics experiment
==============================

This example explores how named models and dated price vectors can work
together. It defines AAPL and MSFT as equity types, holds them in two named
positions, and adds a USD cash balance. Holdings stay fixed; prices and weights
change across observations.

The data is synthetic. The example assumes USD quotes, nonnegative holdings,
no trades or distributions, and cash with no interest accrual. Its returns are
price returns under those assumptions.

Run the example
---------------

From a source checkout, install the development environment and run:

.. code-block:: console

   $ uv sync --locked --dev
   $ uv run --locked python -m examples.portfolio

The example and its analytics helpers live in ``examples/portfolio.py``.

Model instruments and holdings
------------------------------

An ``Equity`` has a symbolic price. ``AAPL`` and ``MSFT`` attach ticker identity
to that type. A position contains an instrument and a quantity; its market value
is quantity times price. The portfolio declares its members directly:

.. code-block:: python

   class Portfolio(Modello):
       aapl = component(ApplePosition)
       msft = component(MicrosoftPosition)
       cash = component(USDCash)
       nav = InstanceDummy("nav", positive=True)
       nav = aapl.market_value + msft.market_value + cash.amount
       aapl_weight = aapl.market_value / nav
       msft_weight = msft.market_value / nav
       cash_weight = cash.amount / nav

Create a portfolio with known quantities and cash, leaving prices symbolic:

>>> from examples.portfolio import PortfolioPricer, example_market_data, example_portfolio
>>> holdings = example_portfolio()
>>> holdings.aapl.instrument.ticker, holdings.aapl.quantity
('AAPL', 10)
>>> holdings.msft.instrument.ticker, holdings.msft.quantity
('MSFT', 20)
>>> holdings.cash.amount
100

The same symbolic relationships support inverse questions. Supplying prices,
the MSFT holding, cash, and a target ``nav`` lets Modello determine the AAPL
quantity. The portfolio tests include that example.

Bind price vectors
------------------

``Series`` owns an immutable vector and its ordered observation dates.
``EquityPrices[AAPL]`` pairs a price series with its equity type. ``MarketData``
has named ``aapl`` and ``msft`` fields and requires their dates to match exactly.
This keeps instrument identity and time alignment available during calculation.

>>> data = example_market_data()
>>> data.aapl.close.values
(100.0, 102.0, 98.0, 104.0, 103.0)
>>> data.aapl.close.as_vector().shape
(5, 1)
>>> pricer = PortfolioPricer(holdings)
>>> path = pricer(data)
>>> path.nav.values
(2100.0, 2100.0, 2040.0, 2180.0, 2210.0)
>>> round(path.weights["aapl"].values[0], 4), round(path.weights["aapl"].values[-1], 4)
(0.4762, 0.4661)

The pricer compiles the model's scalar formulas once, then evaluates that
calculation at each pair of prices. It can be reused with another observation
grid. Hold quantities and cash fixed before compiling; construct a new model and
pricer for a different allocation.

The implementation uses `SymPy lambdify
<https://docs.sympy.org/latest/modules/utilities/lambdify.html>`_ with the standard
library ``math`` backend and common-subexpression elimination. It performs
numeric batch evaluation over the observation axis, while the symbolic model
retains its scalar equations. Inputs are formulas defined in Python model code.
The numerical paths use Python floats; symbolic valuation can still use exact
SymPy arithmetic.

Reconcile returns and estimate risk
-----------------------------------

For each interval, portfolio return is the ratio of successive NAV observations
minus one. Each equity's contribution uses its weight at the **start** of that
interval:

.. math::

   r_{p,t} = \frac{V_t}{V_{t-1}} - 1
           = w_{A,t-1} r_{A,t} + w_{M,t-1} r_{M,t}.

Cash contributes zero return in this example. This identity is checked against
the value path; using final weights throughout would describe a different
allocation.

>>> result = path.analytics(periods_per_year=252)
>>> round(result.performance.total_return * 100, 4)
5.2381
>>> round(result.performance.maximum_drawdown * 100, 4)
2.8571
>>> round(result.performance.annualized_volatility * 100, 4)
64.7513

``performance.annualized_volatility`` measures the observed buy-and-hold return path.
``annualized_covariance`` is a named map of sample equity-return covariance rows,
indexed as ``result.annualized_covariance["aapl"]["msft"]`` and using ``n - 1``
degrees of freedom. Variance is multiplied by the
caller-supplied observations per year. Observations should be evenly spaced in
that trading calendar; irregular data needs resampling before annualizing. This
scaling assumes comparable, uncorrelated periods.

These five synthetic observations provide four daily returns. For CAGR, Sharpe,
and a comparison with an independently supplied total-return index, see
:ref:`the account performance example <account-performance>`. It uses the same
statistics and makes the short measurement window visible in its report.

``latest_weights_risk`` asks a separate question: what volatility does that same
sample covariance imply for the portfolio's latest weights held fixed? For
annualized covariance :math:`\Sigma` and reference weights :math:`w`, it uses
:math:`\sigma = \sqrt{w^T \Sigma w}` and Euler contributions:

.. math::

   RC_i = \frac{w_i (\Sigma w)_i}{\sigma}.

The contributions add to this reference-weight volatility. The implementation
reports zero contributions when volatility is zero. The calculation follows the
standard-deviation allocation described in `Tasche's Euler allocation paper,
section 3.1 <https://arxiv.org/html/0708.2542v3#S3.SS1>`_.

>>> round(result.latest_weights_risk.volatility * 100, 4)
65.1128
>>> risk = result.latest_weights_risk
>>> round((sum(risk.contributions.values())) * 100, 4)
65.1128

The two volatility figures differ because one uses the drifting historical
weights and the other fixes the latest weights. These are sample statistics of
the supplied synthetic observations.

Execution cost
--------------

To separate symbolic setup from repeated numeric execution, run:

.. code-block:: console

   $ uv run --locked python -m examples.benchmark_portfolio

One exploratory run on Python 3.14.8 with SymPy 1.14.0 gave these rounded medians:

.. list-table::
   :header-rows: 1

   * - Work
     - Time
   * - Construct the symbolic portfolio
     - 48 ms
   * - Compile its numerical calculation
     - 1.5 ms
   * - Evaluate complete paths for 10,000 observations
     - 54 ms
   * - Construct and solve a portfolio for each of 10 observations
     - 788 ms

Data creation and risk analytics are outside these timings. The solve-per-row
baseline reuses the component instances and performs one joint solve per row;
its NAV results are checked against the compiled calculation. The benchmark uses
five measured batches after warmup, or three for the slower baseline. These
measurements motivate separating compilation from data execution.

What this suggests for a compute framework
------------------------------------------

The prototype separates three jobs: define symbolic relationships, bind aligned
data, and execute numeric calculations and reductions. Finance provides a useful
test of that separation: value is a pointwise calculation, returns shift the
time axis, covariance reduces it, and drawdown depends on a running peak.

The current scalar solver can already express instruments, positions, portfolio
value, weights, and inverse valuation. An execution layer can reuse those
expressions over vectors. Matrix inputs and time reductions would need their own
shape, axis, and execution semantics before becoming core model fields.

The tests also build a portfolio with a third explicit GOOGL position. Inherited
weights follow the new NAV expression correctly. Extending this fixed
two-instrument numerical example requires changes to its data fields,
input/output bindings, and covariance and attribution fields.

The :doc:`named-member extension <named_members>` takes the next step: it
projects positions to values and weights, reduces them to NAV, and carries the
same names through data binding and risk calculations. Further work could:

* Carry domain identity and knowledge of particular keys through static typing.
* Represent projected maps as model fields that can receive constraints.
* Describe observation and member axes explicitly in an execution plan for
  pointwise, lagged, and reduction operations.

Membership changes would produce a new definition and compiled plan. This keeps
the explicit identities while removing repeated edits to aggregate formulas.
Typing each particular member's concrete subtype may need generated
declarations or stubs alongside the common value type exposed by a map.

The broader compute questions exposed here are typed axes and units, numeric
precision, partial evaluation, reusable execution plans, and the boundary between
forward numeric calculation and inverse solving. Those can be explored
independently of a runtime container model.
