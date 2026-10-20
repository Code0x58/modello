.. _account-performance:

7. Measure performance against a reference
==========================================

Account NAV includes deposits and withdrawals. Start performance measurement
from the unit-NAV index built in chapter 5, then compare it with a separate
index on the same observation grid. Only after that will we consider volatility
and contributions to portfolio risk.

Continue with ``path`` and the account inputs from chapter 5. This setup
recreates them when running the page independently:

.. raw:: html

   <details><summary>Recreate the account path from chapter 5</summary>

>>> from datetime import timedelta
>>> from pathlib import Path
>>> import examples.portfolio as portfolio_examples
>>> from examples.portfolio.explicit import Portfolio, Basket, EquityMembers
>>> from examples.portfolio.basket import equity
>>> from examples.portfolio.models import USDCash
>>> from examples.portfolio.account import Account
>>> from examples.portfolio.sources import CSVTransactionSource, timestamp
>>> times = tuple(timestamp(f"2026-01-{day:02d}T16:01:00Z") for day in range(5, 9))
>>> opening = Portfolio(
...     "opening",
...     basket=Basket(
...         "basket",
...         members=EquityMembers(
...             "equities",
...             aapl=equity("apple", quantity=10),
...             msft=equity("microsoft", quantity=10),
...         ),
...     ),
...     cash=USDCash("cash", amount=1000),
... )
>>> account = Account("research", times[0], opening)
>>> age = timedelta(days=4)
>>> market = opening.load_market()
>>> data_dir = Path(portfolio_examples.__file__).with_name("data")
>>> ledger = CSVTransactionSource(data_dir / "transactions.csv").load()
>>> path = account.replay(ledger, times, market=market, max_price_age=age)

.. raw:: html

   </details>

Define the observation and return statistics
--------------------------------------------

The account supplies four marks over three calendar days, giving three interval
returns. Compute total investment growth from its unit index:

>>> index = path.unit_nav
>>> len(index.simple_returns().values), (index.times[-1] - index.times[0]).days
(3, 3)
>>> round((index.values[-1] / index.values[0] - 1) * 100, 4)
2.4518

A result should retain that window alongside its statistics. Define a typed
record before deciding how to calculate each field:

.. literalinclude:: ../../../examples/portfolio/performance.py
   :pyobject: Performance
   :language: python

``measure`` calculates simple interval returns, sample variance, drawdown, and
elapsed-time growth. Sharpe uses excess returns over an explicitly supplied
cash index. The example withholds annualized CAGR and Sharpe below one full
calendar year and returns no Sharpe for zero excess-return variance:

* Total return is ``final_level / opening_level - 1``.
* CAGR is ``growth ** (365.25 / elapsed_days) - 1`` when eligible.
* Annualized volatility is ``sqrt(sample_variance(returns) * periods_per_year)``.
* Annualized Sharpe is ``mean(excess) / sample_stddev(excess) * sqrt(periods_per_year)``.
* Maximum drawdown is the largest fall from a running index peak.

.. raw:: html

   <details><summary>Inspect the complete statistics calculation</summary>

.. literalinclude:: ../../../examples/portfolio/performance.py
   :pyobject: measure
   :language: python

.. raw:: html

   </details>

The caller supplies a regular observation schedule and its annual frequency.
CAGR uses elapsed calendar time, ACT/365.25. Volatility scales sample return
variance by that frequency; a short sample's descriptive volatility is still
available, but does not establish a reliable risk estimate. Irregular
observations need an explicit resampling policy before this annual scaling.

Give each index a name and currency
-----------------------------------

An account, benchmark, and cash reference all have the same representation:

.. literalinclude:: ../../../examples/portfolio/performance.py
   :pyobject: PerformanceIndex
   :end-before:     def __post_init__
   :language: python

The constructor requires positive levels and a name/currency. Its comparison
method checks exact axis and currency agreement, measures both paths, normalises
them to 100, and derives active returns and relative wealth:

.. literalinclude:: ../../../examples/portfolio/performance.py
   :pyobject: PerformanceIndex.compare
   :dedent: 4
   :language: python

Load the independent synthetic benchmark from its ``at,level`` CSV. Define a
zero-return cash reference explicitly rather than silently choosing a Sharpe
baseline:

>>> from examples.compute.series import Series
>>> from examples.portfolio.sources import CSVIndexSource
>>> from examples.portfolio.performance import PerformanceIndex
>>> account_index = PerformanceIndex(
...     "Account, net of booked fees and including income", path.unit_nav
... )
>>> benchmark = PerformanceIndex(
...     "Synthetic broad-market total-return index",
...     CSVIndexSource(data_dir / "benchmark.csv").load(),
... )
>>> cash_reference = PerformanceIndex(
...     "Zero-return USD cash", Series(times, (100,) * len(times))
... )
>>> comparison = account_index.compare(
...     benchmark, risk_free=cash_reference, periods_per_year=252
... )
>>> (
...     round(comparison.portfolio.total_return * 100, 4),
...     round(comparison.benchmark.total_return * 100, 4),
... )
(2.4518, 3.0)
>>> round(comparison.return_difference * 100, 4), round(comparison.relative_return * 100, 4)
(-0.5482, -0.5322)
>>> comparison.portfolio.cagr, comparison.portfolio.sharpe_ratio
(None, None)

The first difference is in percentage points; relative return compares the two
wealth ratios. The account's USD 66.90 profit, its 2.4518% unitized return, and
its NAV growth answer related but different questions. The reference returned
3% over this same short synthetic window. The report must keep those facts
visible when presenting the toy portfolio.

Wrap valuation columns as portfolio paths
-----------------------------------------

For fixed holdings we can also reconcile each interval's return contribution.
Package the numerical columns already produced by chapter 3 into typed paths:

.. literalinclude:: ../../../examples/portfolio/analytics.py
   :pyobject: PositionPath
   :language: python

.. literalinclude:: ../../../examples/portfolio/analytics.py
   :pyobject: NamedPortfolioPath
   :end-before:     @property
   :language: python

``NamedPortfolioPricer`` binds quantities and cash, compiles the portfolio's
named values and weights, and checks each supplied quote identity. Its time
adapter builds these path records and uses opening weights for contributions:

.. raw:: html

   <details><summary>Inspect the valuation-to-path adapter</summary>

.. literalinclude:: ../../../examples/portfolio/valuation.py
   :pyobject: NamedPortfolioPricer
   :language: python

.. raw:: html

   </details>

Here are the fixed example's five daily observations, entered as data rather
than acquired from the intraday account feed. The opening-book constructor
repeats the quantities and cash we assembled in chapter 4:

.. raw:: html

   <details><summary>Inspect the fixed book and daily-price fixture definitions</summary>

.. literalinclude:: ../../../examples/portfolio/explicit.py
   :pyobject: Portfolio.example
   :dedent: 4
   :language: python

.. literalinclude:: ../../../examples/portfolio/explicit.py
   :pyobject: MarketData.example
   :dedent: 4
   :language: python

.. raw:: html

   </details>

``EquityPrices`` pairs each positive date-price series with its quote identity;
``MarketData.named()`` exposes those two fields as a named map on a common axis.
Use the compiled pricer on the fixed-share portfolio from the early chapters:

>>> from examples.portfolio.explicit import MarketData
>>> fixed = Portfolio.example().pricer().over(MarketData.example().named())
>>> fixed.nav.values
(2100.0, 2100.0, 2040.0, 2180.0, 2210.0)

For each interval, a position contributes its **opening** weight times its
interval price return. With zero-return cash, these contributions add to the
portfolio's return:

>>> observed = fixed.nav.simple_returns()
>>> summed = tuple(
...     sum(position.return_contribution.values[i] for position in fixed.positions.values())
...     for i in range(len(observed.values))
... )
>>> all(abs(a - b) < 1e-12 for a, b in zip(observed.values, summed))
True

This is return attribution for a fixed holding path. It is a different question
from attributing traded P&L to signals, which we will build next.

Optionally estimate fixed-weight portfolio risk
-----------------------------------------------

An asset covariance matrix has two holding-name axes. For annualized sample
covariance ``Sigma`` and reference weights ``w``, portfolio volatility is
``sqrt(w.T * Sigma * w)``. Euler contributions are
``w[i] * (Sigma * w)[i] / volatility`` and add to that volatility. The example
returns zero contributions when volatility is zero.

``analyse`` performs the time reduction, then evaluates risk at the supplied
reference weights. ``path.analytics`` supplies the latest observed weights:

.. raw:: html

   <details><summary>Inspect covariance and volatility contribution calculations</summary>

.. literalinclude:: ../../../examples/portfolio/analytics.py
   :pyobject: analyse
   :language: python

.. raw:: html

   </details>

>>> analytics = fixed.analytics(periods_per_year=252)
>>> round(analytics.performance.annualized_volatility * 100, 4)
64.7513
>>> round(analytics.latest_weights_risk.volatility * 100, 4)
65.1128
>>> round(sum(analytics.latest_weights_risk.contributions.values()) * 100, 4)
65.1128

Observed-path volatility uses drifting historical weights; reference-weight
risk holds the latest weights fixed against the sample covariance. These are
statistics of four synthetic daily returns, with covariance using ``n - 1``
degrees of freedom. Cash has zero return and risk in this example.

The :doc:`next chapter <reporting>` reuses the account's profit models to build
signal-attributed tables, utility views, and charts. An optimiser could consume
these results later; it does not belong in the accounting or evaluation core.
