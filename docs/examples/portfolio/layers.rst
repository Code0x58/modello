Portfolio research layers
==========================

The portfolio examples build on the symbolic modeling API in :mod:`modello`.
The core package defines models, typed components, independent instance symbols,
and joint equation solving. Its runtime dependency is SymPy.

The research code lives under ``examples/`` and is included in the installed
wheel. Install the examples extra and run the walkthroughs:

.. code-block:: console

   $ pip install "modello[examples]"
   $ python -m examples.portfolio
   $ python -m examples.portfolio.named
   $ python -m examples.portfolio.demo --html /tmp/account.html --json /tmp/account.json

The wheel includes both the ``modello`` and ``examples`` packages, including
the portfolio's CSV fixtures. Example modules load when explicitly imported.
The current examples need only the standard library and the core SymPy
dependency; the ``examples`` extra groups any future additional requirements.
Readers can also use a checkout or source distribution to adapt the examples.

Start with numeric data
-----------------------

``examples/compute/named_map.py`` and ``examples/compute/series.py`` use the
Python standard library. They can represent and transform data independently
of symbolic modeling or a financial application:

>>> from datetime import date
>>> from examples.compute.named_map import NamedMap
>>> from examples.compute.series import Series
>>> times = (date(2026, 1, 5), date(2026, 1, 6))
>>> data = NamedMap({"sensor": Series(times, (10, 12))})
>>> dict(data.map(lambda series: series.lag().values))
{'sensor': (10.0,)}

``NamedMap`` preserves names through projection and aligns combinations by
name. ``Series`` owns finite numeric values and an ordered ``times`` axis,
using dates or timezone-aware timestamps. It provides exact alignment, lag,
rolling means, and explicit backward sampling. Array or matrix conversion
belongs at the consumer boundary; for example, SymPy accepts a series' values
through ``ImmutableMatrix(series.values)``.

Interval changes and cumulative totals also belong to this numeric layer.
``differences(periods=1)`` labels each change with its ending observation and
omits the warmup prefix, just like ``lag``. Periods count observations, including
on an irregular axis; these operations do not insert intermediate timestamps.
``cumulative_sum()`` starts from zero and preserves the input times:

>>> levels = Series(times, (100, 112))
>>> change = levels.differences()
>>> change.times, change.values
((datetime.date(2026, 1, 6),), (12.0,))
>>> change.cumulative_sum().map(lambda value: value + levels.values[0]).values
(112.0,)

Account P&L uses these interval changes; cumulative report charts use the same
accumulation operation. Adding an opening balance or a chart's zero point is
an explicit choice at the consumer boundary.

Acquire typed inputs explicitly
--------------------------------

``examples/compute/providers.py`` defines the abstract ``Provider[T]`` contract:
a configured request whose ``load()`` returns ``T``. Its generic batch loader
preserves names and reuses equivalent requests within one call. A later call
acquires fresh inputs. This module uses only the standard library and the
numeric named map; it imports neither Modello nor the portfolio package.

Financial implementations live in ``examples/portfolio/providers.py``.
``EquityProvider`` returns ``PriceHistory`` and exposes its quote identity before
loading. CSV transaction and index sources implement the same generic contract
with ``TransactionLog`` and ``Series[datetime]`` results. The
:doc:`provider walkthrough <providers>` shows configuration and acquisition.

Compile expressions separately
-------------------------------

``examples/compute/scalar_plan.py`` adds SymPy expression compilation.
``ScalarPlan`` accepts named independent symbols and named expressions, then
evaluates either one numeric point or aligned series. It works with ordinary
SymPy expressions as well as expressions resolved from Modello models.

The generic computation package imports neither Modello nor the financial
examples. This keeps the data and execution code usable as a candidate for a
later standalone package, while its interfaces are explored here.

Connect domain models through an adapter
----------------------------------------

``examples/portfolio/models.py`` holds shared equity, position, and cash models.
Source adapters, account replay, signals, performance, and reporting live
alongside them in ``examples/portfolio/``. Tutorial runners assemble the sources
and policies and present their results. This example package imports the generic
computation helpers; importing those helpers does not load the portfolio package.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Layer
     - Modules
     - Role
   * - Numeric data
     - ``compute.named_map``, ``compute.series``
     - Named domains and explicit observation axes; standard library only.
   * - Data acquisition
     - ``compute.providers``
     - Generic typed requests and per-load reuse; standard library only.
   * - Symbolic execution
     - ``compute.scalar_plan``
     - SymPy compilation and numerical execution, independent of domain models.
   * - Model adapter
     - ``named_members``
     - Project declared Modello components into a named domain.
   * - Financial models and calculations
     - ``portfolio.models``, ``portfolio.basket``, ``portfolio.named``, ``portfolio.account``,
       ``portfolio.signals``, ``portfolio.performance``, ``portfolio.reporting``,
       ``portfolio.utility``
     - Domain relationships, ledger transitions, and analytics.
   * - Sources and presentation
     - ``portfolio.providers``, ``portfolio.sources``, ``portfolio.explicit``, ``portfolio.demo``,
       ``portfolio.report_html``
     - External observations, scenario assembly, and report rendering.

The ``component_members`` adapter is the research layer's single access point
to Modello's private component metadata. A future extracted model integration
would need to preserve that adapter boundary or establish a public reflection
interface. The numeric data types and expression executor do not depend on it.

The core API and the research interfaces have separate scopes: the experiments
exercise possible higher layers without adding collection fields, time-series
execution, accounting, or report generation to the model solver. The
:doc:`named-member guide <named_members>` explains the remaining questions about
named domains, and the :doc:`account guide <accounts>` describes the financial
assumptions used in the scenario.

The :doc:`named-solving design note </design/named_solving>` describes the
candidate general Modello feature separately from this application.
