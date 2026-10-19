Using the research layers
==========================

The portfolio examples build on the symbolic modeling API in :mod:`modello`.
The core package defines models, typed components, independent instance symbols,
and joint equation solving. Its runtime dependency is SymPy.

The research code lives under ``examples/``. Run it from a checkout or an
extracted source distribution:

.. code-block:: console

   $ uv sync --locked --dev
   $ uv run --locked python -m examples.portfolio
   $ uv run --locked python -m examples.named_portfolio
   $ uv run --locked python -m examples.account_demo --html /tmp/account.html --json /tmp/account.json

The installed wheel contains the ``modello`` package. Research modules and CSV
fixtures are included in the source distribution for readers to run and adapt.

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

``examples/portfolio_models.py`` holds shared equity, position, and cash models.
Source adapters, account replay, signals, performance, and reporting depend on
these definitions and the computation helpers. Tutorial runners assemble the
sources and policies and present their results.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Layer
     - Modules
     - Role
   * - Numeric data
     - ``compute.named_map``, ``compute.series``
     - Named domains and explicit observation axes; standard library only.
   * - Symbolic execution
     - ``compute.scalar_plan``
     - SymPy compilation and numerical execution, independent of domain models.
   * - Model adapter
     - ``named_members``
     - Project declared Modello components into a named domain.
   * - Financial models and calculations
     - ``portfolio_models``, ``named_portfolio``, ``account``, ``signals``,
       ``performance``, ``reporting``, ``utility``
     - Domain relationships, ledger transitions, and analytics.
   * - Sources and presentation
     - ``account_sources``, ``portfolio``, ``account_demo``, ``report_html``
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
