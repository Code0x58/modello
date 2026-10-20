Implementation map and further reading
======================================

This page is a reference for the completed walkthrough. The chapters begin with
:doc:`scalar holdings <holdings>`; use this map when looking up a definition or
considering which parts to reuse elsewhere.

Where each layer lives
----------------------

.. mermaid::
   :caption: Definitions, data, evaluation, and reporting
   :alt: Core models and named declarations supply expressions to the numerical plan. Providers and the ledger supply observed data. Replay uses both to produce account paths, which support decisions, performance, and reports.

   flowchart TD
       core[Modello scalar models] --> declarations[Named components and bindings]
       declarations --> plan[Compiled model views]
       providers[Typed providers] --> observations[Observed series and external ledger]
       observations --> replay[Chronological account replay]
       plan --> replay
       replay --> decisions[Signals and execution assessment]
       replay --> performance[Performance comparison]
       replay --> reports[Attribution and utility reports]
       performance --> reports
       decisions --> reports

.. list-table::
   :header-rows: 1
   :widths: 23 40 37

   * - Responsibility
     - Modules
     - Built in the walkthrough
   * - Scalar equations and joint solving
     - ``modello``
     - Positions, cash, basket value, and inverse questions.
   * - Named numeric data
     - ``examples.compute.named_map``, ``series``
     - Name-preserving projections, aligned axes, lagged inputs, and reductions.
   * - Generic acquisition
     - ``examples.compute.providers``
     - ``Provider[T]`` and per-batch equivalent-request reuse.
   * - Numerical execution
     - ``examples.compute.scalar_plan``
     - Compile resolved expressions and evaluate points or aligned columns.
   * - Model integration
     - ``examples.named_members``, ``model_plan``, ``symbolic``
     - Reflect components, pair bindings, and evaluate typed model trees and observation records.
   * - Portfolio composition and valuation
     - ``examples.portfolio.models``, ``basket``, ``valuation``
     - Shared definitions, contextual holding queries, and portfolio plans.
   * - Temporal accounting
     - ``examples.portfolio.sources``, ``postings``, ``account``, ``pnl``
     - External records, typed state transitions, unitized accounts, and reconciliation.
   * - Decisions and execution
     - ``examples.portfolio.signals``, ``execution``
     - Trailing features, allocation policy, orders, slippage, and shortfall.
   * - Statistics and reporting
     - ``examples.portfolio.performance``, ``analytics``, ``reporting``, ``utility``
     - Index comparison, risk, signal attribution, and utility-derived preferences.
   * - Variational execution studies
     - ``examples.portfolio.optimal_execution``
     - A continuous cost density, integrated grids, generated stationarity,
       impact/preference calibration, and liquidity sensitivities.
   * - Concrete cases and presentation
     - ``examples.portfolio.explicit``, ``named``, ``demo``, ``report_html``
     - Synthetic fixtures, completed scenario assembly, JSON, and interactive HTML.

The wheel includes typed example modules and CSV fixtures. The ``examples``
extra currently uses only the standard library and core SymPy. These application
and computation interfaces are experimental; they remain separate from the core
Modello package and can be extracted or reorganised independently.

The completed entry points are:

.. code-block:: console

   $ python -m examples.portfolio
   $ python -m examples.portfolio.named
   $ python -m examples.portfolio.demo --html account.html --json account.json
   $ python -m examples.portfolio.optimal_execution

The named runner extends the fixed-share date-vector example with GOOGL. Its
daily vectors are entered explicitly; the account's bundled intraday CSV serves
AAPL and MSFT. A new externally loaded instrument needs a matching provider
implementation or source data as well as its holding declaration.

Inspect the declaration adapter
-------------------------------

``examples/named_members.py`` contains the example layer's access to Modello's
component metadata and class namespace. The namespace unpacks bound declarations
before delegating symbol scoping and equations to the core. It also follows the
core's inheritance precedence so an overridden component carries its new binding:

.. raw:: html

   <details><summary>Inspect inheritance and class construction</summary>

.. literalinclude:: ../../../examples/named_members.py
   :pyobject: _BindingNamespace
   :language: python

.. literalinclude:: ../../../examples/named_members.py
   :pyobject: _BoundMembersMeta
   :language: python

.. literalinclude:: ../../../examples/named_members.py
   :pyobject: BoundMembers
   :language: python

.. literalinclude:: ../../../examples/named_members.py
   :pyobject: member_bindings
   :language: python

.. raw:: html

   </details>

``EquityBasket.define`` and ``NamedPortfolio.define`` retain the declared member
type through runtime composition. They reuse a class definition for the same
composition; each instance still has independent symbols. The small cache
adapter preserves the callable's generic signature:

.. literalinclude:: ../../../examples/named_members.py
   :pyobject: cached_definition
   :language: python

A future public declaration/reflection interface could replace the private
integration. The :doc:`named-solving design note </design/named_solving>`
separates that general modeling question from dates, financial conventions,
providers, statistics, and report formatting.

Use the factory pattern outside finance
---------------------------------------

``ModelFactory`` needs a model class and a binding type, rather than a portfolio.
A source name can be enough configuration for another application:

>>> from modello import Modello, InstanceDummy
>>> from examples.named_members import ModelFactory, BoundMembers, member_bindings
>>> class Reading(Modello):
...     value = InstanceDummy("value", real=True)
>>> reading = ModelFactory(Reading, str)
>>> class Sensors(BoundMembers[str]):
...     temperature = reading.component("weather:temperature")
>>> measured = Sensors("observations", temperature=reading("sample", value=18))
>>> measured.temperature.value, member_bindings(measured)["temperature"]
(18, 'weather:temperature')

Calling the factory passes known scalars and typed component instances to the
ordinary model constructor. ``component(binding)`` instead declares a child and
its metadata. Neither operation acquires external data. This is an example-layer
construction pattern; it adds no new solving rules to Modello.

Separate setup cost from numerical execution
--------------------------------------------

The implementation keeps symbolic construction and compilation outside repeated
observation evaluation. Run the local benchmark to measure those separately:

.. code-block:: console

   $ uv run --locked python -m examples.portfolio.benchmark

One exploratory run on Python 3.14.8 with SymPy 1.14.0 gave rounded medians:

.. list-table::
   :header-rows: 1

   * - Work
     - Time
   * - Construct the symbolic portfolio
     - 128 ms
   * - Compile its numerical calculation
     - 2.7 ms
   * - Evaluate complete paths for 10,000 observations
     - 80 ms
   * - Construct and solve a portfolio for each of 10 observations
     - 1,328 ms

Data creation and risk analytics are outside these timings. The benchmark warms
up first, then measures five batches, or three for the slower baseline. Its
solve-per-row baseline reuses component instances, performs one joint solve per
row, and checks NAV against the compiled calculation. These are local exploratory
timings, not throughput guarantees.

Vector evaluation retains the model schema and columns without creating a
model instance at every observation. A typed numeric case is materialized only
when requested with ``ModelPath.at``. Both input and calculated columns are
checked against the model's numeric sign and integer assumptions.

Boundaries left explicit
------------------------

Named domains and observation axes solve different problems. Market value is
pointwise on both; returns shift time; covariance reduces time and adds a second
member axis. The current solver still receives scalar model fields rather than
list-, set-, map-, or time-series-valued solver inputs.

The numeric maps, series, providers, and expression executor can operate without
importing Modello or finance. Model integration sits above them. Chronology,
publication horizons, missing-data policies, currency conversion, corporate
actions, precision choices, and optimisation require explicit policies in their
respective layers. The walkthrough supplies USD quotes, funded nonnegative
actual positions, and a deliberately simple allocation rule.

``examples/model_plan.py`` keeps field traversal and numeric case materialisation
beside the example's other integration adapters. A compiled result uses the
existing model classes and properties; ``ModelPath`` keeps the observation axis
separate from that scalar schema. Field paths and source expressions remain
available for tables, debugging, and other renderers.

``ModelData`` lets typed observations bind to a compatible case, and
``ModelSeries`` adds their explicit axis. Scalar market snapshots, daily price
vectors, and non-financial measurements can therefore use the same evaluator.
The portfolio's ``pricer()`` simply selects input fields and returns an ordinary
``ModelPlan``. ``MarketSnapshot.sample`` owns causal selection, while
``NamedPortfolioPath.from_model`` supplies the financial analytics projection.

The reporting models expose the chosen decomposition and sensitivities. They
provide useful inputs for further research without making accounting depend on
an optimiser or treating a short synthetic example as investment evidence.
