Build a portfolio research system
=================================

This walkthrough is an exploration of ideas for symbolic computing, using a
portfolio research system as a worked example. It is a working proof of concept
with rough edges and open design questions. Its interfaces, abstractions, and
organisation are still evolving. Read it as a collection of ideas to examine,
adapt, and improve.

The central question is how much of a system we can build from shared
mathematical definitions. A position has a value, a trade changes a state, and
an account relates investment profit to changes in wealth. Connecting those
definitions gives us opportunities to reuse the same mathematics for valuation,
inference, calibration, and reporting.

Ideas to look for
-----------------

* **Build systems by connecting typed components.** Each component owns its
  equations. A parent connects their quantities, and operations can themselves
  become inputs to larger operations. The resulting structure remains available
  for traversal, debugging, and display.
* **Ask different questions of the same model.** Changing knowns, unknowns, and
  fitted parameters lets one definition calculate a result or infer an input.
  Compiled views reuse those expressions over numerical observations.
* **Let the algebra generate other applications.** The final execution studies
  use SymPy to derive differential equations, integrate a continuous cost model
  onto a discrete grid, fit parameters, and calculate sensitivities. This is
  where a shared symbolic definition can save substantial duplicated mathematics.

The portfolio gives these ideas something concrete to work on: external prices,
a transaction log, positions and cash, signals and orders, performance measures,
and account and instrument reports. Some of the supporting code may prove useful
more generally; some may be simplified or replaced as the experiment develops.

Scope of the experiment
-----------------------

The core :doc:`model API </api>` and :doc:`typed components </nesting>` supply
symbolic composition and solving. The acquisition, numerical, time-series,
accounting, and reporting helpers developed here live separately under
``examples/``. Their design can evolve independently, and some may eventually
belong in separate modules or libraries.

The worked account uses synthetic USD data, funded long positions, and a small
illustrative allocation policy. Its reports explain accounting and analytical
relationships; the short sample provides no evidence of a strategy's investment
performance. Financial conventions and data-handling choices are made explicit
as they become relevant.

Work through the example
------------------------

Start with the value of one holding, then add the tools needed to follow an
account through time. The chapters build model definitions, named projections,
numerical evaluation, data acquisition, ledger replay, decisions, and reporting
in that order. Each chapter explains a new requirement before introducing its
helper.

Install the runnable examples with Python 3.10 or newer:

.. code-block:: console

   $ pip install "modello[examples]"

Begin with :doc:`valuing a holding <holdings>`, which uses only core Modello,
and follow the chapters in order when building your own version. Each page's
executable examples also include the imports and setup needed to run it on its
own. Code included directly from a module is the implementation of the helper
being developed at that step.

.. toctree::
   :maxdepth: 1

   holdings
   data
   accounts
   signals
   reporting
   optimal_execution
   execution_calibration

The reporting chapter assembles the complete scenario and links to the
`interactive report <report.html>`_. The two execution studies reuse a
continuous model for scheduling, calibration, and liquidity questions.

Where the layers live
----------------------

.. mermaid::
   :caption: Definitions, observations, evaluation, and reports
   :alt: Models and named declarations supply compiled calculations. Providers supply observations and an external ledger. Replay connects both to decisions, analytics, and reports.

   flowchart TD
       models[Typed mathematical components] --> views[Compiled model views]
       data[Providers and observations] --> replay[Chronological account replay]
       views --> replay
       replay --> decisions[Signals and execution assessment]
       replay --> analytics[Performance and risk]
       replay --> reports[Attribution and utility reports]
       analytics --> reports
       decisions --> reports

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Responsibility
     - Modules under ``examples/``
   * - Named data, causal series, provider requests
     - ``data``; standard library only.
   * - Declaration and numerical model adapters
     - ``named_members``, ``symbolic``.
   * - Financial equations and portfolio composition
     - ``portfolio.models``, ``portfolio.valuation``.
   * - External observations and chronological replay
     - ``portfolio.sources``, ``portfolio.account``.
   * - Decisions and execution assessment
     - ``portfolio.signals``, ``portfolio.execution``.
   * - Statistics, attribution, and presentation
     - ``portfolio.analytics``, ``portfolio.reporting``, ``portfolio.report_html``.
   * - Complete research case and continuous-model studies
     - ``portfolio.demo``, ``portfolio.optimal_execution``.

The account and execution studies are the runnable examples:

.. code-block:: console

   $ python -m examples.portfolio --html account.html --json account.json
   $ python -m examples.portfolio.optimal_execution

The wheel includes the CSV fixtures. Construction and compilation happen before
repeated numerical evaluation; ``ModelPath`` retains columns and creates typed
point models only when requested. Provider acquisition, event chronology,
publication horizons, and reporting conventions stay explicit.

.. _named-member-solving:

Named-member solving as a general feature
------------------------------------------

The reusable Modello question explored here is how to project expressions from
a declared domain of named, typed components and reduce them while retaining
joint forward and inverse solving. ``component_members`` expands a declaration
into ordinary scalar expressions; ``MemberValues`` associates application
results with the actual members of their source case.

This prototype adapts existing component metadata. It introduces no public
map-valued solver input or time-series solving rule. A future core interface
would need to define declaration, reflection, constraint, and error semantics,
including how static types distinguish common member types from particular
keys. Acquisition, chronology, execution backends, and financial conventions
can remain consumers of the resulting expressions.
