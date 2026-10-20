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
   named_members
   modelling
   providers
   accounts
   signals
   performance
   reporting
   optimal_execution
   execution_calibration
   layers

The reporting chapter assembles the complete scenario and links to the
`interactive report <report.html>`_.
The :doc:`implementation map <layers>` is an optional reference for finding the
completed code, numerical timings, and further design questions.
