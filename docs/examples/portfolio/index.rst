Build a portfolio research system
=================================

Start with the value of one holding, then add the tools needed to follow an
account through time. The walkthrough builds model definitions, named
projections, numerical evaluation, data acquisition, ledger replay, decisions,
and reporting in that order. Two final studies derive execution schedules,
calibrations, and liquidity reports from a continuous mathematical model.
Each chapter explains a new requirement before introducing its helper.

Install the runnable examples with Python 3.10 or newer:

.. code-block:: console

   $ pip install "modello[examples]"

The first chapter uses only the core :doc:`model API </api>` and
:doc:`typed components </nesting>`. Later chapters develop the experimental
helpers shipped under ``examples/``. The data is synthetic, denominated in USD,
and intended to demonstrate calculations and accounting relationships.

Follow the chapters in order when building your own version. Each page's
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

By the end, the same component definitions support scalar inverse questions,
account and instrument tables, and timeline charts. The reporting chapter assembles
the complete scenario and links to the `interactive report <report.html>`_.
The final studies use SymPy's calculus, integration, algebraic solving, and
implicit differentiation to reuse a cost functional across applications.
The :doc:`implementation map <layers>` is an optional reference for finding the
completed code, numerical timings, and the boundaries of the experiment.
