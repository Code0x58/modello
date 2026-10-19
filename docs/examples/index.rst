Examples
========

The examples demonstrate applications of Modello and explore possible higher
layers; their helpers and domain models have evolving interfaces. Install the
examples extra and run a walkthrough:

.. code-block:: console

   $ pip install "modello[examples]"
   $ python -m examples.portfolio
   $ python -m examples.portfolio.demo --html account.html --json account.json

The wheel includes the ``examples`` package and its synthetic CSV fixtures, so
the walkthroughs work outside a source checkout. The current examples use the
standard library and SymPy, already required by Modello. A checkout or extracted
source distribution is useful for editing the models and adapters.

Start with the :doc:`modeling API </api>` and :doc:`nested-model guide </nesting>`
for the core package's public interfaces.

.. toctree::
   :maxdepth: 2

   portfolio/index
