modello documentation
=====================

Express a problem as named quantities and relationships, then supply the
observations you have. The same model can calculate results, infer missing
inputs, or keep a formula while an input is unknown.

Install with Python 3.10 or newer:

.. code-block:: console

   $ pip install modello

A small model already works in both directions:

.. literalinclude:: ../README.md
   :start-after: ```python
   :end-before: ```
   :language: python

Read :doc:`getting-started` for a complete, runnable progression:

.. list-table:: Express progressively richer problems
   :header-rows: 1
   :widths: 25 75

   * - Concept
     - Question to express
   * - Scalar models
     - Calculate a brace length or infer a missing measurement.
   * - Component models
     - Use a parcel's clearance to determine a quantity in its base.
   * - Vector models
     - Evaluate a proposed calibration or fit its parameters to readings.
   * - Recurrent models
     - Post account events or infer an event from a closing balance.

You can try the same progression in the :doc:`interactive gallery
<examples/gallery/index>`. To choose by notation or finite versus unbounded
scope, start with :doc:`modelling`. The guides below develop each approach;
the API and diagnostic guarantees are separate reference material.

.. toctree::
   :caption: Start modelling
   :maxdepth: 1

   getting-started
   examples/gallery/index

.. toctree::
   :caption: Modelling guides
   :maxdepth: 1

   modelling
   nesting
   vectors
   series
   model-vectors
   solutions
   interoperability
   debugging
   gui

.. toctree::
   :caption: Reference
   :maxdepth: 1

   api
   generality
   diagnostics

.. toctree::
   :caption: Extensions
   :maxdepth: 1

   contrib/stochastic
   contrib/validation

.. toctree::
   :caption: Recipes
   :maxdepth: 1

   recipes/dataframe-validation

.. toctree::
   :caption: Applications
   :maxdepth: 1

   examples/applications/index
   examples/market-making/index
   examples/portfolio/index
   examples/reconciliation/index
   examples/solution_families/index

.. toctree::
   :caption: Development
   :maxdepth: 1

   contributing

* :ref:`genindex`
* :ref:`search`
