An example gallery
==================

One local app presents six small models. Each introduces a different kind of
relationship; all use the same observations, inverse solving, and explorer:

.. code-block:: console

   $ uv run --extra gui python -m examples.gallery

Open http://127.0.0.1:5070. Choose an example, inspect **Equations**, and edit a
quantity or try its guided question. **All examples** returns to the gallery;
**Change example** switches models. Each browser proposal starts from that
example's original observations.

.. list-table:: Choose the relationship you need
   :header-rows: 1
   :widths: 25 40 35

   * - Example
     - Introduces
     - Try
   * - Vector summaries
     - Sums, products, and dot products
     - Infer an observation from a total
   * - Relative sequences
     - A law independent of its sample window
     - Replace states with innovations
   * - An account across executions
     - Vectors whose members are typed models
     - Infer a price from terminal cash
   * - Continuous exposure
     - A native SymPy integral
     - Infer a closing level from exposure
   * - Fitting a line
     - Vector residuals and objective derivatives
     - Set gradients to zero to select a fit
   * - An Itô diffusion
     - Transformed coefficients and conditional moments
     - Infer process parameters from log coefficients

The equations live in ``examples/gallery/models.py``. Display labels, initial
observations, and guided questions live in ``catalog.py``. The app mounts the
ordinary ``modello.gui.create_app`` explorer for each entry; the gallery adds
navigation rather than another solver.

Vector summaries
----------------

.. literalinclude:: ../../../examples/gallery/models.py
   :start-at: class VectorSummary
   :end-before: class ObservedAutoregression
   :language: python

The observation length determines N, including an empty vector. ``None`` keeps
an entry unknown. See :doc:`../../vectors` for operations and inspection.

Relative sequences
------------------

The :doc:`series guide <../../series>` develops the recurrence and its boundary.
State and innovation vectors observe the same equation, so the guided question
releases one before supplying the other. Innovations have no declared distribution.

Typed account states
--------------------

The :doc:`model-vector guide <../../model-vectors>` shows the account declaration
and typed Python inputs. Executions reuse the portfolio's quantity, price, fee,
and trade-value model. The account vector contains the initial state and one
state per execution. Adding observations determines N; it does not rewrite the law.

Continuous exposure
-------------------

.. literalinclude:: ../../../examples/gallery/models.py
   :start-at: class LinearExposure
   :end-before: class LinearTrend
   :language: python

This model explicitly assumes linear interpolation over a positive duration.
The integral is exposure in level-times-duration units.

Fitting a line
--------------

.. literalinclude:: ../../../examples/gallery/models.py
   :start-at: class LinearTrend
   :end-before: class GeometricDiffusion
   :language: python

Observing both gradients as zero selects an unweighted least-squares line.
Distinct times determine a unique fit; repeated times may leave a family.
Supplying slope and intercept instead evaluates any chosen line. A noise model
or uncertainty estimate would require additional relationships.

An Itô diffusion
----------------

.. literalinclude:: ../../../examples/gallery/models.py
   :start-at: class GeometricDiffusion
   :language: python

The logarithm exposes the Itô correction. ``closing`` conditions on a realised
Wiener increment; ``mean`` and ``variance`` are conditional moments. The
:doc:`series guide <../../series>` describes supported process classes and limits.
