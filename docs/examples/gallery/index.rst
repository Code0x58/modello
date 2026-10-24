A feature walkthrough
=====================

One app introduces the library through four progressively composed models:

.. code-block:: console

   $ uv run --extra gui python -m examples.gallery

Open http://127.0.0.1:5070. Each step uses the same explorer: inspect
**Equations**, edit an observation, or try a guided question. Follow the next
concept in the navigation, or return to **Features**. Browser proposals start
from each model's original observations.

.. list-table:: Four core concepts
   :header-rows: 1
   :widths: 25 45 30

   * - Concept
     - Introduces
     - Example
   * - Scalar models
     - Named quantities, domains, and forward/inverse solving
     - Right-angle triangle
   * - Component models
     - Typed children and joint equations across component boundaries
     - Box with a triangle component
   * - Vector models
     - Observation-sized vectors, indexed formulas, reductions, and derivatives
     - Line fitting
   * - Recurrent models
     - Vectors of typed states, relative indices, and boundary observations
     - Account balances and fills

Scalar models
-------------

.. literalinclude:: ../../../examples/geometry.py
   :start-at:     a = InstanceDummy
   :language: python

Observe two sides to determine the third. The same equation calculates the
hypotenuse or infers a leg; positive domains exclude invalid side lengths.
Leaving observations unspecified retains a symbolic relationship.

:doc:`Solution families <../../solutions>` make those alternatives explicit in
Python. Additional observations narrow the same model:

.. code-block:: pycon

   >>> from examples.geometry import RightAngleTriangle
   >>> from modello import inspect_model
   >>> family = inspect_model(RightAngleTriangle, "triangle", a=3).solutions()
   >>> len(family.parameters)
   1
   >>> family.observe(c=5).one().b
   4

Use :doc:`inspection and diagnostics <../../diagnostics>` to investigate
conflicting observations or unresolved states. These tools apply throughout
the walkthrough.

Component models
----------------

.. literalinclude:: ../../../examples/nested_geometry.py
   :start-at:     base = component
   :language: python

The box reuses the triangle's equations and domains. Observing the box diagonal
can determine a quantity inside the base; the parent and its children form one
system. The :doc:`composition guide <../../nesting>` develops typed observations,
independent scopes, aliases, and model refinement.

Vector models
-------------

.. literalinclude:: ../../../examples/gallery/models.py
   :start-at: class LinearTrend
   :end-before: class GeometricDiffusion
   :language: python

Observation lengths determine N. Vector residuals define a scalar objective;
observing both gradients as zero selects an unweighted least-squares line.
Distinct times give a unique fit; repeated times may leave a family. Supplying
slope and intercept instead evaluates a chosen line. No noise distribution or
uncertainty estimate is implied. See :doc:`../../vectors` for the vector API.

Recurrent models
----------------

The :doc:`model-vector guide <../../model-vectors>` shows the account declaration
and typed Python inputs. Each execution connects two account boundaries.
Observing terminal cash can infer a missing execution price. Adding or removing
executions changes N while retaining the relative law.

Further topics
--------------

The app's **Demos and extensions** page separates specialist examples from the
core sequence:

* Vector reductions: sums, products, dot products, and partial observations.
* Scalar series: a recurrence independent of its observation window; see
  :doc:`../../series`.
* Continuous calculus: a native SymPy integral used in forward and inverse solves.
* Itô diffusion: local transforms and conditional moments from the opt-in
  :doc:`stochastic extension <../../contrib/stochastic>`.

Inheritance and :ref:`extension annotations <extension-annotations>` are covered
in the API guides rather than adding introductory cards.

The equations reuse existing geometry/component examples and
``examples/gallery/models.py``. ``catalog.py`` defines the four tutorial steps,
additional demos, labels, observations, and guided questions. The app mounts the
ordinary ``modello.gui.create_app`` explorer for every entry.
