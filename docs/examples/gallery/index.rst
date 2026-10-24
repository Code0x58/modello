A feature walkthrough
=====================

One app introduces the library through four progressively composed models:

.. code-block:: console

   $ uv run --extra gui python -m examples.gallery

Open http://127.0.0.1:5070. Inspect **Equations**, edit an observation, or select
**Explore a question**. Follow the next concept in the navigation, or return to
**Features**.

.. list-table:: Four fundamentals
   :header-rows: 1
   :widths: 25 40 35

   * - Concept
     - Introduces
     - Practical story
   * - Scalar models
     - Named quantities, domains, and forward/inverse solving
     - Size a diagonal brace
   * - Component models
     - Typed children and joint equations across component boundaries
     - Check a parcel's diagonal clearance
   * - Vector models
     - Observation-sized vectors, reductions, and derivatives
     - Evaluate and fit a sensor calibration line
   * - Recurrent models
     - Typed state vectors, relative indices, and boundaries
     - Post fills and reconcile an account

The same questions, progressively richer models
-----------------------------------------------

Every fundamental starts with enough observations for a numeric result and
provides the same five questions:

1. **Calculate results** follows the supplied inputs to their consequences.
2. **Infer missing inputs** releases inputs and supplies a result or fitting
   condition to determine them.
3. **Try different inputs** changes the measurements or collection length while
   retaining the model's equations.
4. **Leave inputs unknown** keeps a meaningful relationship symbolic.
5. **Check the observations** deliberately supplies a conflicting claim. The
   explorer rejects it and highlights the proposal; choose **Calculate results**
   to return to the starting case.

Each question starts from the example's original observations. Questions do not
accumulate constraints from earlier questions. An inverse question releases the
input it is inferring, so an old observation cannot accidentally fix its answer.

Scalar models
-------------

.. literalinclude:: ../../../examples/geometry.py
   :start-at:     a = InstanceDummy
   :language: python

A 3 cm run and 4 cm rise need a 5 cm brace. Observing an 8 cm rise and a 10 cm
brace instead determines a 6 cm run. Leave the run undecided to retain the
brace-length formula, or supply a conflicting length to inspect the rejection.
Positive domains exclude invalid side lengths.

:doc:`Solution families <../../solutions>` explore those unknowns in Python:

.. code-block:: pycon

   >>> from examples.geometry import RightAngleTriangle
   >>> from modello import inspect_model
   >>> family = inspect_model(RightAngleTriangle, "triangle", a=3).solutions()
   >>> len(family.parameters)
   1
   >>> family.observe(c=5).one().b
   4

:doc:`Inspection and diagnostics <../../diagnostics>` apply throughout the
walkthrough, including conflicts and unresolved states.

Component models
----------------

.. literalinclude:: ../../../examples/nested_geometry.py
   :start-at:     base = component
   :language: python

A parcel with a 3 × 4 cm base and 12 cm height has 13 cm diagonal clearance.
Observing that clearance can determine a width inside its triangle component.
An undecided width propagates into both diagonals; a conflicting base diagonal
is rejected by the child's own equation. The :doc:`composition guide
<../../nesting>` develops typed observations, scopes, aliases, and refinement.

Vector models
-------------

.. literalinclude:: ../../../examples/gallery/models.py
   :start-at: class LinearTrend
   :language: python

The initial line ``y = 1 + t`` has squared error 1 against four sensor readings.
**Infer missing inputs** releases its coefficients and sets both objective
gradients to zero: slope and intercept become 0.9, with error 0.7. Least squares
minimises error rather than requiring noisy measurements to lie on a line.

Six new calibration readings determine N = 6 and fit ``y = 1 + 2t`` without
changing the declaration. Leaving the slope open retains predictions and error
as formulas. Claiming zero error for the original proposed line conflicts with
its residuals. See :doc:`../../vectors` for operations and calculus.

Recurrent models
----------------

The account starts with 3 shares and USD 1,000. Buying 2 shares at USD 100 and
selling 1 at USD 110, with a USD 1 fee per fill, leaves 4 shares and USD 908.

The inverse question omits the sale price and observes closing cash to infer
USD 110. Without that closing observation, cash remains a formula in the
unreported price. Appending a purchase changes the state-vector length; an
inconsistent statement balance demonstrates reconciliation. The
:doc:`model-vector guide <../../model-vectors>` shows the declaration and typed
Python inputs.

Extensions and further reading
------------------------------

**Extensions** contains the opt-in :doc:`Itô example
<../../contrib/stochastic>`. Ordinary vector reductions and scalar recurrences
are covered by the fundamentals and their :doc:`vector <../../vectors>` and
:doc:`series <../../series>` guides. Inheritance and :ref:`extension annotations
<extension-annotations>` remain in the API guides.

The equations reuse existing geometry/component examples and
``examples/gallery/models.py``. ``catalog.py`` defines observations and the
shared question progression; ``extensions.py`` holds the specialist model.
The app mounts the ordinary ``modello.gui.create_app`` explorer for each entry.
