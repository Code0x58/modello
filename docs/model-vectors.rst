Vectors of models and relative equations
========================================

Use ``model_vector(Model, size=N)`` when each position contains related
quantities. A vector of executions keeps quantity, price, and fee together;
a vector of account states keeps inventory and cash together. Their relative
field equations describe how the states relate. The fourth step of
:doc:`getting-started` gives a complete declaration using only core imports.

An account across executions
----------------------------

.. literalinclude:: ../examples/gallery/models.py
   :start-at: class AccountSequence
   :end-before: class LinearTrend
   :language: python

.. math::

   Q_n = Q_{n-1} + u_{n-1},\qquad
   C_n = C_{n-1} - v_{n-1} - c_{n-1},\qquad n\in\{1,\ldots,N\}.

Member schemas retain their own equations, such as execution value
``v_i = u_i*p_i``.

Supply typed observations and let execution count determine the horizon:

.. code-block:: pycon

   >>> from examples.gallery.models import AccountSequence
   >>> from examples.portfolio.models import Execution, HoldingState
   >>> from modello import ModelObservations
   >>> opening = HoldingState("opening", quantity=3, cash=1000)
   >>> fills = [Execution("buy", quantity=2, price=100, fee=1),
   ...          Execution("sell", quantity=-1, price=110, fee=1)]
   >>> case = AccountSequence("account", fills=fills,
   ...                        states=ModelObservations({0: opening}))
   >>> case.N, tuple((state.quantity, state.cash) for state in case.states)
   (2, ((3, 1000), (5, 799), (4, 908)))

There are ``N`` executions and ``N+1`` account boundaries. ``terminal`` refers
to the final member. With no executions, it is the initial state.

Partial observations and inverse solving
----------------------------------------

``ModelObservations`` supplies sparse typed observations without imposing a
window length. Its keys are vector positions; negative keys follow Python
indexing. A complete sequence may instead contain model instances and ``None``.
An input vector, complete state sequence, or explicit size field determines N.

The same model infers an execution price from terminal cash:

.. code-block:: pycon

   >>> partial = [fills[0], Execution("sale", quantity=-1, fee=1)]
   >>> inverse = AccountSequence("inverse", fills=partial,
   ...     states=ModelObservations({0: opening}),
   ...     terminal=HoldingState("target", cash=908))
   >>> inverse.fills[1].price
   110

Leave out the terminal observation to retain a cash formula in the unknown
price. Add or remove fills to ask about a different window. Solution families
accept the same typed observations; see :doc:`solutions` for adding conditions.

History and formulas
--------------------

``model_recurrence(states[n], ...)`` declares equations for named scalar fields
over the prepared window. The default ``start=1`` leaves member zero available
for boundary data; a second-order initial-value law can use ``start=2``.
Those prefix members are ordinary unknown models, so their quantities may be
observed or inferred.

Current-step, backward, and forward references are joint equations within that
window. They can connect fields, express implicit updates, or form centered
difference equations with Piecewise boundaries. Grouping these equations into
one law or splitting them across component laws preserves their meaning.
An explicit ``states[-1]`` reference denotes the terminal finite member.
A scalar :class:`~modello.Recurrence` instead evaluates causally from history.

Relative positions are checked before indexing, avoiding accidental Python
wraparound. Inactive Piecewise branches may contain references outside the
window. Nested sums and calculus operators keep their own lexical coordinates;
distinct ``Index`` objects make their scopes easier to read. Nested field
updates use canonical paths, for example ``**{"holding.quantity": expression}``.

``project(lambda member: member.field)`` provides a scalar :doc:`vector <vectors>`
when a calculation needs one. It can select nested fields or fixed inner
positions. Use a projection directly inside an aggregate, as ``total_fees``
does, instead of declaring extra vectors solely for storage.

Each member needs a declared model type and each case needs a finite window.
Use another construction for different observations, and declare another law
when the relationships themselves change. See :doc:`generality` for solver scope.

Exploring the model
-------------------

The :doc:`gallery <examples/gallery/index>` nests each execution and account
under its vector. Add or remove executions, inspect members, and edit their
quantities. The source recurrence appears once; calculated fields link back
to it. Symbols retain full API paths for editing and navigation.

The :doc:`GUI guide <gui>` describes labels and browser inputs for your own model.

API
---

.. currentmodule:: modello

.. autofunction:: model_vector

.. autoclass:: ModelVector
   :members: shape, project

.. autofunction:: model_recurrence

.. autoclass:: ModelRecurrence
   :members: equations, index, start, stop

.. autoclass:: ModelObservations

``ModelVectorValue[Model]`` accepts a typed vector, a sequence of compatible
models and ``None``, or sparse ``ModelObservations``.
