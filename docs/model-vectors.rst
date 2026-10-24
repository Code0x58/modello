Vectors of models and relative equations
========================================

Use ``model_vector(Model, size=N)`` when each position contains related
quantities. A vector of executions keeps quantity, price, and fee together;
a vector of account states keeps inventory and cash together. Their relative
field equations use the ordinary joint solver.

An account across executions
----------------------------

.. literalinclude:: ../examples/gallery/models.py
   :start-at: class AccountSequence
   :end-before: class LinearExposure
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
   >>> isinstance(case.states[1], HoldingState)
   True
   >>> case.terminal is case.states[-1]
   True

There are ``N`` executions and ``N+1`` account boundaries. ``terminal`` is a
view of the final member, with the same object and symbolic identities. With
no executions, it is the initial state.

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

Copied quantities retain supplied or inferred provenance. Observations from
one constructor argument form a group: ``problem.without("states")`` releases
that group's observations while retaining the equations. Solution families
accept the same typed inputs.

History and formulas
--------------------

``model_recurrence(states[n], ...)`` updates named scalar fields from an
initial history. The default ``start=1`` leaves member zero as history; a
second-order law uses ``start=2`` and can reference ``states[n-1]`` and
``states[n-2]``. Those history members are ordinary unknown models, so their
quantities may be observed or inferred.

Use a distinct ``Index`` inside nested sums or integrals. The recurrence checks
fixed positive self-lags and input bounds before indexing, avoiding accidental
Python wraparound. Nested field updates use canonical paths, for example
``**{"holding.quantity": expression}``.

``project(lambda member: member.field)`` provides a scalar :doc:`vector <vectors>`
when a calculation needs one. It can select nested fields or fixed inner
positions. Use a projection directly inside an aggregate, as ``total_fees``
does, instead of declaring extra vectors solely for storage.

Model-valued laws materialize finite windows and require concrete member schemas.
An inherited collection keeps its shape expression; refine its size parameter's
domain or prepare the original class with different observations. Source
``ModelRecurrence`` templates are read-only inspection values. Redeclare an
active law through ``model_recurrence`` when changing its equations.

Exploring the model
-------------------

The :doc:`gallery <examples/gallery/index>` nests each execution and account
under its vector. Add or remove executions, inspect members, and edit their
quantities. The source recurrence appears once; calculated fields link back
to it. Symbols retain full API paths for editing and navigation.

``ModelInfo.fields`` patterns such as ``fills[]`` and ``fills[].price`` label
every member. A field symbol ``p`` becomes ``p_i``; a model symbol ``F`` becomes
``F_i``. Fields inside nested collections keep qualified public paths unless
given distinct symbols, for example ``rows[].holdings[0].quantity``.

The Python GUI API accepts ``ModelVectorInput`` records such as
``{"fills": [{"quantity": 2, "price": 100, "fee": 1}]}`` alongside scalar
observations like ``"states[0].cash": 1000``.

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
