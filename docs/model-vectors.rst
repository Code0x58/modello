Vectors of models and relative equations
========================================

Use ``model_vector(Model, size=N)`` when each position contains a typed model
with related quantities. An execution holds its quantity, price, fee, and
calculated trade value together. Account states hold inventory and cash.
``model_recurrence(states[n], ...)`` relates those members at relative indices.
Their equations use the ordinary joint solver for forward and inverse problems.

An account across executions
----------------------------

The example declares two model vectors and their field equations:

.. literalinclude:: ../examples/model_sequences.py
   :start-at: class AccountSequence
   :end-before: def main
   :language: python

Its mathematical statement is:

.. math::

   Q_n=Q_{n-1}+u_{n-1},\qquad
   C_n=C_{n-1}-v_{n-1}-c_{n-1},\qquad n\in\{1,\ldots,N\},

where execution ``i`` has signed quantity ``u_i``, price ``p_i``, trade value
``v_i = u_i p_i``, and fee ``c_i``. Each ``states[n]`` is a ``HoldingState``
model; each ``fills[i]`` is an ``Execution`` model. The member schemas retain
all their equations and domains.

.. code-block:: pycon

   >>> from examples.model_sequences import AccountSequence
   >>> from examples.portfolio.models import Execution, HoldingState
   >>> from modello import ModelObservations
   >>> opening = HoldingState("opening", quantity=3, cash=1000)
   >>> fills = [Execution("buy", quantity=2, price=100, fee=1),
   ...          Execution("sell", quantity=-1, price=110, fee=1)]
   >>> case = AccountSequence("settlement", fills=fills,
   ...                        states=ModelObservations({0: opening}))
   >>> case.N
   2
   >>> tuple((state.quantity, state.cash) for state in case.states)
   ((3, 1000), (5, 799), (4, 908))
   >>> isinstance(case.states[1], HoldingState)
   True
   >>> case.terminal is case.states[-1]
   True

The execution vector's length determines ``N``. The account vector contains
``N + 1`` boundaries, including the initial state. A terminal component is a
view of the final member, sharing its object and symbolic field identities.

``ModelObservations`` supplies sparse typed observations within that window.
Its keys are finite vector positions; negative keys use ordinary Python
indexing. The number of observed positions does not determine the window
length. Supply an input vector, a complete state sequence, or an explicit size
field to determine that length.

.. code-block:: pycon

   >>> empty = AccountSequence("empty", fills=[],
   ...                         states=ModelObservations({0: opening}))
   >>> empty.N, len(empty.states), empty.terminal.cash
   (0, 1, 1000)
   >>> empty.terminal is empty.states[0]
   True

Initial history and inverse solving
-----------------------------------

``start=1`` is the default: member zero supplies the first-order history.
For a second-order rule, set ``start=2`` and reference ``states[n-1]`` and
``states[n-2]``. Members zero and one are ordinary unknown models whose
quantities can be observed or inferred. The vector must contain that history;
the recurrence uses fixed positive self-lags covered by it. Reusing an input
index outside its vector window raises an error instead of wrapping around.

A terminal observation constrains the same system as the executions. Leave the
sale price unknown and infer it from terminal cash:

.. code-block:: pycon

   >>> partial = [fills[0], Execution("sale", quantity=-1, fee=1)]
   >>> inverse = AccountSequence("inverse", fills=partial,
   ...     states=ModelObservations({0: opening}),
   ...     terminal=HoldingState("target", cash=908))
   >>> inverse.fills[1].price
   110
   >>> from modello import inspect_model
   >>> problem = inspect_model(AccountSequence, "research", fills=partial,
   ...                         states=ModelObservations({0: opening}))
   >>> family = problem.solutions().observe(terminal=HoldingState("target", cash=908))
   >>> family.one().fills[1].price
   110

A full sequence may contain ``None`` members. A partially constructed model
leaves its unspecified quantities open. Copied observations retain their supplied
or inferred provenance. Observations supplied through one argument belong to
that argument's group: ``problem.without("states")`` releases its copied state
observations while retaining the model equations.

Using scalar formulas
---------------------

``project`` provides an ordinary scalar vector when a calculation needs one.
It accepts a typed callback with nested fields, fixed scalar vectors, and
arithmetic. A projection can be used inside an aggregate without adding another
field to the model. The example's ``total_fees`` uses that approach.

.. code-block:: pycon

   >>> case.fills.project(lambda fill: fill.notional).sum().doit()
   90
   >>> case.states.project(lambda state: state.cash)
   [1000, 799, 908]

Local ``Index`` objects support native SymPy sums and products over member
fields. Use distinct local indices inside nested sums or integrals. Named
updates can also address nested fields with their canonical paths, such as
``**{"holding.quantity": expression}`` or ``**{"samples[0]": expression}``.

The ``ModelRecurrence`` returned by the indexed declaration is an immutable
inspection template. Its original equations and iteration domain remain
available through ``problem.templates`` and the GUI. Active declarations must
come from ``model_recurrence(states[n], ...)``; redeclare a law through that
helper when changing its equations. Assigning a transformed inspection template
to a model class is rejected, keeping displayed and compiled equations aligned.

Finite collections support iteration, slices, and Python negative indices.
``states[-1]`` selects the terminal member of the finite window. The scalar
:doc:`Series <series>` expression ``T[-1]`` instead denotes mathematical position
minus one. Whole-model selections using an integer model field resolve during
structural preparation. Prepare the original model class again to select a
different index or horizon.

Schemas and compatibility
-------------------------

Members may contain nested components, fixed vectors, series, and process
templates. Their schema must be concrete before building the outer collection.
Projections can select a fixed member of a nested collection, for example
``rows.project(lambda row: row.holdings[0].quantity)``. Unresolved values copied
from a subtype that replaces symbolic templates require that concrete subtype's
declared schema.

The transition-model overload of ``model_recurrence`` remains available for
existing component-based models. It returns a ``ModelSequence`` and preserves
shared transition ports. Direct indexed field equations provide the simpler
account-sequence declaration shown above and target a declared ``model_vector``.
An inherited collection keeps its declared shape expression; refine the domain
of its size parameter or provide a different observation window when preparing
the model. Declare a new model shape to change that expression.

A model-valued recurrence materializes a finite window. Nonlinear equations work
to the extent supported by the ordinary solver. See :doc:`series` for
horizon-independent scalar laws, continuous Itô coefficients, and supported
exact process queries.

Exploring model members
-----------------------

Run ``python -m examples.model_sequences`` for the console walkthrough:

.. code-block:: console

   $ uv run --extra gui python -m examples.gui_timeseries --example sequence --port 5071
   $ uv run --extra gui python -m examples.gui_timeseries --example diffusion --port 5070

The account explorer nests each execution and state under its vector. Add or
remove executions in the vector header, inspect a model, and edit its scalar
quantities. Empty cells remain unknown. The original relative law is shown
once in the equations view; individual calculated fields link back to it.
Mathematical symbols retain the complete API paths for navigation and editing.

``ModelInfo.fields`` can label every member with patterns such as ``fills[]``
and ``fills[].price``. A field symbol ``p`` becomes the indexed symbol ``p_i``;
a member symbol ``F`` becomes ``F_i``. Scalar projections appear when explicitly
declared for another purpose. Fields inside nested collections keep qualified
public paths by default, so fixed inner positions remain distinct. Assign
separate symbols with patterns such as ``rows[].holdings[0].quantity`` when
presenting those fields in an outer recurrence.

For the Python GUI API, ``ModelVectorInput`` describes an ordered sequence of
field-observation records. For example, ``inputs={"fills": [{"quantity": 2,
"price": 100, "fee": 1}], "states[0].quantity": 3, "states[0].cash": 1000}``
sets up the one-execution case. Browser cells use the bounded scalar arithmetic
language, with the existing observation history and validation behavior.

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

.. autoclass:: ModelSequence

``ModelVectorValue[Model]`` describes a typed model vector, a sequence of
compatible model instances and ``None``, or sparse ``ModelObservations``.
