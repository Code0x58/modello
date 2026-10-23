Vectors of models and account sequences
=======================================

Use ``model_vector(Model, size=N)`` when each position contains a typed component
with several related quantities. A vector of executions holds quantity, price,
fees, and calculated notional together. ``model_recurrence`` connects components
through a reusable transition model, using the usual flat solver for forward
calculations and inverse problems.

A sequence of account states
----------------------------

The portfolio example defines ``HoldingState``, ``Execution``, and ``FillPosting``.
One posting connects ``before`` and ``after`` account states through a ``fill``.
Reuse that transition without writing a new solver:

.. code-block:: pycon

   >>> from examples.portfolio.models import Execution, FillPosting, HoldingState
   >>> from modello import InstanceDummy, Modello, component, model_recurrence, model_vector
   >>> class AccountPath(Modello):
   ...     N = InstanceDummy("N", integer=True, nonnegative=True)
   ...     opening = component(HoldingState)
   ...     fills = model_vector(Execution, size=N)
   ...     states = model_recurrence(FillPosting, initial=opening, fill=fills)
   ...     terminal = states[-1]
   ...     prices = fills.project(lambda fill: fill.price)
   ...     quantities = states.project(lambda account: account.quantity)
   ...     balances = states.project(lambda account: account.cash)
   >>> opening = HoldingState("opening", quantity=3, cash=1000)
   >>> fills = [Execution("buy", quantity=2, price=100, fee=1),
   ...          Execution("sell", quantity=-1, price=110, fee=1)]
   >>> case = AccountPath("settlement", opening=opening, fills=fills)
   >>> case.N, case.quantities, case.balances
   (2, [3, 5, 4], [1000, 799, 908])
   >>> case.terminal.quantity, case.terminal.cash
   (4, 908)

The execution vector's length determines ``N``. There are ``N`` transitions and
``N + 1`` state boundaries:

.. math::

   A_0=A_{\mathrm{opening}},\qquad
   A_n=\operatorname{FillPosting.after}(A_{n-1},F_{n-1}),
   \quad n\in\{1,\ldots,N\}.

This notation summarizes the wiring. Every scalar equation and domain declared
by ``FillPosting`` still participates in the combined problem. The transition
vector lives at ``states.steps``. Its first ``before`` is the opening account;
each later ``before`` is the preceding transition's ``after``. State positions
are views of those same components:

.. code-block:: pycon

   >>> case.states[0] is case.opening
   True
   >>> case.states.steps[1].before is case.states[1]
   True
   >>> case.states.steps[0].fill is case.fills[0]
   True
   >>> case.terminal is case.states[-1]
   True
   >>> empty = AccountPath("empty", opening=opening, fills=[])
   >>> empty.N, len(empty.states), empty.terminal.cash
   (0, 1, 1000)
   >>> empty.terminal is empty.opening
   True

Views share populated objects and symbolic field identities. They introduce no
duplicate state variables or repeated component equations. Changing an attribute
after construction does not recompute a path; construct a new case or propose
observations through a solution family.

Inverse problems and partial observations
-----------------------------------------

A terminal observation constrains the same system as its executions. Leave the
sale price unknown and infer it from terminal cash:

.. code-block:: pycon

   >>> partial = [fills[0], Execution("sale", quantity=-1, fee=1)]
   >>> inverse = AccountPath("inverse", opening=opening, fills=partial,
   ...                       terminal=HoldingState("target", cash=908))
   >>> inverse.fills[1].price
   110
   >>> from modello import inspect_model
   >>> problem = inspect_model(AccountPath, "research", opening=opening, fills=partial)
   >>> family = problem.solutions().observe(terminal=HoldingState("target", cash=908))
   >>> family.one().fills[1].price
   110

Observations are sequences of compatible model instances. A ``None`` entry
leaves that entire member unknown; a partially constructed model leaves its
unspecified fields open. Supplied and previously inferred fields retain their
provenance when copied. Member observations supplied through ``fills=...``
belong to that argument's observation group: ``problem.without("fills")``
releases them together.

Supplying ``N`` explicitly also allocates an unknown collection. Conflicting
lengths or size observations raise ``ModelInputError`` before solving. An empty
execution collection is valid. A state collection includes its opening boundary;
its observed length determines ``N + 1``.

Scalar formulas and selection
-----------------------------

``project`` takes a typed callback and returns an ordinary scalar ``Vector``.
The callback can access nested scalar fields, fixed scalar vectors, and
arithmetic expressions. Direct field projections alias existing coordinates;
computed projections retain their formula. Use ``sum``, ``product``, ``dot``,
or native SymPy operators:

.. code-block:: pycon

   >>> from modello import Index
   >>> from sympy import Sum
   >>> class TotalTradeValue(AccountPath):
   ...     n = Index("n")
   ...     traded = Sum(AccountPath.fills[n].notional, (n, 0, AccountPath.N - 1))
   ...     closing_cash = AccountPath.states[-1].cash
   >>> result = TotalTradeValue("totals", opening=opening, fills=fills)
   >>> result.traded, result.closing_cash
   (90, 908)

Finite collections support iteration, slices, and Python negative indices.
``states[-1]`` means the terminal member of that finite window; the scalar
:doc:`Series <series>` expression ``T[-1]`` means mathematical position minus
one. A whole-model selection such as ``chosen = fills[index]`` requires a
concrete integer from supplied fields or their declared expressions before
solving the remaining scalar problem. Scalar indexed formulas can retain a
local ``Index`` under a sum or comprehension.

The prepared model records that structural selection, just as it records a
resolved vector shape. Releasing an observation does not retarget its component
views. Prepare the original model class again to choose another index or horizon.

Transition ports and scope
--------------------------

``model_recurrence`` uses ``before`` and ``after`` by default. Set
``previous="..."`` and ``result="..."`` for another schema. Both ports and the
opening component must declare the same model class. Additional keyword
arguments bind component inputs: a model vector supplies one per transition;
an ordinary component is shared by every transition. Supply ``size=...`` when
no vector input determines the number of steps. Multiple input vectors
contribute simultaneous shape equations and must have compatible lengths.

Wiring preserves field assumptions and symbolic function identities.
Incompatible schemas or assumptions raise declaration errors. Members may
contain nested components, fixed scalar vectors, series, and process templates.
Projections can also select a fixed member of a nested model vector, such as
``rows.project(lambda row: row.holdings[0].quantity)``; that inner index must be
a concrete integer.
Their component schema must already be concrete; independently sized or
structurally selected components inside members require specialization first.
Field-kind changes between quantities, collections, and component views are
rejected during inheritance.

Recurrences can chain from a preceding sequence's ``terminal`` or ``[-1]``
boundary. Source selections using supplied integer indices resolve during the
same structural preparation as collection lengths. Unresolved function-valued
observations retain their declared template identities when copied. A subtype
that replaces those templates must be declared as that concrete subtype when
its unresolved values are supplied to a component or collection.

This API materializes a finite window of a model-valued recurrence. Nonlinear
transition equations work to the extent supported by the ordinary solver. Use
:doc:`series` for horizon-independent scalar recurrence laws, continuous Itô
coefficients, and supported exact process queries.

Exploring the sequence
----------------------

Run ``python -m examples.model_sequences`` for the console walkthrough. Start
the sequence and diffusion demos on browser-compatible loopback ports:

.. code-block:: console

   $ uv run --extra gui python -m examples.gui_timeseries --example sequence --port 5071
   $ uv run --extra gui python -m examples.gui_timeseries --example diffusion --port 5070

The sequence explorer shows model vectors with arrow accents, bracketed member
paths, editable scalar projections, and the relative transition rule in MathML.
Transition details are initially folded. State boundaries remain separately
inspectable, with the usual editing, domain validation, and observation history.

API
---

.. currentmodule:: modello

.. autofunction:: model_vector

.. autoclass:: ModelVector
   :members: shape, project

.. autofunction:: model_recurrence

.. autoclass:: ModelSequence

``ModelVectorValue[Model]`` describes observations as a typed model vector or
a sequence of compatible model instances and ``None``.
