Explore a solution family
==========================

An observation can leave several alternatives, or a continuous family of
answers. ``problem.solutions()`` keeps those answers together so that you can
restrict them, examine possible outputs, and select ordinary model instances.
This API is a WIP exploration; its current scope and guarantees are below.

Start with the :doc:`debugging guide's <debugging>` small composed model:

>>> from modello import InstanceDummy, Modello, component, inspect_model
>>> class Reading(Modello):
...     x = InstanceDummy("x", real=True)
...     squared = x**2
>>> class Check(Modello):
...     reading = component(Reading)
...     total = reading.squared + 1
>>> problem = inspect_model(Check, "explore")
>>> check = problem.symbols
>>> family = problem.solutions()
>>> family.parameters[0].symbol == check.reading.x
True
>>> family.parameter_domain
Reals
>>> family.expression(check.total) == check.reading.x**2 + 1
True
>>> family.values(check.total)
Interval(1, oo)

``problem.symbols`` exposes the declared component tree with its model type,
so attribute completion follows ``Check`` and ``Reading``. Like component
references in a class definition, this read-only namespace provides symbols,
rather than solved instances or their methods. Each quantity has its own symbol,
including derived fields such as ``check.total``: observing that symbol can
constrain the inputs that produce it. ``family.symbols`` exposes the same view.

The family uses the model's expressions and domains. Here one real coordinate
describes every possible state. ``expression(...)`` rewrites a scalar expression
in the remaining coordinates; ``values(...)`` maps it to a SymPy set of possible
values. Both accept model symbols or expressions built from them. For example:

>>> family.expression(check.total - check.reading.squared)
1

Coordinate count is not generally a dimension: a
coordinate domain can still contain equations, discrete alternatives, or
unresolved conditions.

Refine observations and select a state
---------------------------------------

Add observations with the same scalar and component arguments as a constructor:

>>> observed = family.observe(total=5)
>>> observed.values(check.reading.x)
{-2, 2}
>>> observed.values(check.reading.x, check.total)
{(-2, 5), (2, 5)}

With several expressions, ``values`` preserves coupling and returns tuples in
the selected expressions' order.
Taking independent Cartesian products of each field's values would lose that
relationship. ``cases()`` materialises an explicit finite family as checked,
independently bound models:

>>> sorted((case.reading.x, case.total) for case in observed.cases())
[(-2, 5), (2, 5)]
>>> chosen = observed.where(check.reading.x > 0).one()
>>> isinstance(chosen, Check), isinstance(chosen.reading, Reading)
(True, True)
>>> chosen.reading.x, chosen.total
(2, 5)

Each case is checked against the original equations, all declared assumptions,
and the added conditions before being returned. Components and methods retain
their model types. Materialisation uses the same population mechanism as solved
components; custom constructor hooks are not invoked. The source problem and
earlier families stay unchanged.
Calling ``observe()`` or ``where()`` without arguments returns the same family
without repeating symbolic reduction.

Observations accumulate; supplying a different value adds a constraint rather
than overwriting the earlier one. Typed component observations use the same
copying and input/inferred provenance rules as ordinary construction:

>>> fixed = family.observe(reading=Reading("measured", x=2)).one()
>>> fixed.reading.x, fixed.total
(2, 5)
>>> observed.observe(total=6).is_empty
True

Use symbolic relations to constrain a nested quantity without constructing its
component, or to relate quantities across layers:

>>> from sympy import Eq
>>> family.where(Eq(check.reading.squared, 4), check.reading.x > 0).one().total
5

Bounds can be more useful than an exact observation:

>>> bounded = family.where(check.total >= 4, check.total <= 6)
>>> from sympy import Interval, sqrt
>>> bounded.values(check.reading.x) == Interval(-sqrt(5), -sqrt(3)) | Interval(sqrt(3), sqrt(5))
True

The :doc:`impact calibration example <examples/solution_families/index>` uses a
bounded measurement to constrain a parameter and predict another trade's cost.
Such sets describe possibilities, without assigning probabilities to them.

Take a section through a reference
----------------------------------

``section(..., through=case)`` lets selected current coordinates vary and fixes
the others at a checked model instance. Derived outputs follow the model's
equations. With no selected coordinates, the result is the reference point:

>>> point = family.section(through=chosen)
>>> point.values(check.reading.x, check.total)
{(2, 5)}

Pass symbols such as ``check.reading.x`` to let those coordinates vary. Aliases
of the same coordinate are counted once. Selecting every current coordinate
returns the original family after checking the reference. Use ``values(...)``
to project all remaining states onto outputs; a section instead restricts the
states by fixing omitted coordinates.

The reference must satisfy the original equations, declared domains, and added
conditions. A proven violation raises ``ModelValidationError``; an undecided
check raises ``UnresolvedSolutionError`` with the candidate audit. A section can
still be discrete or coupled, so the number of selected coordinates does not
certify its dimension. The example also explores small combinations of controls
and their response plots.

Continue after an exception
----------------------------

Ordinary construction still requires one solver candidate. A failed
construction already carries the problem needed for exploration:

>>> from modello import ModelSolveError, SolutionSelectionError, UnresolvedSolutionError
>>> try:
...     Check("ambiguous", total=5)
... except ModelSolveError as error:
...     alternatives = error.problem.solutions()
>>> sorted(case.reading.x for case in alternatives.cases())
[-2, 2]
>>> try:
...     observed.one()
... except SolutionSelectionError as error:
...     print(error.solution_count, error.family is observed)
2 True

``SolutionSelectionError`` retains the family and an exact count of checked
states when ``one()`` finds zero or several. ``UnresolvedSolutionError`` retains
``family``, ``reason``, and an optional candidate ``check`` when numerical
materialisation cannot be established. Its family may be continuous or still
conditional. Printing a family shows its coordinates, domain, and nested
expressions; it does not run another solver. ``family.issues`` keeps failed
reduction attempts with their original backend exception objects.

Guarantees and current scope
-----------------------------

A family represents the captured equations over finite, commutative scalar
values. Every field's declared assumptions remain explicit domain restrictions,
including assumptions on quantities eliminated during reduction. Inferred
component values remain constraints just as they are in the recorded problem.
``family.conditions`` retains added symbolic relations; ``family.problem``
records constructor-style observations and their origins. Its symbols retain
their identities across refinements. Symbols from another prepared problem
raise ``ModelValidationError``, even when their printed names coincide.

Directed substitutions and exact rational affine elimination simplify the
system without discarding other equations. A one-coordinate system can use
``solveset`` and real inequality solving. External coefficients and unevaluated
functions remain conditional where generic solving could lose a singular case.
Solver failures retain the condition; they do not imply an empty family.
General multivariate nonlinear systems keep their joint ``ConditionSet``.

Returned sets can therefore remain implicit. An ``ImageSet`` or ``ConditionSet``
can be an exact representation without an explicit enumeration, a decided
membership test, or a plotting parameterisation. ``cases()`` requires an
explicit finite coordinate domain and independently verifies every state.
``family.is_empty`` preserves SymPy's ``True``, ``False``, or undecided ``None``.
``parameters`` are coordinates, not a general identifiability or dimension
certificate. Pass symbols to ``problem.solutions(check.reading.x, ...)`` to
prefer particular coordinates; remaining constraints can still restrict them.
The ``parameters`` field records remain available for path and alias inspection.

The exploratory API requires exact expressions: use ``Rational`` for decimal
observations. Evaluate bound-variable expressions such as integrals and
derivatives before exploring. Other model-construction behaviour is unchanged.
Scalar queries and observations follow the same commutative-scalar scope as
model fields. ``where(...)`` accepts symbolic relations and Python booleans;
the ``SymbolicCondition`` protocol keeps this boundary visible to type checkers.
Unsupported domain predicates raise ``ModelInputError`` instead of being omitted.
These limits keep unsupported operations explicit while the WIP develops.

The implementation builds on SymPy's
`sets <https://docs.sympy.org/latest/modules/sets.html>`_,
`set-valued and linear solvers <https://docs.sympy.org/latest/modules/solvers/solveset.html>`_,
and `scalar assumptions <https://docs.sympy.org/latest/guides/assumptions.html>`_.
The :doc:`diagnostic reference <diagnostics>` describes the distinct, conservative
consistency and conflict reports. Its one-field subsystem ``solution_sets`` do
not claim to be the joint family explored here.

Reference
-----------

.. currentmodule:: modello.solutions

.. class:: SolutionFamily

   Created by ``ModelProblem.solutions()``. ``problem`` retains the source;
   ``symbols`` provides typed quantity access. ``parameters`` and
   ``parameter_domain`` describe the chart. ``conditions`` and ``issues``
   retain added relations and unsuccessful
   reductions. ``is_empty`` is ``True``, ``False``, or ``None``.

.. automethod:: SolutionFamily.format

.. automethod:: SolutionFamily.expression

.. automethod:: SolutionFamily.values

.. automethod:: SolutionFamily.where

.. automethod:: SolutionFamily.observe

.. automethod:: SolutionFamily.section

.. automethod:: SolutionFamily.cases

.. automethod:: SolutionFamily.one

.. autoclass:: SolutionIssue

.. currentmodule:: modello.errors

.. autoclass:: SolutionSelectionError

.. autoclass:: UnresolvedSolutionError
