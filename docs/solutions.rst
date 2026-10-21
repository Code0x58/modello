Explore a solution family
==========================

An observation can leave several alternatives, or a continuous family of
answers. ``problem.solutions()`` keeps those answers together so that you can
restrict them, examine possible outputs, and select ordinary model instances.
This API is a WIP exploration; its current scope and guarantees are below.

Start with the :doc:`debugging guide's <debugging>` small composed model:

>>> from modello import InstanceDummy, Modello, component, inspect_model
>>> from modello.diagnostics import FieldValue
>>> class Reading(Modello):
...     x = InstanceDummy("x", real=True)
...     squared = x**2
>>> class Check(Modello):
...     reading = component(Reading)
...     total = reading.squared + 1
>>> problem = inspect_model(Check, "explore")
>>> x = problem.field("reading", "x")
>>> total = problem.field("total")
>>> family = problem.solutions()
>>> family.parameters == (x,)
True
>>> family.parameter_domain
Reals
>>> family.expression(total) == x.symbol**2 + 1
True
>>> family.values(total)
Interval(1, oo)

The family uses the model's expressions and domains. Here one real coordinate
describes every possible state. ``expression(field)`` gives an expression in
the remaining coordinates; ``values(field)`` maps the joint family to a SymPy
set of possible values. Coordinate count is not generally a dimension: a
coordinate domain can still contain equations, discrete alternatives, or
unresolved conditions.

Refine observations and select a state
---------------------------------------

Add a named observation using the actual field record:

>>> observed = family.observe(FieldValue(total, 5))
>>> observed.values(x)
{-2, 2}
>>> observed.project(x, total)
{(-2, 5), (2, 5)}

``project`` preserves coupling and returns tuples in the selected fields' order.
Taking independent Cartesian products of each field's values would lose that
relationship. ``cases()`` materialises an explicit finite family as checked,
independently bound models:

>>> sorted((case.reading.x, case.total) for case in observed.cases())
[(-2, 5), (2, 5)]
>>> chosen = observed.where(x.symbol > 0).one()
>>> isinstance(chosen, Check), isinstance(chosen.reading, Reading)
(True, True)
>>> chosen.reading.x, chosen.total
(2, 5)

Each case is checked against the original equations, all declared assumptions,
and the added conditions before being returned. Components and methods retain
their model types. Materialisation uses the same population mechanism as solved
components; custom constructor hooks are not invoked. The source problem and
earlier families stay unchanged.

Bounds can be more useful than an exact observation:

>>> bounded = family.where(total.symbol >= 4, total.symbol <= 6)
>>> from sympy import Interval, sqrt
>>> bounded.values(x) == Interval(-sqrt(5), -sqrt(3)) | Interval(sqrt(3), sqrt(5))
True

The :doc:`impact calibration example <examples/solution_families/index>` uses a
bounded measurement to constrain a parameter and predict another trade's cost.
Such sets describe possibilities, without assigning probabilities to them.

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
``family.conditions`` and ``family.observations`` retain added restrictions.

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
certificate. Pass actual fields to ``problem.solutions(field, ...)`` to prefer
particular coordinates; remaining constraints can still restrict them.

The exploratory API requires exact expressions: use ``Rational`` for decimal
observations. Evaluate bound-variable expressions such as integrals and
derivatives before exploring. Other model-construction behaviour is unchanged.
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
   ``parameters`` and ``parameter_domain`` describe the chart. ``conditions``,
   ``observations``, and ``issues`` retain added restrictions and unsuccessful
   reductions. ``is_empty`` is ``True``, ``False``, or ``None``.

.. automethod:: SolutionFamily.format

.. automethod:: SolutionFamily.expression

.. automethod:: SolutionFamily.image

.. automethod:: SolutionFamily.values

.. automethod:: SolutionFamily.project

.. automethod:: SolutionFamily.where

.. automethod:: SolutionFamily.observe

.. automethod:: SolutionFamily.cases

.. automethod:: SolutionFamily.one

.. autoclass:: SolutionIssue

.. currentmodule:: modello.errors

.. autoclass:: SolutionSelectionError

.. autoclass:: UnresolvedSolutionError
