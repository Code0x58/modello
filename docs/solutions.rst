Explore a solution family
==========================

An observation can leave several alternatives, or a continuous family of
answers. ``problem.solutions()`` keeps those answers together so that you can
restrict them, examine possible outputs, and select ordinary model instances.
Use this after :doc:`getting-started` when you need bounds, alternative states,
or a formula for the remaining unknowns. Families use exact inputs; use SymPy
``Rational(9, 10)`` or ``Rational("0.9")`` for fractional or decimal observations.

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
>>> family.expression(check.total) == check.reading.x**2 + 1
True
>>> family.values(check.total)
Interval(1, oo)

``problem.symbols`` provides model quantities by their usual attribute paths.
Use these symbols to ask about outputs or add conditions. ``family.symbols``
provides the same quantities.

The family uses the model's expressions and domains. Here one real coordinate
describes every possible state. ``expression(...)`` rewrites a scalar expression
in the remaining coordinates; ``values(...)`` maps it to a SymPy set of possible
values. Both accept model symbols or expressions built from them. For example:

>>> family.expression(check.total - check.reading.squared)
1

Continue with SymPy
-------------------

The returned expression uses the family's actual symbols. Pass it directly to
SymPy for calculus or numerical compilation:

>>> from sympy import diff, lambdify
>>> response = family.expression(check.total)
>>> diff(response, check.reading.x) == 2 * check.reading.x
True
>>> calculate = lambdify(check.reading.x, response, modules="math")
>>> calculate(2)
5

Use ``expression`` before differentiating to include dependencies resolved by
the model. The family retains admissibility conditions; a compiled numerical
function evaluates the formula without checking those conditions. For finite
vector outputs, :ref:`export their elements <custom-vector-operations>`.

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
and the added conditions before being returned. Components retain their model
types. The source problem and earlier families stay unchanged; see the
:doc:`API reference <api>` for copying and model methods.

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

``section(..., through=case)`` varies selected current coordinates and fixes
the others at a checked model instance. Derived outputs follow the equations;
selecting no coordinates gives the reference point:

>>> point = family.section(through=chosen)
>>> point.values(check.reading.x, check.total)
{(2, 5)}

Pass symbols such as ``check.reading.x`` to select coordinates. In contrast,
``values(...)`` projects outputs while letting all remaining coordinates vary.

The reference is checked against the equations, domains, and added conditions:
a violation raises ``ModelValidationError`` and an undecided check raises
``UnresolvedSolutionError`` with its candidate audit. Sections can remain
discrete or coupled. The :doc:`example <examples/solution_families/index>`
explores their dimensions and response plots.

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

Materialised cases retain model types and methods; custom constructor hooks
are not invoked. See :doc:`api` for model and component construction behavior.

A family represents the captured equations over finite, commutative scalar
values. Every field's declared assumptions remain explicit domain restrictions,
including assumptions on quantities eliminated during reduction. Inferred
component values remain constraints just as they are in the recorded problem.
``family.conditions`` retains added symbolic relations; ``family.problem``
records constructor-style observations and their origins. Its symbols retain
their identities across refinements. Symbols from another prepared problem
raise ``ModelValidationError``, even when their printed names coincide.

Some questions retain conditions rather than reducing to an explicit list of
answers. External parameters and unevaluated functions may leave conditional
results. A failed reduction does not imply an empty family. See
:doc:`generality` for the supported problem classes.

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
observations. Finite sums and products, and evaluable integrals, derivatives,
and limits are expanded through model definitions before exploring. Operators
that remain unevaluated raise ``ModelInputError``. Vector observations use the
same scalar coordinate domains; vector queries retain coupled tuple outputs.
See :doc:`vectors` for examples. ``where(...)`` accepts symbolic relations and Python booleans.
Unsupported domain predicates raise ``ModelInputError`` instead of being omitted.
Unsupported calculations remain explicit.

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

.. automethod:: SolutionFamily.latex

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
