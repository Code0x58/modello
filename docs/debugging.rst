Debugging a model
=================

Start with the question the observations leave unresolved. A missing value,
conflicting measurements, several valid states, and an unsupported calculation
need different responses:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - What you see
     - What to try
   * - An output remains symbolic
     - Keep the formula, or supply information that determines its inputs.
   * - Additional observations agree
     - Keep them as a consistency check; redundancy is allowed.
   * - Observations appear to conflict
     - Inspect the recorded equations and ask for a conflict report.
   * - Several answers are possible
     - Add a meaningful domain or condition, or explore a solution family.
   * - A calculation remains unresolved
     - Inspect the remaining conditions and check the supported problem classes.

A solve failure retains a ``problem`` snapshot. Its candidate count alone does
not establish whether the mathematical model is inconsistent. Use the
:doc:`diagnostic reference <diagnostics>` for report guarantees and the
:ref:`error reference <errors>` for exception handling.

The following small model illustrates how to investigate each case:

>>> from modello import InstanceDummy, Modello, ModelSolveError, SolutionCountError, component, inspect_model
>>> class Reading(Modello):
...     x = InstanceDummy("x", real=True)
...     squared = x**2
>>> class Check(Modello):
...     reading = component(Reading)
...     total = reading.squared + 1

Inspect what remains symbolic
------------------------------

Without an observation, the result depends on the reading's unknown ``x``:

>>> partial = Check("partial")
>>> partial.total
_partial_reading_x**2 + 1
>>> partial.total.free_symbols
{_partial_reading_x}

The bound symbol belongs to ``partial.reading.x``. Check ``free_symbols`` on the
particular result you need. An empty set means no free symbols occur in that
expression; an unevaluated function such as ``f(1)`` can still prevent a numerical
result. It does not establish that every other field is determined. Nor is the
number of free symbols in one expression a general count of the system's degrees
of freedom.

Dependencies can also be intentional external parameters:

>>> from sympy import Symbol
>>> parameter = Symbol("parameter", real=True)
>>> parameterised = Check("external", reading=Reading("input", x=parameter))
>>> parameterised.total
parameter**2 + 1

An external symbol is distinct from a model's independently bound quantity,
even if their printed names look similar. Supply another observation or retain
the expression for later substitution or differentiation, according to the
question you want the model to answer.

Distinguish redundancy from contradiction
------------------------------------------

Supplying both the reading and its total is valid when they agree:

>>> known = Reading("known", x=2)
>>> Check("consistent", reading=known, total=5).total
5

Changing that total conflicts with the reading. Catch the exception and keep
its structured snapshot for investigation:

>>> try:
...     Check("conflict", reading=known, total=6)
... except SolutionCountError as error:
...     problem = error.problem
...     print(error.solution_count)
0
>>> problem
ModelProblem(Check, 'conflict', fields=3, constraints=5)
>>> print(problem)
Check('conflict')
  c0: total = reading.squared + 1 [definition]
  c4: total = 6 [input 'total']
  reading: Reading
    c1: squared = x**2 [definition]
    c2: x = 2 [input 'reading']
    c3: squared = 4 [inferred in 'reading']

Here the child fixes ``x = 2``, so its square is 4 and the parent's total must
be 5. Remove or correct the conflicting observation. Adding more observations
cannot repair that contradiction.

Within each component, equations omit the shared path prefix, so the reading's
definition is simply ``squared = x**2``. When an equation crosses layers, both
sides use paths from their nearest common ancestor; see the connection example
below. ``problem.format(expression)`` uses full paths from the model root for
standalone expressions. External
parameters appear as ``external[name]``; distinct external symbols sharing a
name are numbered. Copied component values
retain whether they were supplied or inferred during construction: ``x = 2``
was an input when building ``known``, while ``squared = 4`` was inferred.
Supplying ``known`` as ``reading`` to ``Check`` preserves that distinction:
``x`` remains an input, grouped under the ``reading`` argument.
The distinction comes from the actual construction inputs, so solving a
positive reading backwards from ``squared=4`` instead marks ``x`` as inferred.
Aliased fields share a quantity and may appear
under one representative path. The original identities remain available through
``problem.fields``, ``problem.aliases(field)``, and ``problem.uses(field)``.

Resolve ambiguity with a meaningful constraint
-----------------------------------------------

Observing only a total of 5 leaves two real readings, ``-2`` and ``2``:

>>> try:
...     Check("ambiguous", total=5)
... except SolutionCountError as error:
...     ambiguous = error.problem
...     print(error.solution_count, error.returned_count)
2 2
>>> field = ambiguous.field("reading", "x")
>>> sorted(candidate.value(field) for candidate in ambiguous.candidates)
[-2, 2]
>>> answers = ambiguous.analyse()
>>> answers.status
'consistent'
>>> roots, = answers.solution_sets
>>> roots.field == field, roots.values, roots.domain
(True, {-2, 2}, Reals)

Here ``consistent`` establishes existence, not uniqueness. Each candidate can
be audited with ``ambiguous.check(candidate)``. Its ``verdict`` is ``satisfied``,
``violated``, or ``unknown``; residual and assumption checks explain the outcome.

Modello does not choose a branch arbitrarily. Supply an observation that selects
the intended answer, or refine the model's domain if that is part of its
meaning. For example, a positive reading selects the positive root:

>>> class PositiveReading(Reading):
...     x = InstanceDummy("x", positive=True)
>>> class PositiveCheck(Check):
...     reading = component(PositiveReading)
>>> PositiveCheck("positive", total=5).reading.x
2

Assumptions are constraints too. A total of 1 would require ``x = 0``, which
violates this model's strict positivity:

>>> try:
...     PositiveCheck("outside_domain", total=1)
... except SolutionCountError as error:
...     print(error.solution_count)
0
>>> PositiveCheck("inspect_domain").reading.x.is_positive
True

Check the assumptions on the symbolic field, not just the supplied number.
Undefined expressions, such as ``1 / x`` at zero, can also leave no valid
solution. Assumptions about external symbolic parameters may remain undecided;
successful construction is not proof of all possible parameter values' validity.

``solution_count`` counts retained solver candidates. A candidate may contain
symbolic expressions representing a family of numerical answers, as in the
unresolved ``partial`` model above. ``returned_count`` counts captured candidates
before domain filtering; ``error.problem.candidates`` retains them for inspection,
including rejected proposals.

``solution_sets`` uses SymPy's ``solveset`` where equations reduce to one
unknown after forced numerical substitutions. Unlike a finite candidate list,
a SymPy set can retain an infinite family or an unresolved ``ConditionSet``:

>>> from sympy import ImageSet, pi, sin
>>> class Oscillation(Modello):
...     angle = InstanceDummy("angle", real=True)
...     sine = sin(angle)
>>> periodic = inspect_model(Oscillation, "periodic", sine=0).analyse()
>>> family, = periodic.solution_sets
>>> family.values.has(ImageSet), family.values.contains(7 * pi)
(True, True)

These are necessary values for the recorded subsystem, with its substitutions
and search domain retained. Other equations and declared predicates can still
exclude values; a nonempty set alone does not establish whole-model consistency.
An unresolved set is not evidence of inconsistency. See :doc:`diagnostics` for
the scope of these results.

To retain the joint family, add observations, inspect possible outputs, or
select several valid model instances, see :doc:`solutions`. That exploration
starts from the same recorded problem, including one retained on an exception.

Inspect components and their connections
-----------------------------------------

You can prepare the same problem without solving it, then inspect components
and the equations crossing their boundaries:

>>> prepared = inspect_model(Check, "investigate", reading=known, total=6)
>>> reading = prepared.at("reading")
>>> reading.model is Reading
True
>>> tuple(c.path for c in reading.boundary_constraints)
(('total',),)

``reading.owned_constraints`` contains the subtree's definitions, copied
observations, and inferred construction results. Its boundary includes the
parent's total equation. This describes
structure; a conflict can span individually valid layers, with no single guilty
component.

A copied symbolic input can reference a quantity in an enclosing component.
For example, link the reading to its check's total, then inspect that check
inside another layer:

>>> class Audit(Modello):
...     check = component(Check)
>>> linked_reading = Reading("linked_reading", x=Audit.check.total - 1)
>>> linked_check = Check("linked_check", reading=linked_reading)
>>> linked = inspect_model(Audit, "linked", check=linked_check)
>>> print(linked)
Audit('linked')
  check: Check
    c0: total = reading.squared + 1 [definition]
    c2: total = (total - 1)**2 + 1 [inferred in 'check']
    reading: Reading
      c1: squared = x**2 [definition]
      c3: reading.x = total - 1 [input 'check']
      c4: reading.squared = (total - 1)**2 [inferred in 'check']

In ``c1``, both fields belong to ``reading``, so neither needs a prefix. In
``c3``, ``total`` belongs to its parent ``check``. Both sides are therefore
relative to ``check``: ``reading.x = total - 1``. The common ``check.`` prefix
is omitted; keeping ``reading.`` on the left makes the cross-layer connection
visible. Structured constraint paths remain complete:

>>> linked.constraints[3].path
('check', 'reading', 'x')

An explicitly constructed child such as ``Reading("bad", x=2, squared=5)``
raises before it can be passed to ``Check``. In that case, inspect the child's
exception: the parent has not been constructed. A supplied child has therefore
already passed construction in its own context, so investigate the additional
parent observations and connections first. That does not prove the child is
correct for the intended application or for every external parameter value.

``component(Reading)`` declares a type rather than constructing a child.
Class definitions do not check satisfiability: even an inconsistent child type
can appear in a parent declaration. An omitted child, or a supplied child with
symbolic quantities, can gain constraints when the parent is solved. Those
constraints can conflict with the
child's equations or domain even though its declaration or earlier construction
succeeded.

Ask for evidence explicitly
----------------------------

Analysis is opt-in because it can simplify expressions, perform exact linear
algebra, generate and check solver candidates, and assess many subsets:

>>> report = problem.analyse(conflicts=True, relaxations=True)
>>> print(report)
Check('conflict'): inconsistent
  An exact linear combination proves a contradiction.
  Exact equality rank 3; augmented rank 4; nullity 0 (affine subset only, domains excluded).
  Proven conflict (irreducible; declared domains retained):
    c0: total = reading.squared + 1 [definition]
    c4: total = 6 [input 'total']
    reading: Reading
      c3: squared = 4 [inferred in 'reading']
    Certificate: (-1)*c0 + (-1)*c3 + (1)*c4 gives -1 = 0.
  Input removals:
    reading: consistent
    total: consistent
  Subset assessments: 6.
>>> report.conflict.linear_certificate.verify()
True

The smaller conflict proves that the copied square and observed total cannot
both agree with the parent's equation. ``irreducible`` means removing any one
member admits a checked witness, with declared symbol domains retained. It does
not mean the smallest possible conflict or the only conflict.

``report.input_checks`` retains every attempted whole-input removal, including
``inconsistent`` and ``unknown`` outcomes. ``report.relaxations`` selects only those
with validated witnesses. Removing ``reading`` releases all its copied values,
including derived fields. These are possibilities to investigate, not choices
of which observation is wrong:

>>> reconsidered = problem.without("reading").analyse()
>>> reconsidered.status
'consistent'

For a single report in an exception handler, invoke analysis at that call site:

.. code-block:: python

   try:
       result = Check("case", reading=known, total=6)
   except ModelSolveError as error:
       print(error.problem.analyse(
           conflicts=True, relaxations=True, max_checks=100,
       ))

``max_checks`` limits distinct subset assessments, not elapsed time. A single
symbolic operation may be expensive. Exhausted budgets and inconclusive checks
remain visible; neither is evidence of inconsistency. ``unknown`` means the
analyser has not established a result, and can occur even for a valid model.
Float arithmetic is not used for exact equality proofs; use SymPy ``Rational``
values when exact decimal relationships matter. See :doc:`diagnostics` for
the guarantees, supported checks, and research behind the tools.

Use observations on joining models to ask questions across components, as in
the :doc:`portfolio example <examples/portfolio/index>`. Reconstruct a model
when changing observations; changing an attribute does not solve it again.

Unknown field names, invalid scalar inputs, and incompatible component types
raise ``ModelInputError`` (a ``TypeError``) before solving. Backend failures
raise ``SolverBackendError`` with the original ``backend_error`` also chained as
the cause, with the recorded problem still available. Keep that
cause when investigating unsupported mathematics. See :doc:`api` for input rules and
:doc:`nesting` for component copying, scope, and refinement.
