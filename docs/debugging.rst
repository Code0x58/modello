Debugging a model
=================

A model can be useful before every quantity is numerical. An under-specified
model may describe a family of answers. Over-specifying a model with extra
observations can be useful for consistency checks. Distinguish these cases:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Situation
     - Behaviour
   * - Some quantities remain undetermined
     - Construction can succeed with symbolic expressions.
   * - Extra observations agree with the equations
     - Construction succeeds; redundancy is allowed.
   * - The constructor retains no solver candidate
     - ``ValueError: 0 solutions ...``
   * - Several solver candidates remain
     - ``ValueError: 2 solutions ...`` (or another count).

Counts of inputs and equations alone cannot establish consistency or uniqueness.
Solve failures raise ``ModelSolveError``, a subclass of ``ValueError`` carrying
the recorded problem. A reported zero count alone is not an inconsistency proof.
The following small model illustrates how to investigate each case:

>>> from modello import InstanceDummy, Modello, ModelSolveError, component, inspect_model
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
particular result you need. An empty set means that expression has no unresolved
symbolic dependencies; it does not prove
every other field in the model is determined. Nor is the number of free symbols
in one expression a general count of the system's degrees of freedom.

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
... except ModelSolveError as error:
...     problem = error.problem
...     print(error)
0 solutions reported for Check('conflict')
Inspect .problem or call .problem.analyse() for an opt-in diagnostic report.
>>> problem
ModelProblem(Check, 'conflict', fields=3, constraints=5)
>>> print(problem)
Check('conflict')
  c0 [definition total]: self.total = self.reading.squared + 1
  c1 [definition reading.squared]: self.reading.squared = self.reading.x**2
  c2 [observation reading.x]: self.reading.x = 2
  c3 [observation reading.squared]: self.reading.squared = 4
  c4 [observation total]: self.total = 6

Here the child fixes ``x = 2``, so its square is 4 and the parent's total must
be 5. Remove or correct the conflicting observation. Adding more observations
cannot repair that contradiction.

The snapshot uses ``self.`` paths for model fields and includes derived values
copied from supplied components. Aliased fields share a quantity and may appear
under one representative path. The original identities remain available through
``problem.fields``, ``problem.aliases(field)``, and ``problem.uses(field)``.

Resolve ambiguity with a meaningful constraint
-----------------------------------------------

Observing only a total of 5 leaves two real readings, ``-2`` and ``2``:

>>> try:
...     Check("ambiguous", total=5)
... except ModelSolveError as error:
...     ambiguous = error.problem
>>> field = ambiguous.field("reading", "x")
>>> sorted(candidate.value(field) for candidate in ambiguous.candidates)
[-2, 2]
>>> ambiguous.analyse().status
'consistent'

Here ``consistent`` establishes existence, not uniqueness. Each candidate can
be audited with ``ambiguous.check(candidate)``; its residual and assumption
checks distinguish satisfied, violated, and undecided outcomes.

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
... except ModelSolveError as error:
...     print(str(error).splitlines()[0])
0 solutions reported for PositiveCheck('outside_domain')
>>> PositiveCheck("inspect_domain").reading.x.is_positive
True

Check the assumptions on the symbolic field, not just the supplied number.
Undefined expressions, such as ``1 / x`` at zero, can also leave no valid
solution. Assumptions about external symbolic parameters may remain undecided;
successful construction is not proof of all possible parameter values' validity.

The solution count describes retained solver answers, not the number of numeric
points represented by a symbolic model. The unresolved ``partial`` model above
represents a family of readings and is valid.

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

``reading.owned_constraints`` contains the subtree's definitions and copied
observations. Its boundary includes the parent's total equation. This describes
structure; a conflict can span individually valid layers, with no single guilty
component. An invalid explicitly constructed input still fails before it can
reach the parent.

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
  c0 [definition total]: self.total = self.reading.squared + 1
  c3 [observation reading.squared]: self.reading.squared = 4
  c4 [observation total]: self.total = 6
  Certificate: (-1)*c0 + (-1)*c3 + (1)*c4 gives -1 = 0.
Removing input 'reading': consistent.
Removing input 'total': consistent.
Subset assessments: 6.
>>> report.conflict.linear_certificate.verify()
True

The smaller conflict proves that the copied square and observed total cannot
both agree with the parent's equation. ``irreducible`` means removing any one
member admits a checked witness, with declared symbol domains retained. It does
not mean the smallest possible conflict or the only conflict.

``report.input_checks`` retains every attempted whole-input removal, including
inconsistent or undecided outcomes. ``report.relaxations`` selects only those
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
raise ``TypeError`` before solving. Backend solver failures are chained as the
``ModelSolveError`` cause, with the recorded problem still available. Keep that
cause when investigating unsupported mathematics. See :doc:`api` for input rules and
:doc:`nesting` for component copying, scope, and refinement.
