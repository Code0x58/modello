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
   * - No solution satisfies the equations and assumptions
     - ``ValueError: 0 solutions ...``
   * - Several discrete solutions remain
     - ``ValueError: 2 solutions ...`` (or another count).

Counts of inputs and equations alone cannot establish consistency or uniqueness.
The following small model illustrates how to investigate each case:

>>> from modello import InstanceDummy, Modello, component
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
particular result you need. An empty set means
that expression has no unresolved symbolic dependencies; it does not prove
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

Changing that total makes the joint system inconsistent. Catch the exception
to see the model, its equations, and the supplied values:

>>> try:
...     Check("conflict", reading=known, total=6)
... except ValueError as error:
...     print(error)
0 solutions for Check('conflict')
Equations and supplied values (self.* are model fields):
  self.total = self.reading.squared + 1
  self.reading.squared = self.reading.x**2
  self.reading.x = 2
  self.reading.squared = 4
  self.total = 6

Here the child fixes ``x = 2``, so its square is 4 and the parent's total must
be 5. Remove or correct the conflicting observation. Adding more observations
cannot repair that contradiction.

The report uses ``self.`` paths for model fields and includes derived values
copied from supplied components. Aliased fields share a quantity and may appear
under one representative path. It shows the submitted system, rather than
identifying a smallest conflicting subset.

Resolve ambiguity with a meaningful constraint
-----------------------------------------------

Observing only a total of 5 leaves two real readings, ``-2`` and ``2``:

>>> try:
...     Check("ambiguous", total=5)
... except ValueError as error:
...     print(error)
2 solutions for Check('ambiguous')
Equations and supplied values (self.* are model fields):
  self.total = self.reading.squared + 1
  self.reading.squared = self.reading.x**2
  self.total = 5

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
... except ValueError as error:
...     print(str(error).splitlines()[0])
0 solutions for PositiveCheck('outside_domain')
>>> PositiveCheck("inspect_domain").reading.x.is_positive
True

Check the assumptions on the symbolic field, not just the supplied number.
Undefined expressions, such as ``1 / x`` at zero, can also leave no valid
solution. Assumptions about external symbolic parameters may remain undecided;
successful construction is not proof of all possible parameter values' validity.

The solution count describes retained solver answers, not the number of numeric
points represented by a symbolic model. The unresolved ``partial`` model above
represents a family of readings and is valid.

Narrow a failure to the layer that introduces it
------------------------------------------------

* Start with a symbolic instance and inspect the output expression and its
  dependencies. Add observations incrementally until the unexpected result or
  exception appears.
* For a composed model, check the input components separately, then the parent
  that joins them. A valid child can conflict with a parent's constraints. An
  invalid explicitly constructed child fails before the parent receives it.
* Put cross-component observations on a joining model. Leave its children
  partially symbolic so the joint solve can determine their missing quantities.
  The :doc:`portfolio example <examples/portfolio/index>` demonstrates fitting
  through such an outer constraint.
* Reconstruct the model when changing observations. Assigning an attribute on
  an existing result does not solve its equations again.

Unknown field names, invalid scalar inputs, and incompatible component types
raise ``TypeError`` before solving. Other exceptions from SymPy can indicate
unsupported mathematics; keep the smallest reproducing model and the original
exception when investigating those cases. See :doc:`api` for input rules and
:doc:`nesting` for component copying, scope, and refinement.
