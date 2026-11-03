How much time should we spend checking a record?
================================================

An archive is digitising paper records. Typing faster increases its daily
output, but mistakes create work later. A colleague can spend time checking
each entry before it leaves the team. How much checking is worthwhile?

We will build one model of that decision, propose a checking time using
calculus, and establish the properties that make the calculation valid. Then
we will change the response curve and revisit the same argument. You need
Python, derivatives, and the idea of a concave function; no finance or
optimisation background is assumed. :doc:`/getting-started` introduces the
underlying Modello declarations.

The executable example is ``examples.record_review``. Its symbolic calculations
need only the ordinary ``modello`` installation. Every input below is
illustrative: verifying the mathematics does not establish that the chosen
response curve describes a real team.

Start with the workload budget
-------------------------------

The older ``examples.streaming`` data-entry example relates staff, processing
rates, and cost. We start from the same practical question using a simpler
identity: staff minutes equal records multiplied by minutes per record.
Modello lets us supply quantities on either side of that equation.

>>> from examples.record_review import EntryBudget
>>> day = EntryBudget("day", staff_minutes=480, minutes_per_record=4)
>>> day.records
120
>>> target = EntryBudget("target", records=150, minutes_per_record=4)
>>> target.staff_minutes
600

This budget counts work, rather than elapsed time or queueing delays. Its
record count is a continuous planning quantity. We have so far counted only
typing; checking and later repairs must come from the same pool of staff time.

Give checking a benefit and a cost
----------------------------------

Suppose typing takes four minutes per record. One in ten records would need
a later repair, taking sixty staff minutes. Without checking, repairs therefore
add :math:`(1/10)\times60=6` minutes per record **in expectation**. Total expected
work is ten minutes per record, giving capacity of 48 records per 480 staff
minutes under this workload approximation.

Let :math:`t\geq0` be checking minutes per record, and let :math:`u(t)` be the
fraction of those otherwise missed errors that checking catches. For a first
model, use

.. math::

   u(t)=1-e^{-kt},\qquad k>0.

The parameter :math:`k` controls how quickly checking becomes effective.
The curve starts at zero and approaches one: extra checking keeps helping,
with diminishing returns. We assume that a caught error needs no further
repair time beyond the checking time already counted.

Use minutes as the common unit for benefit and cost:

.. math::

   G(t)&=a\,u(t)-t,\\
   M(t)&=e+t+a(1-u(t))=e+a-G(t).

Here :math:`G` is net staff time saved compared with no checking, and :math:`M`
is total expected staff time per record. Maximising :math:`G` minimises
:math:`M`. The benefit :math:`a\,u(t)` plays the role of a utility function;
its units are avoided repair minutes, so we can interpret its scale directly.

.. list-table:: Illustrative inputs
   :header-rows: 1

   * - Quantity
     - Meaning
     - Value
   * - :math:`e`
     - Entry time
     - 4 minutes per record
   * - :math:`a`
     - Repair work that perfect checking could avoid
     - 6 minutes per record
   * - :math:`k`
     - Checking effectiveness rate
     - :math:`1/2` per minute

.. raw:: html

   <figure>
     <img src="study/curves.svg" alt="Checking has diminishing benefit. Net saving rises and then falls; exponential, rational, and capped responses give different checking decisions." style="max-width:100%;height:auto">
     <figcaption>The left panel separates benefit, checking cost, and net saving.
     The right panel compares net saving for the replacements we will investigate below.</figcaption>
   </figure>

State what the calculation needs
--------------------------------

Three distinct properties matter:

* **Strict increase:** more checking should catch a larger fraction of errors.
* **Concavity:** the marginal benefit should diminish. A concave net saving
  lets us turn a stationary point into a global maximum.
* **C² regularity:** the first two derivatives should be continuous. We will
  use a second-order approximation to describe workload near the chosen time.

For the exponential curve, ordinary calculus gives
:math:`u'(t)=ke^{-kt}` and :math:`u''(t)=-k^2e^{-kt}`. Its increase needs
:math:`k>0`; concavity and C² regularity hold for every real :math:`k`.
These are separate mathematical facts. A smooth curve can still be a poor
description of checking.

``specifies`` attaches these claims to the response field. ``VerificationModel``
attempts them when the class is defined, retaining conditional and failed
claims so we can explore them:

.. literalinclude:: ../../../examples/record_review.py
   :language: python
   :pyobject: ExponentialReview

``effort`` is a ``Coordinate``: the argument that varies along the curve.
``rate`` is an ``InstanceDummy`` parameter. The coordinate and the obligations
both state the nonnegative effort domain; an observation of ``rate`` can
specialise a curve without fixing its argument. Keep symbolic inputs exact,
for example ``Rational(1, 2)`` rather than a rounded decimal.

>>> from examples.record_review import ExponentialReview
>>> definition = ExponentialReview.definition_report()
>>> definition.explain("value/strictly_increasing").verdict
'conditionally true'
>>> definition.explain("value/concave").verdict
'strictly true'
>>> definition.explain("value/continuously_differentiable").verdict
'strictly true'

A verdict describes a claim under its recorded domain and premises.
``strictly true`` means established, ``conditionally true`` retains a
sufficient parameter condition, and ``strictly false`` records a checked
refutation. ``unresolved`` means the current attempt supplies neither a proof
nor a refutation. It can call for a different argument, not just different data.

Explore a premise before choosing data
--------------------------------------

A ``VerificationContext`` is an immutable branch of the investigation.
Inspect the parameter condition using that context's symbols, then add it as
an explicit assumption:

>>> from modello.contrib.verification import VerificationContext
>>> curve = VerificationContext.for_model(ExponentialReview)
>>> increase = curve.explain("value/strictly_increasing")
>>> increase.condition == (curve.symbols.rate > 0)
True
>>> positive_rate = curve.assume(curve.symbols.rate > 0)
>>> positive_rate.explain("value/strictly_increasing").verdict
'strictly true'

An assumption restricts allowed parameter values; an observation fixes a value
for a particular branch. With zero effectiveness, the curve is constant and
the strict increase claim fails:

>>> curve.observe(rate=0).explain("value/strictly_increasing").verdict
'strictly false'
>>> curve.explain("value/strictly_increasing").verdict
'conditionally true'

The earlier context still has its original premises. We can also remove an
explicit assumption on a new branch, recording why it changed:

>>> reopened = positive_rate.without_assumptions(
...     curve.symbols.rate > 0, reason="Investigate ineffective checking too.")
>>> reopened.explain("value/strictly_increasing").verdict
'conditionally true'

Observations accumulate. To compare different data, branch from the common
context before either value was supplied. Contradictory premises cannot
establish a conclusion; reports track assumption consistency separately.

Compose the decision model
--------------------------

Now put the response inside the workload model. ``candidate`` is a proposed
checking time; it is separate from the coordinate over which we compare all
possible checking times.

.. literalinclude:: ../../../examples/record_review.py
   :language: python
   :pyobject: ReviewPlanning

``Derivative(..., evaluate=False)`` defers differentiation until Modello has
resolved ``net_saving`` to its formula. An eager derivative of a named field
would differentiate its placeholder symbol instead. ``Subs`` then evaluates
that derivative, or another curve, at the candidate.

The native derivative also creates a differentiability obligation
automatically. That obligation needs an existing derivative; the declared C²
claim is stronger and serves our later quadratic approximation.

>>> from sympy import Rational, log, simplify
>>> from examples.record_review import ReviewPlanning
>>> measured = VerificationContext.for_model(
...     ReviewPlanning, "records", entry_minutes=4, avoidable_minutes=6,
...     review=ExponentialReview("response", rate=Rational(1, 2)))
>>> measured.explain("review.value/strictly_increasing").verdict
'strictly true'

The nested response keeps its claims with the prefix ``review.value``.
This context still leaves the candidate open.

Propose a checking time
-----------------------

With our inputs,

.. math::

   G(t)=6(1-e^{-t/2})-t,\qquad G'(t)=3e^{-t/2}-1.

Setting :math:`G'(t)=0` gives the candidate :math:`t_*=2\log3`, about
2.20 minutes. We can confirm stationarity exactly before making any numerical
comparison:

>>> proposed = measured.observe(candidate=2*log(3))
>>> proposed.expression("candidate_marginal").doit()
0

Stationarity alone does not establish an optimum. A stationary point might
be a minimum, and a numerical grid might miss a better point.

.. _why-this-is-a-global-optimum:

Explain why it is a global optimum
----------------------------------

For a differentiable concave function, its tangent line lies above its graph:

.. math::

   G(y)\leq G(s)+G'(s)(y-s).

If :math:`s\geq0` and :math:`G'(s)=0`, this becomes
:math:`G(y)\leq G(s)` for every :math:`y\geq0`. Thus a feasible stationary
candidate maximises net saving over the whole effort domain. Subtracting the
linear checking cost preserves the concavity of the response benefit.

``InferenceRule`` records this mathematical implication, with named premises.
The rule is **trusted reasoning supplied by the example**; the library
recursively checks its premises and retains them in the evidence. Registering
a rule does not make the library prove the implication itself.

.. literalinclude:: ../../../examples/record_review.py
   :language: python
   :start-at: GLOBAL_MAXIMUM = Obligation(
   :end-before: @dataclass

The conclusion states :math:`G(s)\geq G(t)` on the effort domain. The premises
establish concavity, differentiability, stationarity, and feasibility. Add the
rule to the common branch point, then supply the proposed candidate:

>>> from examples.record_review import STATIONARY_MAXIMUM, review_plan
>>> ready = measured.with_rule(STATIONARY_MAXIMUM)
>>> proposed = ready.observe(candidate=2*log(3))
>>> result = proposed.run(review_plan)
>>> result.report.explain("global_maximum").verdict
'strictly true'
>>> all(p.verdict == "strictly true"
...     for p in result.report.explain("global_maximum").premises)
True

The ``review_plan`` method reads the candidate workload and saving. Its
``uses`` paths consume the response's declared properties through those
dependencies. Its explicit preconditions add the global-maximum conclusion
and a positive workload. Execution starts only once the consumed premises are
established. Its Python callback evaluates the resulting expressions; that
callback, the registered rule, and the CAS remain part of the trust boundary.

>>> result.data.net_saving
4 - 2*log(3)
>>> result.data.minutes_per_record
2*log(3) + 6
>>> result.data.workload_curvature
1/2

We save about 1.80 staff minutes per record. Total expected work falls from
10 to about 8.20 minutes. Returning to our original budget gives a practical
interpretation:

>>> day = EntryBudget("day", staff_minutes=480,
...                   minutes_per_record=result.data.minutes_per_record)
>>> round(float(day.records), 1)
58.6
>>> target = EntryBudget("target", records=60,
...                      minutes_per_record=result.data.minutes_per_record)
>>> round(float(target.staff_minutes), 1)
491.8

The workload curvature is :math:`M''(t_*)=1/2`. C² regularity justifies the
local expansion

.. math::

   M(t_*+h)=M(t_*)+\tfrac14h^2+o(h^2).

Small changes around the optimum therefore have only a second-order workload
cost. This is a local approximation, while the concavity argument establishes
global optimality. Expected workload capacity also remains an approximation;
it is not a guarantee of how many records finish in a particular day.

Try a different response without changing the argument
------------------------------------------------------

The exponential curve was a modelling choice. Suppose we instead propose

.. math::

   u(t)=\frac{kt}{1+kt}.

This response has the same initial slope :math:`k`, but saturates more slowly.
``RationalReview`` inherits the same property declarations. Replacing the
component rechecks them against the new formula and preserves the observed
rate. Start from ``ready`` so the candidate has not yet been fixed:

>>> from sympy import sqrt
>>> from examples.record_review import RationalReview
>>> changed = ready.replace("review", RationalReview)
>>> changed = changed.observe(candidate=2*(sqrt(3)-1))
>>> alternative = changed.run(review_plan)
>>> alternative.report.explain("global_maximum").verdict
'strictly true'
>>> alternative.data.minutes_per_record
2 + 4*sqrt(3)
>>> round(float(alternative.data.effort), 2)
1.46
>>> result.data.minutes_per_record == 6 + 2*log(3)
True

The new candidate solves :math:`G'(t)=3/(1+t/2)^2-1=0`.
The same concavity argument establishes its optimum. Keeping the exponential
candidate instead would fail the stationarity premise; replacing a component
does not preserve conclusions about the old formula. Our earlier result
remains available for comparison.

Learn from a failed replacement
-------------------------------

A capped linear response, :math:`u(t)=\min(kt,1)`, looks convenient: a constant
benefit per minute until all errors have been caught. At :math:`k=1/2`, the
join is at two minutes. The left derivative is :math:`1/2`, while the right
derivative is zero. The curve is not differentiable there, hence cannot be C².

``CappedReview`` inherits the original claims too. A failed claim is retained
for inspection rather than preventing the class from being defined:

>>> from examples.record_review import CappedReview
>>> from modello.contrib.verification import VerificationError
>>> capped = ready.replace("review", CappedReview).observe(candidate=2)
>>> capped.explain("review.value/continuously_differentiable").verdict
'strictly false'
>>> try:
...     capped.run(review_plan)
... except VerificationError as error:
...     print(error.report.explain("global_maximum").verdict)
unresolved

Two minutes actually maximises this piecewise model: net saving rises before
the join and falls after it. Our smooth stationary-point method cannot
establish that result. A method for corners would need a different argument,
such as the one-sided slopes. The capped response also has a plateau, so the
original strict increase claim no longer describes it.

Finite observations are still useful for exploring its shape:

>>> sampled = capped.sample("review.value", wrt="effort", points=(1, 2, 3))
>>> tuple(float(value) for value in sampled.values)
(0.5, 1.0, 1.0)
>>> sampled.kind
'numerical observation'

A ``Sample`` supplies observations, not evidence of a global property or
regularity at a join. It cannot discharge the method's failed premises.

Separate a policy target from a mathematical premise
----------------------------------------------------

Suppose the team hopes to save at least two minutes per record. Add that as
an exploratory goal, associated with the method that must respect it:

>>> from modello.contrib.verification import predicate
>>> ambitious = proposed.with_obligations(predicate(
...     proposed.symbols.candidate_saving >= 2,
...     name="saving_target", consumer="review_plan",
...     reason="The proposed process should save two minutes per record."))
>>> ambitious.explain("saving_target").verdict
'strictly false'

The best exponential plan saves less than the target. ``run(review_plan)`` is
blocked on this branch. To retain the measured baseline for comparison while
investigating another process, retire the experimental target explicitly:

>>> baseline = ambitious.without_obligations(
...     "saving_target", reason="Keep the measured baseline for comparison.")
>>> baseline.run(review_plan).report.explain("global_maximum").verdict
'strictly true'
>>> ambitious.explain("saving_target").verdict
'strictly false'

This removes a context-added goal. It does not remove the response's
declarations or the method's preconditions. A real two-minute target would
call for changing the process, its inputs, or the target itself.

Help an open proof with a checked bound
---------------------------------------

Another question is whether *any* checking time could save seven minutes per
record. Our model has only six repair minutes to avoid, and checking itself
costs time, so six is a valid uniform upper bound on net saving. Use that
argument to help a bound obligation:

>>> from examples.record_review import EFFORT_DOMAIN
>>> from modello.contrib.verification import supremum_below
>>> ceiling = proposed.with_obligations(supremum_below(
...     "net_saving", wrt="effort", domain=EFFORT_DOMAIN,
...     bound=7, name="saving_ceiling"))
>>> ceiling.explain("saving_ceiling").verdict
'unresolved'
>>> bounded = ceiling.with_bound_hint("saving_ceiling", upper=6)
>>> bounded.explain("saving_ceiling").verdict
'strictly true'

``with_bound_hint`` proposes a certificate; it adds no assumption. The checker
establishes :math:`G(t)\leq6<7` uniformly. A loose or incorrect hint cannot
refute the original bound. Here ``unresolved`` meant a proof aid was missing,
even though the mathematical argument was short.

Keep evidence with the decision
-------------------------------

The successful result retains the report that permitted its computation.
``report.explain(id)`` exposes conditions, domains, derivation steps, witnesses,
and checked rule premises. ``VerificationError`` similarly retains its
originating context and report when a method is blocked.

>>> import json
>>> audit_json = result.report.to_json()
>>> json.loads(audit_json)["assumptions_consistent"]
True
>>> html_page = result.report.to_html(title="Record-checking decision")

These are descriptive exports, not executable saved contexts. To reproduce the
complete example, including the blocked replacement and exact result values,
run the module:

.. code-block:: console

   $ python -m examples.record_review --output artefacts/record-review

The output contains ``results.json`` with inputs, package versions, and exact
and decimal results, plus a JSON audit and an offline HTML report for each
response. To draw the figure as well, install the optional
``modello[numerics]`` extra and add ``--plot``. The documentation build publishes
the `results <study/results.json>`_, `baseline evidence <study/exponential.html>`_,
`alternative evidence <study/rational.html>`_, and
`blocked replacement <study/capped.html>`_.

For a further experiment, change the avoidable repair work in a new context,
derive a new candidate, and reuse the same rule. Check which premises still
hold before trusting the resulting capacity. A constrained or corner optimum
may require a different method rather than a different parameter value.
The :doc:`verification reference </contrib/verification>` explains the binding,
replacement, proof-domain, and execution rules in detail.
