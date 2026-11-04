How much time should we spend checking a record?
================================================

An archive is digitising paper records. Typing faster increases its daily
output, but mistakes create work later. A colleague can spend time checking
each entry before it leaves the team. How much checking is worthwhile?

We will build a workload model, use calculus to propose a checking time, and
check the properties that justify that decision. Then we will change the
checking response and revisit the result. You need Python and undergraduate
calculus; we explain the concavity argument as we use it.
:doc:`/getting-started` introduces Modello's declarations and observations.

The executable example is ``examples.record_review``. Its symbolic calculations
need only the ordinary ``modello`` installation. Every input below is
illustrative: verifying the mathematics does not establish that the chosen
response curve describes a real team.

Start with the workload budget
-------------------------------

Start with a familiar identity: staff minutes equal records multiplied by
minutes per record. Declare that relationship as an ordinary Modello model:

.. code-block:: python

   from modello import InstanceDummy, Modello

.. literalinclude:: ../../../examples/record_review.py
   :language: python
   :pyobject: EntryBudget

Each ``InstanceDummy`` names a quantity whose value can be supplied or inferred.
``records`` is the field name used for inputs; ``"n"`` is its symbolic label.
These two quantities are positive. We can supply values on either side of the
equation. The examples below import this declaration; the first argument to
``EntryBudget`` names the instance.

>>> from examples.record_review import EntryBudget
>>> day = EntryBudget("day", staff_minutes=480, minutes_per_record=4)
>>> day.records
120
>>> target = EntryBudget("target", records=150, minutes_per_record=4)
>>> target.staff_minutes
600

Staff time adds across the team: two people working for 240 minutes each
provide 480 staff minutes. We use a continuous record count as a planning
estimate. So far we have counted only typing; checking and later repairs must
come from the same pool of staff time.

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

Let :math:`b` be entry time and :math:`a` be the expected repair work that
perfect checking could avoid. Our illustrative inputs are:

.. list-table:: Inputs and units
   :header-rows: 1

   * - Quantity
     - Meaning
     - Value
   * - :math:`b`
     - Entry time
     - 4 minutes per record
   * - :math:`a`
     - Repair work that perfect checking could avoid
     - 6 minutes per record
   * - :math:`k`
     - Checking effectiveness rate
     - :math:`1/2` per minute

Now use minutes as the common unit for benefit and cost:

.. math::

   G(t)&=a\,u(t)-t,\\
   M(t)&=b+t+a(1-u(t))=b+a-G(t).

Here :math:`G` is net staff time saved compared with no checking, and :math:`M`
is total expected staff time per record. Maximising :math:`G` minimises
:math:`M`. Avoided repair minutes give us a concrete measure of utility, in
the same units as the checking cost.

.. raw:: html

   <figure>
     <img src="study/benefit.svg" alt="Avoided repair time rises with diminishing returns. Net saving rises to a peak, then falls as checking costs outweigh the extra benefit." style="max-width:100%;height:auto">
     <figcaption>Net saving is the avoided repair time minus the time spent checking.</figcaption>
   </figure>

State what the calculation needs
--------------------------------

Three distinct properties matter:

* **Strict increase:** more checking should catch a larger fraction of errors.
* **Concavity:** the marginal benefit should not increase. For a concave net saving,
  a point with zero derivative is a global maximum.
* **C² regularity:** the first two derivatives should be continuous. We will
  use a second-order approximation to describe workload near the chosen time.

For the exponential curve, ordinary calculus gives
:math:`u'(t)=ke^{-kt}` and :math:`u''(t)=-k^2e^{-kt}`. Its increase needs
:math:`k>0`; concavity and C² regularity hold for every real :math:`k`.
These are separate mathematical facts. A smooth curve can still be a poor
description of checking.

A property we ask the checker to establish is a **proof obligation**.
``specifies`` attaches obligations to a field. ``VerificationModel`` attempts
them when the class is defined, retaining each result for inspection.

In the declaration below, ``effort`` is a ``Coordinate`` for the curve's argument
:math:`t`. ``rate`` is an ``InstanceDummy`` parameter for :math:`k`. Although
our story expects a positive rate, we leave it unrestricted so the checker
can tell us which values establish the requested properties.

Field paths use the Python attribute names: ``wrt="effort"`` selects
:math:`t`. ``EFFORT_DOMAIN`` is
the interval of nonnegative real efforts, so the claims concern every
:math:`t\geq0`. The module uses these imports and domain:

.. code-block:: python

   from sympy import Interval, S, exp
   from modello import Coordinate, InstanceDummy
   from modello.contrib.verification import (
       VerificationModel, concave, continuously_differentiable,
       specifies, strictly_increasing,
   )

   EFFORT_DOMAIN = Interval(0, S.Infinity)

.. literalinclude:: ../../../examples/record_review.py
   :language: python
   :pyobject: ExponentialReview

``definition_report()`` contains the results of those checks. ``explain(id)``
selects one result: ``value/strictly_increasing`` combines the field path
``value`` with the obligation's name. The commands below import the class
whose declaration we have just read.

>>> from examples.record_review import ExponentialReview
>>> definition = ExponentialReview.definition_report()
>>> definition.explain("value/strictly_increasing").verdict
'conditionally true'
>>> definition.explain("value/concave").verdict
'strictly true'
>>> definition.explain("value/continuously_differentiable").verdict
'strictly true'

A verdict concerns the stated domain and the available definitions, data,
and assumptions. ``strictly true`` means established. ``conditionally true``
retains a sufficient parameter condition; we will inspect it next.
``strictly false`` means the checker established a failure. ``unresolved``
means the attempt established neither the claim nor its failure.

Find the parameter condition
-----------------------------

A ``VerificationContext`` holds the definitions, observations, and assumptions
for one branch of the investigation. Its methods return new contexts, leaving
earlier branches available for comparison. Inspect the parameter condition
using the symbols that belong to this context, then add it as an assumption:

>>> from modello.contrib.verification import VerificationContext
>>> curve = VerificationContext.for_model(ExponentialReview)
>>> increase = curve.explain("value/strictly_increasing")
>>> increase.condition == (curve.symbols.rate > 0)
True
>>> positive_rate = curve.assume(curve.symbols.rate > 0)
>>> positive_rate.explain("value/strictly_increasing").verdict
'strictly true'

An assumption restricts allowed parameter values; an observation fixes a value
for a particular branch. SymPy's ``Rational(1, 2)`` keeps the observed rate as
an exact fraction:

>>> from sympy import Rational
>>> curve.observe(rate=Rational(1, 2)).explain("value/strictly_increasing").verdict
'strictly true'

With zero effectiveness, the curve is constant and the strict increase claim
fails:

>>> curve.observe(rate=0).explain("value/strictly_increasing").verdict
'strictly false'
>>> curve.explain("value/strictly_increasing").verdict
'conditionally true'

The earlier context is unchanged. Observations accumulate; to compare different
values, branch from the common context before either value was supplied. The
reference covers :ref:`removing assumptions <verification-retire-premises>`
when you need to reopen an earlier choice.

Compose the decision model
--------------------------

Now put the response inside the workload model. ``component(ExponentialReview)``
creates a nested response whose fields are available as ``review.value`` and
``review.rate``. The fields ``entry_minutes`` and ``avoidable_minutes`` represent
:math:`b` and :math:`a`. ``candidate`` is a proposed checking time, separate from
the ``effort`` coordinate over which we compare all possible times.

``effort = ExponentialReview.effort`` reuses the response's coordinate, so its
benefit and the total workload vary with the same :math:`t`.

We also record the derivative of net saving and evaluate it at the candidate.
SymPy's ``Derivative(..., evaluate=False)`` keeps differentiation deferred until
Modello has resolved the named quantity to its formula. Without that delay,
it would differentiate the field's placeholder symbol. ``Subs`` substitutes
the candidate for the effort coordinate:

.. code-block:: python

   from sympy import Derivative, Subs
   from modello import component

.. literalinclude:: ../../../examples/record_review.py
   :language: python
   :pyobject: ReviewPlanning

Writing the derivative generates a differentiability obligation automatically.
The C² claim declared on the response is stronger: it also asks for continuous
first and second derivatives, for our later quadratic approximation.

The example's ``review_context()`` helper supplies our inputs and registers the
optimality argument explained below. It leaves the candidate open.
This helper belongs to the example; :ref:`its implementation
<verification-example-contract>` shows how to prepare the same context yourself.
``expression(path)`` returns a field's resolved SymPy expression, so we can
inspect the inputs:

>>> from sympy import log
>>> from examples.record_review import review_context
>>> measured = review_context()
>>> measured.expression("entry_minutes"), measured.expression("avoidable_minutes")
(4, 6)
>>> measured.expression("review.rate")
1/2
>>> measured.explain("review.value/strictly_increasing").verdict
'strictly true'

The nested response keeps its claims with the prefix ``review.value``.

Propose a checking time
-----------------------

With our inputs,

.. math::

   G(t)=6(1-e^{-t/2})-t,\qquad G'(t)=3e^{-t/2}-1.

Setting :math:`G'(t)=0` gives the candidate :math:`t_*=2\log3`, about
2.20 minutes, where :math:`\log` denotes the natural logarithm. A point with
zero derivative is called a stationary point.
We can confirm this exactly. ``doit()`` carries out the deferred derivative
and substitution in ``candidate_marginal``:

>>> proposed = measured.observe(candidate=2*log(3))
>>> proposed.expression("candidate_marginal").doit()
0

Stationarity alone does not establish an optimum. A stationary point might
be a minimum, and a numerical grid might miss a better point.

.. _why-this-is-a-global-optimum:

Explain why it is a global optimum
----------------------------------

Let :math:`s\geq0` denote the proposed checking time and :math:`y\geq0` any
other checking time. For a differentiable concave function, the tangent
line at :math:`s` lies above its graph:

.. math::

   G(y)\leq G(s)+G'(s)(y-s).

If :math:`G'(s)=0`, this becomes
:math:`G(y)\leq G(s)` for every :math:`y\geq0`. Thus a nonnegative stationary
candidate maximises net saving over the whole effort domain. Subtracting the
linear checking cost preserves the concavity of the response benefit.

The example provides ``review_plan``, a ``Method``: a Python computation
paired with its **preconditions**, the claims to establish before it runs.
It needs the response's declared properties, a global maximum, and a positive
workload for the capacity calculation.

``review_context()`` registered the stationary-point implication as a trusted
inference rule. The checker establishes its premises for our particular curve
and candidate; it does not prove the general implication itself.
``stationarity`` means a zero derivative at the candidate; ``feasibility``
means a nonnegative candidate. Run the method, then inspect the four premises:

>>> from examples.record_review import review_plan
>>> result = proposed.run(review_plan)
>>> result.report.explain("global_maximum").verdict
'strictly true'
>>> for premise in result.report.explain("global_maximum").premises:
...     print(premise.obligation.name, premise.verdict)
concavity strictly true
differentiability strictly true
stationarity strictly true
feasibility strictly true

The global-maximum result retains these premise results so we can inspect a
failed argument too.

``result.data`` holds the exact calculation; ``result.report`` retains the
evidence that permitted it. The implementation of the method and rule is
shown in :ref:`the contract reference <verification-example-contract>`.

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

The workload curvature is :math:`M''(t_*)=1/2`. Also :math:`M'(t_*)=0`, since
:math:`M=b+a-G` and the candidate has :math:`G'(t_*)=0`. C² regularity lets us
use Taylor's formula for a small change :math:`h`. The linear term vanishes,
and the quadratic coefficient is :math:`M''(t_*)/2=1/4`:

.. math::

   M(t_*+h)\approx M(t_*)+\tfrac14h^2.

Small changes around the optimum therefore have only a second-order workload
cost. This is a local approximation, while the concavity argument establishes
global optimality. Expected workload capacity also remains an approximation;
it is not a guarantee of how many records finish in a particular day.

Change the repair-time estimate
-------------------------------

Suppose later repairs take thirty staff minutes instead of sixty. With one
record in ten needing repair, the avoidable work becomes :math:`a=3` minutes
per record. The new marginal saving is :math:`G'(t)=\tfrac32e^{-t/2}-1`, giving
the candidate :math:`t_*=2\log(3/2)`.

``review_context`` accepts the entry time, avoidable repair work, and rate as
keyword arguments. Each call starts a fresh context with those inputs, so the
original observations remain available for comparison:

>>> revised = review_context(avoidable_minutes=3)
>>> revised = revised.observe(candidate=2*log(Rational(3, 2)))
>>> revised_result = revised.run(review_plan)
>>> round(float(revised_result.data.effort), 2)
0.81
>>> revised_result.report.explain("global_maximum").verdict
'strictly true'
>>> measured.expression("avoidable_minutes")
6

Cheaper repairs make less checking worthwhile. The same method establishes
its preconditions for the revised inputs.

Try a different response without changing the argument
------------------------------------------------------

Return to ``measured``, with six avoidable repair minutes per record. The
exponential curve was a modelling choice. Suppose we instead propose

.. math::

   u(t)=\frac{kt}{1+kt}.

This response has the same initial slope :math:`k`, but saturates more slowly.
``RationalReview`` inherits the same property declarations. Replacing the
component rechecks them against the new formula and preserves the observed
rate. This is the replacement declaration:

.. literalinclude:: ../../../examples/record_review.py
   :language: python
   :pyobject: RationalReview

With the same inputs, the new marginal saving is
:math:`G'(t)=3/(1+t/2)^2-1`. Solving :math:`G'(t)=0` on :math:`t\geq0`
gives :math:`t_*=2(\sqrt{3}-1)`. Start from ``measured`` so the candidate
has not yet been fixed:

>>> from sympy import sqrt
>>> from examples.record_review import RationalReview
>>> changed = measured.replace("review", RationalReview)
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

``CappedReview`` expresses the two parts with SymPy's ``Piecewise``. The first
branch applies while :math:`kt<1`; the final ``True`` branch covers the rest.
It inherits the original claims too:

.. literalinclude:: ../../../examples/record_review.py
   :language: python
   :pyobject: CappedReview

A failed claim is retained for inspection, so we can define this model and
find out why our method cannot use it:

>>> from examples.record_review import CappedReview
>>> from modello.contrib.verification import VerificationError
>>> capped = measured.replace("review", CappedReview).observe(candidate=2)
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
Adopting it would require a response specification that permits this shape,
as well as an appropriate method.

Evaluate a few points to explore the proposed shape:

>>> sampled = capped.sample("review.value", wrt="effort", points=(1, 2, 3))
>>> tuple(float(value) for value in sampled.values)
(0.5, 1.0, 1.0)
>>> sampled.kind
'numerical observation'

A ``Sample`` records numerical evaluations. A few points cannot establish a
global property or regularity at a join, so they cannot satisfy the method's
failed premises.

.. raw:: html

   <figure>
     <img src="study/responses.svg" alt="Net saving for the exponential and rational curves peaks smoothly. The capped curve reaches its peak at a corner at two minutes." style="max-width:100%;height:auto">
     <figcaption>Changing the response changes the best checking time.
     A curve can have a useful maximum even when our smooth method cannot establish it.</figcaption>
   </figure>

Separate a policy target from a mathematical premise
----------------------------------------------------

Return to the exponential plan in ``proposed``. Suppose the team hopes to save
at least two minutes per record. That target is a goal to check, rather than
an assumption we can use to justify the plan. ``predicate`` turns the
inequality into an obligation, and ``consumer="review_plan"`` makes the method
respect it:

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

These are descriptive exports, not executable saved contexts. To export the
comparisons of the three responses with the original inputs, including the
blocked capped response, run the module:

.. code-block:: console

   $ python -m examples.record_review --output artefacts/record-review

The output contains ``results.json`` with inputs, package versions, and exact
and decimal results, plus a JSON audit and an offline HTML report for each
response. To draw both figures as well, install the optional
``modello[numerics]`` extra and add ``--plot``. The documentation build publishes
the `results <study/results.json>`_, `baseline evidence <study/exponential.html>`_,
`alternative evidence <study/rational.html>`_, and
`blocked replacement <study/capped.html>`_.

For a further experiment, try ``review_context(rate=Rational(1, 3))``. For
positive :math:`a,k` with :math:`ak>1`, solving :math:`G'(t)=0` gives the
positive candidate :math:`t_*=\log(ak)/k`. Check its premises before trusting
the new capacity.

If :math:`ak\leq1`, the best choice is zero checking time. This method checks
ordinary two-sided derivatives. The nonnegative effort domain supplies only
one side at zero, so the endpoint check stays unresolved even when
:math:`ak=1` gives a formal derivative of zero. A boundary argument using the
right-hand behavior is needed for this case.

Continue with :ref:`writing the proof contract <verification-example-contract>`
when you want to adapt the method, or :ref:`checked bounds
<verification-bound-hints>` when a proof needs an extra argument. The
:doc:`verification reference </contrib/verification>` covers the detailed
binding, replacement, domain, and execution rules.
