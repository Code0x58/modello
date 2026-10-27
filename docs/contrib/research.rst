Research with requirements and evidence
=======================================

An optimization method consumes more than component values: it may need a
function to be monotone, convex, or differentiable on a particular domain.
``modello.contrib.research`` records those obligations, checks supported scalar
claims, and explains which conclusions depend on them. Numerical observations
remain separate from mathematical evidence. The :doc:`model collection
<../examples/research-models/index>` applies the same workflow to pricing,
execution, and allocation with changing restrictions.

The symbolic API requires only SymPy. Install optional numerical and plotting
dependencies for the complete example:

.. code-block:: console

   $ pip install 'modello[research]'
   $ python -m examples.market_making --study --output market-making

Open ``market-making/report.html``. The directory also contains JSON evidence,
CSV results, separate evidence pages, and SVG/PNG figures. The executable Python
study is the replayable source; report JSON is descriptive and is never loaded
as Python or symbolic expressions.

Declare the consumer's requirements
-----------------------------------

Use ``ResearchModel`` to refresh declaration checks automatically for every
subclass. ``annotate`` attaches requirements without changing the equations:

>>> from sympy import exp, S
>>> from modello import Coordinate, InstanceDummy, annotate
>>> from modello.contrib.research import (
...     ResearchCase, ResearchModel, Requirement, Requires, increasing, smooth,
... )
>>> class Utility(ResearchModel):
...     wealth = Coordinate("w", real=True)
...     gamma = InstanceDummy("gamma", real=True)
...     value = annotate(-exp(-gamma*wealth), Requires(
...         Requirement(increasing("value", wrt="wealth"), "valuation", "More wealth must be preferred."),
...         Requirement(smooth("value", wrt="wealth"), "price_curvature", "This calculation uses C2 regularity."),
...     ))
>>> Utility.definition_report().explain("value/increasing").verdict
'conditionally true'

Requirements use relative paths. Nested components acquire their parent path
when inspected; replacing a component or redefining an inherited formula
rebinds its requirements. The stored annotations describe claims, not old
verdicts. Definition checking does not construct a model or select a solver
candidate. Failed and unresolved requirements remain available for exploration.

Native expressions in declaration annotations are rebound in their component's
scope. Paths are useful for contracts that should follow a named formula
through replacement. Equivalent component coordinates are matched by their
relative attribute paths; ambiguous joins of changed and unchanged coordinates
require an explicit new declaration.

Replacement preserves explicit observations. Copied inferred values whose
definitions depend on the replaced component are invalidated and listed in
the report, so an old calculated intensity cannot constrain its replacement.

``ResearchCase.for_model`` also accepts an ordinary ``Modello`` class, which
then receives checks when explicitly inspected.

Branch assumptions and observations
-----------------------------------

Cases are immutable. Use their actual symbols, preserving identities across
branches:

>>> case = ResearchCase.for_model(Utility)
>>> positive_risk = case.assume(case.symbols.gamma > 0)
>>> positive_risk.check().explain("value/increasing").verdict
'strictly true'
>>> case.observe(gamma=0).check().explain("value/increasing").verdict
'strictly false'
>>> case.check().explain("value/increasing").verdict
'conditionally true'
>>> impossible = case.assume(case.symbols.gamma > 0, case.symbols.gamma < 0).check()
>>> impossible.consistent, impossible.checks[0].verdict
(False, 'unresolved')

``observe`` accepts the same scalar and component observations as model
inspection. It retains the original equations and input provenance through the
existing solution-family machinery. Observations accumulate; use a new branch
from the earlier case to change one. Exact research uses exact values, such as
``Rational("0.1")``; numerical tolerances belong to a numerical method's settings.

A quantified curve coordinate is protected from scalar observations. Observing
inventory at 3 specializes a state, while a convexity claim about the risk law
still quantifies inventory over its explicit domain. A claim's remaining free
symbols are parameters. Conditions on them are retained rather than silently
assumed.

Explicit assumptions can be removed on a new branch, with a recorded reason:

>>> relaxed = positive_risk.without_assumptions(
...     case.symbols.gamma > 0, reason="Compare zero and negative risk aversion.")
>>> relaxed.check().explain("value/increasing").verdict
'conditionally true'
>>> positive_risk.check().explain("value/increasing").verdict
'strictly true'

``without_assumptions`` removes only explicitly recorded predicates. To change
an intrinsic symbol domain or an observation, change the declaration or branch
before that observation. ``without_requirements(*ids, reason=...)`` removes
requirements added with ``case.require`` and records their retirement in the
report. Declaration annotations and automatic operation obligations are changed
at their source. Removing an exploratory requirement never removes a method's
own contract; the method checks that requirement again when it runs. Identical
requirement records shared by a case and a method reuse one check, while
conflicting uses of an ID remain definition errors.

Check a proposed proof
----------------------

The logistic intensity makes a useful research substitution:

>>> from examples.market_making_study import LogisticIntensity
>>> from modello.contrib.research import upper_bound
>>> logistic = ResearchCase.for_model(LogisticIntensity)
>>> logistic.check().explain("rate/curvature_bound").verdict
'unresolved'
>>> bounded = logistic.with_hint("rate/curvature_bound", upper_bound(1))
>>> bounded.check().explain("rate/curvature_bound").verdict
'strictly true'

For :math:`\Lambda(\delta)=A/(1+e^{k\delta})`, with :math:`A,k>0`, the checker
derives :math:`\Lambda\Lambda''/(\Lambda')^2=1-e^{-k\delta}`. It verifies the
proposed uniform upper bound 1 and that :math:`1<2`. The nonzero derivative
premise is checked before cancellation. A pointwise inequality strictly below
2 would not by itself establish a supremum strictly below 2.

An incorrect hint leaves the claim unresolved; it does not refute the property.
Parameter poles are also retained. For example, regularity of :math:`e^{x/a}`
cannot be unconditional when the declaration allows :math:`a=0`.

Read the proof through ``case.explain(id)`` or ``case.check().explain(id)``.
Evidence includes the claim, coordinate domain, condition and its strength,
derivation steps, dependencies, counterexamples, and recursively checked
theorem premises. Expected forms, target bounds, and checked hints retain their
specialized values in the audit and its fingerprint. ``to_json()`` and
``to_html()`` export the report.

Exports retain the declared subject and the expected formula before parameter
observations, alongside the specialized expression actually checked. This
keeps a successful equality readable even when both evaluated sides become
the same number. The JSON remains descriptive; it cannot reconstruct a case
or execute a stored formula.

Property truth and proof attempts
---------------------------------

``strictly true`` means established under the current assumptions.
``conditionally true`` retains a sufficient parameter condition; it does not
claim that condition is necessary unless separately established.
``strictly false`` requires a checked refutation. ``unresolved`` preserves a
missing proof, unsupported operation, exhausted attempt budget, or inapplicable
theorem. Contradictory assumptions do not produce successful proofs.

Checks also bind active SymPy ``assuming(...)`` predicates through observations
and record them in their report, checking the combined premises for consistency.
Cached expression bindings, solution families, and proof attempts remain scoped
to that global assumption context. Leaving the context checks the original
branch again without retaining context-dependent simplifications. Use explicit
``case.assume(...)`` branches for premises that should persist in the study.

The positive-derivative test is sufficient for strict increase, but it is not
its definition. The polynomial rule can establish strict increase of
:math:`x^3` despite its zero derivative at zero. Likewise, :math:`|x|` is convex
and fails differentiability at zero. Failed C2 certificates alone do not refute
convexity or twice differentiability. Discontinuity alone does not refute
positivity, nonzero values, or an identity. Tail-limit claims inspect eventual
behavior rather than requiring regularity at every finite point.

The curvature characterization of convexity applies on real intervals with
nonempty interior. A singleton is checked by its finite real value instead;
other discrete or disconnected domains remain unsupported by that rule.

Supported proofs cover scalar real functions on real intervals, sign and
polynomial arguments, supported continuity domains, limits, and checked
uniform bounds. Unknown functions, mixed derivative regularity, and unsupported
parameter domains can remain unresolved. ``check(max_steps=100)`` bounds proof
attempts, not wall-clock time within a CAS call.

Native derivatives introduce differentiability obligations with their order
and evaluation point. They do not imply convexity or continuity of the
derivatives. Divisions introduce definedness obligations before research-layer
cancellation. Inspection sees the expressions SymPy and Modello retained; it
cannot recover syntax already simplified away before declaration.

Automatic obligations use a coordinate's declared domain: a positive
coordinate is checked on :math:`(0,\infty)`, while a nonnegative coordinate
still includes zero and retains a denominator failure there. Explicit claims
keep their own requested domains. Native ``Subs`` and ``Derivative`` expressions
on either side of an identity are evaluated after checking derivative scopes;
undefined evaluation points remain failures.

A nonzero real coordinate keeps zero excluded even when its sign is unknown.
An inner derivative cannot borrow an unrelated outer coordinate's domain.
For a supported two-branch piecewise curve with one constant interior join,
the checker compares actual assigned values and finite one-sided derivatives
through order two. Unsupported joins stay unresolved; an included interval
endpoint with only one active side does not establish ordinary differentiability.

Use deferred ``Derivative(named_quantity, coordinate)`` inside declarations.
A named quantity acts as a symbol while the class body is collected, so eager
``diff`` can treat it as independent of the coordinate. Differentiate a resolved
``case.expression`` when performing a calculation outside the declaration.

Methods and theorems
--------------------

``Method`` records a computation, its explicit requirements, and the field
paths it reads. ``case.run(method, **settings)`` checks its premises and relevant
automatic obligations before calling it. A blocked computation raises
``ResearchError`` with ``case`` and ``report``; numerical failures retain the
same context. Unrelated unresolved claims do not block a method.

>>> from modello.contrib.research import Method
>>> method = Method("symbolic valuation", lambda current: current.expression("value", wrt="wealth"),
...                 uses=("value",))
>>> positive_risk.run(method).report.explain("value/increasing").verdict
'strictly true'

A method factory that captures ``case.expression(...)`` should return
``Method(...).bind(case)``. Binding retains the entire immutable research
context, including assumptions, hints, registered rules, and the active SymPy
global assumption context. A different branch or global context must rebuild
the factory; sharing the same underlying problem snapshot is not enough.
``case.run`` rejects a stale bound method before checking its captured
requirements or executing it. Generic methods that resolve expressions from
the supplied ``current`` case can remain unbound.

``Theorem`` registers a trusted rule with a conclusion, named premises, source,
and version. ``with_theorem`` adds it to a case. The engine recursively checks
the premises and retains them in the conclusion's evidence. Failed premises
make a sufficient theorem inapplicable; they do not refute its conclusion.
Circular rules remain unresolved. The theorem implementation and CAS form the
trust boundary; a citation or user assertion alone supplies no proof.

``sample`` evaluates a finite set of points and returns an ``Experiment``.
This record is labelled numerical observation and cannot discharge a global
requirement.

Complete the time-dependent study
---------------------------------

>>> from examples.market_making_study import study_case
>>> from examples._market_making_numerics import run_quotes
>>> result = run_quotes(study_case(logistic=True, capacity=1), points=5)
>>> result.data.bid[0][-1] is None, result.data.ask[0][0] is None
(True, True)
>>> result.data.diagnostics["terminal_error"]
0.0

The full runner compares exponential :math:`A=2,k=1` with logistic
:math:`A=4,k=2`, matching rate and slope at zero. Defaults are
:math:`T=1`, :math:`\Delta=1`, :math:`Q=5`, :math:`\gamma=0.1`,
:math:`\sigma=1`, and :math:`\eta=0.05`.

Both sides consume the general Hamiltonian interface. A separate exact-form
contract protects the exponential shortcut; changing intensity does not carry
that formula into the new model. The CARA reduction has its own exact-utility
contract. Additional contracts protect the nearest-neighbor continuation
differences, CARA gain, boundary gates, and HJB structure used by the numerical
adapter. Shape properties alone do not establish these identities.

The solver integrates in remaining time using SciPy Radau with an analytic
tridiagonal Jacobian. The general quote method brackets the unique
first-order-condition root; exponential and logistic expressions use stable
numerical evaluations. Inventory boundary gates come from the shared HJB.
Inactive quotes are absent values in reports. Invalid lattice membership,
nonfinite values, failed bracketing, and solver failures retain diagnostics.
The full ODE requires independent state inputs and parameter observations;
observations or assumptions fixing a state are rejected because the study
traverses the entire inventory and time grid.

Default tolerances are ``rtol=1e-8``, ``atol=1e-10``, and a maximum time step
of ``T/100``. The report compares a tighter run, measures independent central
time-difference defects, and checks the exponential case against its linear
matrix-exponential transformation. Both calculations use the case's declared
risk and terminal laws, including supported component replacements. These are numerical checks rather than
proofs of error bounds. Source fingerprints, exact parameters, solver settings,
and dependency versions accompany the results.

API
----

.. automodule:: modello.contrib.research
   :members: ResearchModel, ResearchCase, Claim, Requirement, Requires, Evidence, ResearchReport, Method, Theorem, ResearchResult, ResearchError, Experiment, ProofHint, upper_bound, positive, nonnegative, nonzero, increasing, decreasing, convex, concave, smooth, differentiable, vanishes, uniform_bound, curvature_bound, exact_form, predicate
   :member-order: bysource
