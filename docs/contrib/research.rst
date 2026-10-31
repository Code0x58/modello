Research with requirements and evidence
=======================================

A model change can preserve a useful formula while invalidating the method that
uses it. A new utility function may need a different reduction; an execution
law may retain convexity while losing a closed-form optimizer.
``modello.contrib.research`` helps you keep those questions explicit: declare
the properties a computation needs, branch the model or its assumptions, and
inspect the evidence before running the computation.

This guide is for researchers who use Python and SymPy to express models.
:doc:`/getting-started` introduces Modello's quantities, components, and
observations. The symbolic workflow below uses the regular ``modello``
installation; numerical and plotting dependencies are needed for the later
worked studies.

Work through one small loop first: declare a requirement, inspect its verdict,
add an assumption or observation, run a method, and export its evidence.
Then revise a premise or try a proof hint before following a :doc:`worked model
<../examples/research-models/index>`. Detailed replacement and proof-domain
rules follow the walkthrough as reference material.

Declare a function and its requirements
---------------------------------------

Suppose a valuation needs a smooth utility function that prefers more
wealth. Keep risk aversion as an unrestricted real parameter so you can ask
which values make the method applicable. ``ResearchModel`` checks each
declaration, and ``requires`` attaches the properties the valuation needs:

>>> from sympy import exp
>>> from modello import Coordinate, InstanceDummy
>>> from modello.contrib.research import (
...     ResearchCase, ResearchModel, increasing, positive, requires, smooth,
... )
>>> class Utility(ResearchModel):
...     wealth = Coordinate("w", real=True)
...     gamma = InstanceDummy("gamma", real=True)
...     value = requires(
...         -exp(-gamma*wealth),
...         increasing(
...             "value", wrt="wealth",
...             reason="More wealth must be preferred.",
...         ),
...         smooth(
...             "value", wrt="wealth", order=2,
...             reason="This calculation uses C2 regularity.",
...         ),
...     )
>>> Utility.definition_report().explain("value/increasing").verdict
'conditionally true'

Each property function returns a ``Requirement``. The ``wrt`` path selects
the variable; ``reason`` explains why the computation needs that property.
Here ``smooth(order=2)`` requires continuous first and second derivatives,
the usual C2 condition.

Interpret the first verdict
---------------------------

The utility's increase is conditional because ``gamma`` is still unrestricted.
Read a verdict in the context of its coordinate domain and current assumptions:

``strictly true``
   The property is established under those premises.

``conditionally true``
   ``condition`` gives a sufficient parameter condition. It need not be the
   only condition under which the property holds.

``strictly false``
   A checked refutation establishes that the stated property fails.

``unresolved``
   The checker has an open obligation: for example, an unsupported operation,
   a missing proof, an exhausted attempt budget, or an inapplicable theorem.

Definition checks keep all four outcomes available for exploration. A method
can run once the premises it consumes are established. The next step is to
make the utility's parameter condition explicit on a case.

Branch assumptions and observations
-----------------------------------

A ``ResearchCase`` combines a declaration with assumptions and observations.
Create an early case as a common branch point, then use its actual symbols
when stating assumptions:

>>> case = ResearchCase.for_model(Utility)
>>> needed = case.explain("value/increasing")
>>> needed.condition == (case.symbols.gamma > 0)
True
>>> positive_case = case.assume(case.symbols.gamma > 0)
>>> positive_case.check().explain("value/increasing").verdict
'strictly true'
>>> case.observe(gamma=0).check().explain("value/increasing").verdict
'strictly false'
>>> case.check().explain("value/increasing").verdict
'conditionally true'

``observe`` accepts the same scalar and component observations as model
inspection. It retains the original equations and input provenance through the
existing solution-family machinery. Observations accumulate; use a new branch
from the earlier case to change one. Exact research uses exact values, such as
``Rational("0.1")``; numerical tolerances belong to a numerical method's settings.

``ResearchCase.for_model`` also accepts an ordinary ``Modello`` class, which
then receives checks when explicitly inspected.

Run a method with established premises
--------------------------------------

A ``Method`` names a computation and the field paths it reads. Its ``uses``
paths connect it to the requirements on those fields. This symbolic valuation
reads the utility curve from whichever case is supplied:

>>> from modello.contrib.research import Method, ResearchError
>>> valuation = Method(
...     "symbolic valuation",
...     lambda current: current.expression("value", wrt="wealth"),
...     uses=("value",),
... )
>>> valued = positive_case.observe(gamma=1).run(valuation)
>>> valued.data.subs(Utility.wealth, 0)
-1
>>> valued.report.explain("value/increasing").verdict
'strictly true'

The result's ``data`` is a SymPy utility curve; here it is evaluated at zero
wealth. Its ``report`` retains the premises that permitted the computation.
Running the same method on the unrestricted case exposes the premise still
needed:

>>> try:
...     case.run(valuation)
... except ResearchError as error:
...     print(error.report.explain("value/increasing").verdict)
conditionally true

``ResearchError`` retains both the case and its report. Methods check their
explicit requirements and the relevant declaration and automatic-operation
requirements before executing. Unrelated open claims remain available in the
report. Case-specific method factories and trusted theorem rules are described
under :ref:`research-method-reference`.

Inspect and export evidence
---------------------------

Use ``valued.report.explain(id)`` to inspect the premises checked for that
computation. ``case.explain(id)`` or ``case.check().explain(id)`` instead checks
the case with its own observations and assumptions.
Evidence includes the claim, coordinate domain, condition and its strength,
derivation steps, dependencies, counterexamples, and recursively checked
theorem premises. Expected forms, target bounds, and checked hints retain their
specialized values in the audit and its fingerprint. ``to_json()`` and
``to_html()`` return text strings that you can save to files. Export the
successful computation's report:

>>> audit_json = valued.report.to_json()
>>> html_page = valued.report.to_html(title="Utility valuation")
>>> import json
>>> json.loads(audit_json)["assumptions_consistent"]
True

Exports retain the declared subject and the expected formula before parameter
observations, alongside the specialized expression actually checked. This
keeps a successful equality readable even when both evaluated sides become
the same number. The JSON remains descriptive; it cannot reconstruct a case
or execute a stored formula.

Revise an assumption or requirement
-----------------------------------

Explicit assumptions can be removed on a new branch, with a recorded reason:

>>> relaxed = positive_case.without_assumptions(
...     case.symbols.gamma > 0, reason="Compare zero and negative risk aversion.")
>>> relaxed.check().explain("value/increasing").verdict
'conditionally true'
>>> positive_case.check().explain("value/increasing").verdict
'strictly true'

The report records why an explicit premise changed. Declared symbol domains
and existing observations remain part of the case.

An exploratory requirement can also be added and retired on a branch:

>>> exploratory = case.require(positive("gamma", name="positive_risk"))
>>> exploratory.check().explain("positive_risk").verdict
'conditionally true'
>>> restored = exploratory.without_requirements(
...     "positive_risk", reason="Compare using only the declared utility requirements.")
>>> len(exploratory.requirements), len(restored.requirements)
(1, 0)
>>> restored.check().explain("value/increasing").verdict
'conditionally true'

Retiring this case-added claim leaves the utility's declared requirements in
place. A method still checks the premises its computation needs.

Check consistency when combining assumptions. Contradictory premises produce
an inconsistent report and cannot establish a result:

>>> impossible = case.assume(case.symbols.gamma > 0, case.symbols.gamma < 0).check()
>>> impossible.consistent, impossible.checks[0].verdict
(False, 'unresolved')

Check a proposed proof
----------------------

An unresolved result can call for a mathematical argument rather than a new
parameter value. A logistic trade-arrival intensity has a curvature-bound
requirement that illustrates this distinction:

>>> from examples.market_making_study import LogisticIntensity
>>> logistic = ResearchCase.for_model(LogisticIntensity)
>>> logistic.check().explain("rate/curvature_bound").verdict
'unresolved'
>>> bounded = logistic.with_hint("rate/curvature_bound", upper=1)
>>> bounded.check().explain("rate/curvature_bound").verdict
'strictly true'

For :math:`\Lambda(\delta)=A/(1+e^{k\delta})`, with :math:`A,k>0`, the checker
derives :math:`\Lambda\Lambda''/(\Lambda')^2=1-e^{-k\delta}`. It verifies the
proposed uniform upper bound 1 and that :math:`1<2`. The nonzero derivative
premise is checked before cancellation. A pointwise inequality strictly below
2 would not by itself establish a supremum strictly below 2.

An incorrect hint leaves the requirement unresolved; it does not refute the property.
The ``upper`` value is a proposed proof aid, not an assumption or a requirement
that is accepted without checking. A uniform target bound must also be
independent of the quantified coordinate, including after resolving aliases.
A coordinate-dependent target is a ``ModelDefinitionError``; it cannot become
a uniform bound by assuming a predicate about that coordinate.
Parameter poles are also retained. For example, regularity of :math:`e^{x/a}`
cannot be unconditional when the declaration allows :math:`a=0`.

Continue with a worked model
----------------------------

Start with :doc:`Markowitz allocation
<../examples/research-models/markowitz>` to compare economic restrictions on
one scalar decision. The :doc:`model collection
<../examples/research-models/index>` then develops component replacement,
deterministic schedules, stochastic premises, and piecewise regularity.

For the complete utility/risk/liquidity application, continue with
:doc:`market making <../examples/market-making/index>`. It connects checked
component requirements to the coupled time-dependent numerical study and its
offline reports.

Reference: contracts, domains, and proof limits
-----------------------------------------------

Use these rules when a replacement or proof attempt needs closer inspection.
The :ref:`API reference <research-api>` below lists the workflow objects and
property functions.

Model and component contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Requirements use relative paths. Nested components acquire their parent path
when inspected; replacing a component or redefining an inherited formula
rebinds its requirements. The stored annotations describe claims, not old
verdicts. Definition checking does not construct a model or select a solver
candidate. Failed and unresolved requirements remain available for exploration.

Every property function accepts optional ``consumer``, ``reason``, and
``source`` metadata. ``reason`` and ``source`` describe the claim's purpose and
provenance. ``consumer`` can associate a claim with a method name or field path
for execution. Methods also consume the requirements declared on the fields
in ``Method.uses`` and their dependencies. These labels leave the mathematical
property being checked unchanged. The same ``Requirement`` works in
``case.require(...)``, ``Method.requirements``, and ``Theorem.premises``.

``Constrained = requires(Model, *requirements)`` returns a subclass carrying
additional model-wide conditions. For a ``ResearchModel`` subclass, it also
refreshes the definition report automatically. For an ordinary ``Modello``
class, inspect the conditions with ``ResearchCase.for_model(Constrained).check()``.
The original model and earlier cases remain unchanged. These conditions follow
the model into nested components and are consumed by methods that read its fields.
Coordinate paths follow inheritance: a subclass that shadows a coordinate with
a regular attribute removes that path from the research interface. Scalar
fields declared at the same path remain available as model quantities.

Native expressions in declaration annotations are rebound in their component's
scope. Paths are useful for contracts that should follow a named formula
through replacement. Equivalent component coordinates are matched by their
relative attribute paths; ambiguous joins of changed and unchanged coordinates
require an explicit new declaration. Coordinate rebinding includes native
derivative and reduction variables, even when they do not occur freely.
Keep component field references in reusable native claims and hints: expanded
expressions cannot identify which component owns a shared coordinate, so
replacement rejects that ambiguity.
Replacement cannot split previously aliased fields or coordinate paths into
independent quantities within the same component. Start a new model and case
to give them separate observations, argument scopes, and contracts.
Named symbolic declarations, such as series and recurrence laws,
follow the same path-based rebinding as fields in observations, claims, and
hints. Recorded declaration paths and argument signatures must remain
compatible; removing or changing that interface requires a new model and case.
Local argument names in independent declarations can change independently.
Coordinate paths also retain their argument role during replacement. Changing
one into an observed or derived model field requires a new model and case;
replacement rejects that change before it can alter an inherited derivative.
Coordinates used by inherited calculations, observations, assumptions, claims,
or hints must remain available through a compatible path or coordinate alias.
Callable expected formulas are evaluated when a claim is checked. Resolve
coordinates through their reference callback so the formula follows the current
component interface. A callback that captures a coordinate from a changed
interface, or a scalar symbol from an earlier case, must be rebuilt before it
can establish a claim. Fresh local variables in native operations remain valid.

Replacement preserves explicit observations. Inferred values copied from an
affected constructor snapshot are invalidated and listed in the report, so
its equations can be solved again in either direction. Inferred values whose
definitions depend on the replaced component are invalidated too. Unrelated
constructor snapshots retain their inferred observations. An old calculated
intensity therefore cannot constrain its replacement.
Dependencies include named laws and composed series queries, even when they
have no free scalar parameters. Methods also consume the component contracts
reached through those declarations. Selecting one recurrence explicitly does
not consume an alternative law merely because it targets the same series.
Equivalent declarations with renamed local variables share their contracts;
ambiguous series queries remain unresolved until a law is selected explicitly.

Independent coordinates with the same name receive distinct scoped labels in
exports, such as ``a.x`` and ``b.x``. Deliberately shared coordinates retain one
identity even when reached through several component paths. Report printers
label the original symbolic tree without substituting or simplifying it, so
formatting cannot turn a difference of independent coordinates into zero.

Retire explicit assumptions and requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``without_assumptions`` removes only explicitly recorded predicates, matching
their retained native scopes. Mathematically equal substitutions with different
coordinate domains remain separate assumptions. To change an intrinsic symbol
domain or an observation, change the declaration or branch before that
observation. ``without_requirements(*ids, reason=...)`` removes
requirements added with ``case.require`` and records their retirement in the
report. Declaration annotations and automatic operation obligations are changed
at their source. Removing an exploratory requirement never removes a method's
own contract; the method checks that requirement again when it runs. Identical
requirement records shared by a case and a method reuse one check, while
conflicting uses of an ID remain definition errors.
The same policy applies to theorem premises: identical requirements reuse one
check, while conflicting conditions need distinct ``name`` values.
All evidence IDs, including nested theorem premise paths, must be unique within
a report; a collision is a definition error.

Coordinates and observation domains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fully bound native values, including reductions, retain the model field's
declared domain. ``Subs(log(x), x, 0)`` leaves no admissible finite complex value
and blocks execution, while substitution at ``-1`` preserves the finite complex
result. This admissibility check does not replace the native operation's own
premises, such as differentiability at an evaluation point.

A quantified curve coordinate is protected from scalar observations. Observing
inventory at 3 specializes a state, while a convexity claim about the risk law
still quantifies inventory over its explicit domain. A claim's remaining free
symbols are parameters. Conditions on them are retained rather than silently
assumed.

``wrt`` selects an independent model quantity or a declared ``Coordinate``.
Use a path or the original symbol identity. A new same-named SymPy symbol is
rejected as foreign to the case. A defined quantity such as ``doubled = 2*x``
is rejected as a differentiation coordinate with ``ModelDefinitionError``:
reparameterize the curve explicitly before asking about its derivatives.
Quantities in unresolved defining cycles, and quantities that depend on them,
also need reparameterization before serving as coordinates.
``expression`` and ``sample`` apply the same coordinate validation as property
checks. Parameter observations, including cyclic observations, do not turn an
independent quantity into a defined coordinate.

A curve through unresolved defining cycles also needs reparameterization when
its sources depend on the selected coordinate, including when observation
equations refer to that curve indirectly. Property checks retain unresolved
evidence; ``expression`` and ``sample`` raise ``NotImplementedError``. Point
queries without ``wrt`` can still resolve the joint equations.

Property truth and proof attempts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Boolean predicates also require defined scalar operands. Negation, disjunction,
and implication do not turn a comparison with an undefined operand into a proof.
For example, ``predicate(Eq(1/a, 1/a, evaluate=False))`` retains ``Ne(a, 0)``;
observing ``a=0`` cannot establish it. Explicit and active SymPy assumptions
retain these domains too. Unsupported comparisons remain inspectable and block
methods that need them.

Scalar declarations retain their operands before simplification too. A held
``0*log(x)`` or ``exp(log(x))`` therefore keeps the hole at zero when inspected,
checked, or sampled. Native operations in assumptions use the same premise
checks as property subjects. A refuted native premise makes the case
inconsistent; an unestablished premise blocks methods until supported
parameter facts establish it. The native assumption cannot establish its own
operation's premises.
If upstream CAS source binding, proof, or assumption-consistency checks exhaust
their recursion limit, the report keeps an unresolved obligation and blocks
dependent methods. A branch selector with unsupported source binding also
leaves an inspectable report, including during ``ResearchModel`` declaration.

Checks also bind active SymPy ``assuming(...)`` predicates through observations
and record them in their report, checking the combined premises for consistency.
Cached expression bindings, solution families, and proof attempts remain scoped
to that global assumption context, including its retained native coordinate
scopes. Case-bound methods distinguish those scopes too. Leaving the context
checks the original branch again without retaining context-dependent
simplifications. Use explicit ``case.assume(...)`` branches for premises that
should persist in the study.

Native substitution coordinates keep their own declared domains in cached
bindings, proof attempts, and theorem matching. SymPy can consider two ``Subs``
expressions mathematically equal after renaming their bound variables; that
equality does not transfer a derivative certificate between their scopes.
Coordinate discovery for component replacement visits every retained scope,
including mathematically equal substitutions with different bound coordinates.
For example, evaluation at zero on a real coordinate does not establish the
same operation on a positive coordinate whose domain excludes zero.
Binding and replacement also retain pending native operations before checking
their premises. If a CAS constructor drops an operation or substitutes a stale
coordinate scope, ``check()`` reports unresolved evidence and ``run()`` blocks
methods that need it. ``expression()``, ``sample()``, and replacement raise
``ModelInputError`` when that scope cannot be preserved; declare the expression
with its new coordinates instead.

The positive-derivative test is sufficient for strict increase, but it is not
its definition. The polynomial rule can establish strict increase of
:math:`x^3` despite its zero derivative at zero. Likewise, :math:`|x|` is convex
and fails differentiability at zero. Failed C2 certificates alone do not refute
convexity or twice differentiability. Discontinuity alone does not refute
positivity, nonzero values, or an identity. Tail-limit claims inspect eventual
behavior rather than requiring regularity at every finite point. Their
requested domain, after applying source guards, must be established unbounded
above. A bounded domain or a domain extending only toward negative infinity
cannot establish a right-tail certificate.

For a parameterized polynomial such as :math:`a x^3`, that rule retains the
nonconstancy condition. Assuming only :math:`a\geq0` cannot establish strict
increase because :math:`a=0` remains admissible.

Continuity-domain failures can also depend on parameters. For example,
:math:`a\sqrt{x}` on :math:`[0,1]` has a smooth zero specialization at
:math:`a=0`. Its general smoothness check remains unresolved; observing
:math:`a=1` establishes failure. A CAS continuity-domain exclusion alone does
not refute every parameter specialization. Derivative formulas can also have
removable singularities. Such exclusions remain unresolved unless a checked
one-sided divergence or another supported rule establishes a refutation.

The curvature characterization of convexity applies on real intervals with
nonempty interior. A singleton is checked by its finite real value instead;
other discrete or disconnected domains remain unsupported by that rule.

Requirement domains must be established to be nonempty. Empty domains, including
reversed intervals after observing their endpoints, remain unresolved with an
explicit diagnostic. A symbolic interval with unknown endpoint order also stays
unresolved. This policy applies before trusted theorem application and prevents
an empty scope from certifying a computation or supplying a refutation.

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
Empty sums, products, and zero-width integrals skip their inactive bodies, but
their endpoint operations still need established premises. An undefined endpoint
derivative cannot certify an identity merely because the resulting range is empty.
For a supported single-limit scope, an inactive body can contain an unknown
function or a cyclic model definition without blocking the empty operation.

Parameter observations also retain the source operands' definedness. For
example, setting ``a=0`` in ``a*log(x)`` does not define its value at ``x=0``.
Symbolic observations retain their native operations too: supplying an
undefined derivative as an observation cannot establish a method's premises,
even if a later multiplication would erase its value. Automatic obligations
follow these observations as well as model definitions.
Joint solving retains the source operands of observations and implicit defining
equations for checking and sampling. Evidence and exported reports expose them
as ``source_operands``, alongside the resolved expression.
Supported source domains are checked alongside the simplified formula;
unsupported operand domains remain unresolved. These checks keep local
substitutions and derivative evaluation points in their original scopes.
Real continuity domains provide sufficient definedness and regularity
certificates. Their failure alone does not refute claims involving finite
complex intermediate values, such as ``0*log(x)`` or ``cos(sqrt(x))`` on a
negative interval. Unsupported source domains remain
unresolved even when point samples happen to be finite.
Native SymPy substitutions retain these domains. Numerical compilation of
unsupported operand guards raises ``NotImplementedError`` instead of using
backend-specific values to fill undefined points.

Automatic obligations use a coordinate's declared domain: a positive
coordinate is checked on :math:`(0,\infty)`, while a nonnegative coordinate
still includes zero and retains a denominator failure there. Explicit claims
keep their own requested domains, which must lie inside the intrinsic domain.
The same domain rules as solution families retain integer, rational, parity,
zero, and other declared restrictions. A discrete coordinate is not silently
treated as a real continuum; unsupported proof scopes remain unresolved.
Sampling checks membership using those same restrictions.
Ordinary derivative certificates require an established real interior in the
coordinate's intrinsic domain. Global regularity claims and native derivatives
on integer or rational coordinates therefore remain unresolved. Value checks
and sampling retain their declared restrictions. A real coordinate can still
be differentiated at a discrete set of evaluation points, and supported closed
real intervals retain their regularity certificates.

Native ``Subs`` and ``Derivative`` expressions on either side of an identity,
in target bounds, evaluation points, domains, and proposed hints are evaluated
after checking their own derivative scopes. Their local evaluation points
need an established real neighborhood inside the derivative coordinate's domain.
Undefined evaluation points remain failures. An invalid optional hint leaves
the attempted proof unresolved; it
does not refute the requested bound.
Native derivative premises in ``Piecewise`` expressions respect the active
branch's domain. Missing regularity proofs remain unresolved, including when
the operation occurs in a conditional branch.
Automatic obligations skip branches excluded by established parameter choices;
a global certificate also establishes an operation whose activation is unknown.
Otherwise the obligation remains unresolved. Coordinate assumptions cannot
select away branches of a globally quantified automatic obligation.

Native reductions retain their bound variables and supported integration or
summation ranges. Assumptions on a free parameter cannot certify a locally bound
variable with the same name. Single definite real integration intervals and
integer reduction ranges have explicit scopes; unsupported nested ranges remain
unresolved. These checks establish the operations used in an integrand or term;
they do not establish convergence of an integral, series, or limit.

Implicit defining equations and intrinsic field domains participate in joint
feasibility checks even without observations. Inconsistent cases block method execution.
Unsupported cyclic calculus remains inspectable as unresolved evidence, and
blocks computations that depend on the unavailable model algebra.
Unsupported inequality reasoning, including the solver's limits on irrational
coefficients, leaves consistency or evidence conditions unresolved. The report
retains the assumptions, and a method still needs its required evidence before
it can execute.

A nonzero real coordinate keeps zero excluded even when its sign is unknown.
An inner derivative cannot borrow an unrelated outer coordinate's domain.
For a supported two-branch piecewise curve with one constant interior join,
the checker compares actual assigned values and finite one-sided derivatives
through order two. A local differentiability check away from the join checks
the active branch's neighborhood and source guards; matching the other branch
at the join is unnecessary. Unsupported joins stay unresolved; an included
interval endpoint with only one active side does not establish ordinary differentiability.
Pointwise definedness at a guarded boundary is insufficient: an ordinary
derivative needs a defined neighborhood around every point, including points
in finite or closed interval domains. This requirement also applies to
``smooth(order=1)`` and ``smooth(order=2)``. Continuity checks remain relative to
the requested domain.

A uniform-bound hint supplies a sufficient certificate. If its parameter guard
fails, the attempted proof is unresolved; a loose hint cannot refute the bound.
Consumed hint keys, including canonical paths and theorem-premise hints, are
excluded from unused-hint diagnostics.

Use deferred ``Derivative(named_quantity, coordinate)`` inside declarations.
A named quantity acts as a symbol while the class body is collected, so eager
``diff`` can treat it as independent of the coordinate. Differentiate a resolved
``case.expression`` when performing a calculation outside the declaration.

.. _research-method-reference:

Methods and theorems
~~~~~~~~~~~~~~~~~~~~

Pass either a prepared method or a method factory to
``case.run(method, **settings)``. A factory receives the current case and returns
a ``Method``; the runner then checks its premises and relevant automatic
obligations before executing it. Settings go to the computation, not the
factory. A blocked computation raises
``ResearchError`` with ``case`` and ``report``; numerical failures retain the
same context. Unrelated unresolved claims do not block a method.

Reading a field through an alias consumes the same automatic obligations as
reading its original path, including aliases exposed from nested components.
Branch selection uses the report's resolved parameter assumptions, including
assumptions expressed through aliases and active SymPy assumptions. Predicates
involving local or quantified coordinates cannot select away their operations.
Piecewise selectors retain their scalar domains before choosing a branch.
Only selectors reached in branch order need to be defined. Their checks keep
local substitution points and the full domains of unevaluated curves separate.
A locally evaluated expression such as ``Subs(curve, x, 1)`` keeps its own
scope: its obligations concern that evaluation, while the unevaluated source
curve can still have a pole elsewhere.

>>> from modello.contrib.research import Method
>>> def valuation(current):
...     curve = current.expression("value", wrt="wealth")
...     return Method(
...         "symbolic valuation", lambda inner: curve, uses=("value",),
...     ).bind(current)
>>> positive_case.run(valuation).report.explain("value/increasing").verdict
'strictly true'

Use ``case.run(valuation)`` after branching: it builds the method from the
current case on every call. The model examples use this form, such as
``case.run(symbolic_method)`` or ``case.run(numerical_method, xatol=1e-10)``.

Factories capturing expressions bind the returned method with
``Method(...).bind(case)``. If a prepared method is retained for later use,
``case.run(prepared)`` rejects a different case or SymPy assumption context
before checking its captured requirements or executing it. Generic methods
that resolve expressions from the supplied ``current`` case can remain unbound.

``Theorem`` registers a trusted rule with a conclusion, named premises, source,
and version. ``with_theorem`` adds it to a case. The engine recursively checks
the premises and retains them in the conclusion's evidence. Failed premises
make a sufficient theorem inapplicable; they do not refute its conclusion.
Circular rules remain unresolved. The theorem implementation and CAS form the
trust boundary; a citation or user assertion alone supplies no proof.

``sample`` evaluates a finite set of points and returns an ``Experiment``.
This record is labelled numerical observation and cannot discharge a global
requirement. Points must be fully bound finite real scalars in the coordinate's
declared domain, and their evaluated values must be fully bound, finite, and
real. Invalid points and unresolved symbolic values raise
``ModelValidationError``.

.. _research-api:

API
----

.. automodule:: modello.contrib.research

.. currentmodule:: modello.contrib.research

Workflow objects
~~~~~~~~~~~~~~~~~

``Requirement`` is the single type for conditions. ``Method`` describes a
guarded computation; ``Theorem`` describes an explicitly trusted implication.

.. autoclass:: Requirement

.. autofunction:: requires

.. autoclass:: ResearchModel
   :members:

.. autoclass:: ResearchCase
   :members:

.. autoclass:: Method
   :members:

.. autoclass:: Theorem

Mathematical conditions
~~~~~~~~~~~~~~~~~~~~~~~

Every function below returns a ``Requirement`` and accepts the same optional
``consumer``, ``reason``, and ``source`` metadata. Sign, order, shape, regularity,
and bounds are different mathematical conditions, rather than different
container types. In particular, ``smooth(order=n)`` requires continuous
derivatives through order n; ``differentiable(order=n)`` only requires those
derivatives to exist.

.. autofunction:: positive

.. autofunction:: nonnegative

.. autofunction:: nonzero

.. autofunction:: increasing

.. autofunction:: decreasing

.. autofunction:: convex

.. autofunction:: concave

.. autofunction:: smooth

.. autofunction:: differentiable

.. autofunction:: vanishes

.. autofunction:: uniform_bound

.. autofunction:: curvature_bound

.. autofunction:: exact_form

.. autofunction:: predicate

Returned evidence and results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These records are returned by the workflow; they do not introduce additional
ways to declare a requirement.

.. autoclass:: Evidence
   :members:

.. autoclass:: ResearchReport
   :members:

.. autoclass:: ResearchResult

.. autoclass:: Experiment

.. autoexception:: ResearchError
