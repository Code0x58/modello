Deductive verification of model properties
==========================================

A model change can preserve a useful formula while invalidating the method that
uses it. A new utility function may need a different reduction; an execution
law may retain convexity while losing a closed-form optimizer.
``modello.contrib.verification`` checks whether specified properties follow
from a model's definitions, observations, and assumptions. These properties
become **proof obligations**: statements to establish before a computation
that needs them can run.

This guide is for researchers who use Python and SymPy to express models.
:doc:`/getting-started` introduces Modello's quantities, components, and
observations. The symbolic workflow below uses the regular ``modello``
installation; numerical and plotting dependencies are needed for the later
worked studies.

Work through one small loop first: specify a property, inspect its verdict,
add an assumption or observation, run a method, and export its evidence.
Then revise a premise or try a bound hint before following a :doc:`worked model
<../examples/financial-models/index>`. Detailed replacement and proof-domain
rules follow the walkthrough as reference material.

.. _declare-a-function-and-its-requirements:

Specify a function and its properties
-------------------------------------

Suppose a valuation needs a utility function that prefers more wealth and has
continuous first and second derivatives. Keep risk aversion unrestricted so
you can ask which values make the method applicable. ``VerificationModel``
checks each declaration, and ``specifies`` attaches the properties the
valuation needs:

>>> from sympy import exp
>>> from modello import Coordinate, InstanceDummy
>>> from modello.contrib.verification import (
...     VerificationContext, VerificationModel, continuously_differentiable,
...     positive, specifies, strictly_increasing,
... )
>>> class Utility(VerificationModel):
...     wealth = Coordinate("w", real=True)
...     gamma = InstanceDummy("gamma", real=True)
...     value = specifies(
...         -exp(-gamma*wealth),
...         strictly_increasing(
...             "value", wrt="wealth",
...             reason="More wealth must be preferred.",
...         ),
...         continuously_differentiable(
...             "value", wrt="wealth", order=2,
...             reason="This calculation uses C2 regularity.",
...         ),
...     )
>>> Utility.definition_report().explain("value/strictly_increasing").verdict
'conditionally true'

Each property function returns an ``Obligation``. The ``wrt`` path selects
the variable; ``reason`` explains why the computation needs that property.
Here ``continuously_differentiable(order=2)`` requires continuous first and
second derivatives, the usual C2 condition.

In this declaration, ``wealth`` is the curve's argument, represented by a
``Coordinate``. These properties are checked over the real line. ``gamma`` is
a parameter represented by ``InstanceDummy``. An assumption restricts its
allowed values; an observation can fix it for a particular context.

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
   a missing proof, an exhausted attempt budget, or an inapplicable inference rule.

Definition checks keep all four outcomes available for exploration. A method
can run once the premises it consumes are established. The next step is to
make the utility's parameter condition explicit on a context.

Branch assumptions and observations
-----------------------------------

A ``VerificationContext`` combines a model declaration with assumptions and
observations for one branch of a study. Start with a common branch point,
then use its actual symbols when stating assumptions:

>>> context = VerificationContext.for_model(Utility)
>>> needed = context.explain("value/strictly_increasing")
>>> needed.condition == (context.symbols.gamma > 0)
True
>>> positive_context = context.assume(context.symbols.gamma > 0)
>>> positive_context.verify().explain("value/strictly_increasing").verdict
'strictly true'
>>> context.observe(gamma=0).verify().explain("value/strictly_increasing").verdict
'strictly false'
>>> context.verify().explain("value/strictly_increasing").verdict
'conditionally true'

``observe`` accepts the same scalar and component observations as model
inspection. It retains the original equations and input provenance through the
existing solution-family machinery. Observations accumulate; use a new branch
from the earlier context to change one. Use exact values, such as
``Rational("0.1")``, for symbolic checks. Numerical tolerances belong to a
numerical method's settings.

``VerificationContext.for_model`` also accepts an ordinary ``Modello`` class, which
then receives checks when explicitly inspected.

Run a method with established premises
--------------------------------------

A ``Method`` pairs a computation with an explicit contract. Set ``uses`` to
the model fields the computation reads directly; the runner follows their
model dependencies. State extra mathematical premises in ``Method.preconditions``.
This symbolic valuation reads the utility curve from whichever context is
supplied:

>>> from modello.contrib.verification import Method, VerificationError
>>> valuation = Method(
...     "symbolic valuation",
...     lambda current: current.expression("value", wrt="wealth"),
...     uses=("value",),
... )
>>> valued = positive_context.observe(gamma=1).run(valuation)
>>> valued.data.subs(Utility.wealth, 0)
-1
>>> valued.report.explain("value/strictly_increasing").verdict
'strictly true'

The result's ``data`` is a SymPy utility curve; here it is evaluated at zero
wealth. Its ``report`` retains the premises that permitted the computation.
Running the same method on the unrestricted context exposes the premise still
needed:

>>> try:
...     context.run(valuation)
... except VerificationError as error:
...     print(error.report.explain("value/strictly_increasing").verdict)
conditionally true

``VerificationError`` retains the originating context in ``context`` and its
evidence in ``report``. Before calling ``Method.execute``, ``run`` requires
established evidence for the method's explicit preconditions and the
applicable obligations from declarations and symbolic operations. Unrelated
open claims remain available in the report. Context-specific method factories
and trusted inference rules are described under
:ref:`verification-method-reference`.

Inspect and export evidence
---------------------------

Use ``valued.report.explain(id)`` to inspect the premises checked for that
computation. ``context.explain(id)`` or ``context.verify().explain(id)`` instead
verifies the context with its own observations and assumptions.
Evidence includes the claim, coordinate domain, condition and its strength,
derivation steps, dependencies, counterexamples, and recursively checked
rule premises. Expected forms, target bounds, and checked hints retain their
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
the same number. ``VerificationReport.evidence`` holds the verification
attempts; schema version 2 exports them under ``evidence``. Study records use
``context`` or ``contexts`` for their model contexts. The JSON remains
descriptive; it cannot reconstruct a context or execute a stored formula.

.. _revise-an-assumption-or-requirement:

Revise assumptions and obligations
----------------------------------

Explicit assumptions can be removed on a new branch, with a recorded reason:

>>> relaxed = positive_context.without_assumptions(
...     context.symbols.gamma > 0, reason="Compare zero and negative risk aversion.")
>>> relaxed.verify().explain("value/strictly_increasing").verdict
'conditionally true'
>>> positive_context.verify().explain("value/strictly_increasing").verdict
'strictly true'

The report records why an explicit premise changed. Declared symbol domains
and existing observations remain part of the context.

You can add a proof obligation for exploration, then retire it on a new branch:

>>> exploratory = context.with_obligations(positive("gamma", name="positive_risk"))
>>> exploratory.verify().explain("positive_risk").verdict
'conditionally true'
>>> restored = exploratory.without_obligations(
...     "positive_risk", reason="Compare using only the declared utility obligations.")
>>> len(exploratory.obligations), len(restored.obligations)
(1, 0)
>>> restored.verify().explain("value/strictly_increasing").verdict
'conditionally true'

Retiring this added obligation leaves the utility's declared obligations
in place. A method still needs evidence for its preconditions.

Check consistency when combining assumptions. Contradictory premises produce
an inconsistent report and cannot establish a result:

>>> impossible = context.assume(context.symbols.gamma > 0, context.symbols.gamma < 0).verify()
>>> impossible.consistent, impossible.evidence[0].verdict
(False, 'unresolved')

.. _check-a-proposed-proof:

Propose a uniform bound
-----------------------

An unresolved result can call for a mathematical argument rather than a new
parameter value. A logistic trade-arrival intensity has a uniform
derivative-ratio obligation that illustrates this distinction:

>>> from examples.market_making_study import LogisticIntensity
>>> logistic = VerificationContext.for_model(LogisticIntensity)
>>> logistic.verify().explain("rate/derivative_ratio_bound").verdict
'unresolved'
>>> bounded = logistic.with_bound_hint("rate/derivative_ratio_bound", upper=1)
>>> bounded.verify().explain("rate/derivative_ratio_bound").verdict
'strictly true'

For :math:`\Lambda(\delta)=A/(1+e^{k\delta})`, with :math:`A,k>0`, the checker
derives :math:`\Lambda\Lambda''/(\Lambda')^2=1-e^{-k\delta}`. It verifies the
proposed uniform upper bound 1 and that :math:`1<2`. The nonzero derivative
premise is checked before cancellation. A pointwise inequality strictly below
2 would not by itself establish a supremum strictly below 2.

An incorrect hint leaves the obligation unresolved; it does not refute the property.
The ``upper`` value is a proposed proof aid, not an assumption or an obligation
that is accepted without checking. A uniform target bound must also be
independent of the quantified coordinate, including after resolving aliases.
A coordinate-dependent target is a ``ModelDefinitionError``; it cannot become
a uniform bound by assuming a predicate about that coordinate.
Parameter poles are also retained. For example, regularity of :math:`e^{x/a}`
cannot be unconditional when the declaration allows :math:`a=0`.

Continue with a worked model
----------------------------

Start with :doc:`Markowitz allocation
<../examples/financial-models/markowitz>` to compare economic restrictions on
one scalar decision. The :doc:`model collection
<../examples/financial-models/index>` then develops component replacement,
deterministic schedules, stochastic premises, and piecewise regularity.

For the complete utility/risk/liquidity application, continue with
:doc:`market making <../examples/market-making/index>`. It connects checked
component obligations to the coupled time-dependent numerical study and its
offline reports.

Reference: contracts, domains, and proof limits
-----------------------------------------------

Use these rules when a replacement or proof attempt needs closer inspection.
The :ref:`API reference <verification-api>` below lists the workflow objects and
property functions. For a study written against the earlier draft, see the
:ref:`name migration tables <verification-draft-migration>`.

Specifications, obligations, and inference rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The terminology follows deductive verification, as used in the
`Why3 specification and verification guide <https://why3.org/doc/whyml.html>`_.
Here the verified subjects are mathematical expressions and their domains.
``Method.execute`` is ordinary Python code whose implementation remains part
of the study's trust boundary.

This is deductive verification of mathematical model properties. A
``VerificationContext`` supplies the premises :math:`\Gamma`; an
``Obligation`` states a proposition :math:`P`. The verifier attempts to
establish :math:`\Gamma\vdash P` using supported scalar rules and explicitly
registered inference rules. It checks context consistency separately and
retains conditional, refuted, and unresolved results for exploration.

* A **specification** states desired properties of a declaration.
  ``specifies`` attaches these properties to a field or model; their truth
  is determined by verification, rather than assumed from the annotation.
* An **obligation** is a named proposition to establish. The same
  ``Obligation`` record represents a model property, a method precondition,
  or an inference-rule premise.
* A **context** supplies definitions, observations, and assumptions.
  ``assume`` adds premises; ``with_obligations`` adds goals. A goal does not
  become an assumption.
* A **precondition** must be established before its method runs.
  ``Method.preconditions`` adds explicit preconditions to those consumed
  through ``Method.uses``.
* An **inference rule** derives a conclusion when its premises are
  established. ``InferenceRule`` registers such a trusted implication;
  the report retains evidence for its recursively checked premises.
* **Evidence** records a verification attempt, including a sufficient
  condition, counterexample, or open obligation. A ``Sample`` instead records
  finite numerical observations.

The API names the activity ``verify``, since an attempted check can leave an
obligation unresolved. ``VerificationModel`` automatically checks declarations;
its name does not assert that every declaration has been established.
``expression``, ``observe``, ``replace``, ``sample``, ``explain``, and ``run``
retain their names because they describe distinct operations on the context.

Model and component contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Obligations use relative paths. Nested components acquire their parent path
when inspected; replacing a component or redefining an inherited formula
rebinds its obligations. The stored annotations describe claims, not old
verdicts. Definition checking does not construct a model or select a solver
candidate. Failed and unresolved obligations remain available for exploration.

Every property function accepts optional ``consumer``, ``reason``, and
``source`` metadata. ``reason`` and ``source`` describe the claim's purpose and
provenance. ``consumer`` can associate a claim with a method name or field path
for execution. Methods also consume the obligations declared on the fields
in ``Method.uses`` and their dependencies. These labels leave the mathematical
property being checked unchanged. The same ``Obligation`` works in
``context.with_obligations(...)``, ``Method.preconditions``, and ``InferenceRule.premises``.

``Constrained = specifies(Model, *obligations)`` returns a subclass carrying
additional model-wide conditions. For a ``VerificationModel`` subclass, it also
refreshes the definition report automatically. For an ordinary ``Modello``
class, inspect the conditions with ``VerificationContext.for_model(Constrained).verify()``.
The original model and earlier contexts remain unchanged. These conditions follow
the model into nested components and are consumed by methods that read its fields.
Coordinate paths follow inheritance: a subclass that shadows a coordinate with
a regular attribute removes that path from the verification interface. Scalar
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
independent quantities within the same component. Start a new model and context
to give them separate observations, argument scopes, and contracts.
Named symbolic declarations, such as series and recurrence laws,
follow the same path-based rebinding as fields in observations, claims, and
hints. Recorded declaration paths and argument signatures must remain
compatible; removing or changing that interface requires a new model and context.
Local argument names in independent declarations can change independently.
Coordinate paths also retain their argument role during replacement. Changing
one into an observed or derived model field requires a new model and context;
replacement rejects that change before it can alter an inherited derivative.
Coordinates used by inherited calculations, observations, assumptions, claims,
or hints must remain available through a compatible path or coordinate alias.
Callable expected formulas are evaluated when a claim is checked. Resolve
coordinates through their reference callback so the formula follows the current
component interface. A callback that captures a coordinate from a changed
interface, or a scalar symbol from an earlier context, must be rebuilt before it
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

.. _retire-explicit-assumptions-and-requirements:

Retire explicit assumptions and obligations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``without_assumptions`` removes only explicitly recorded predicates, matching
their retained native scopes. Mathematically equal substitutions with different
coordinate domains remain separate assumptions. To change an intrinsic symbol
domain or an observation, change the declaration or branch before that
observation. ``without_obligations(*ids, reason=...)`` removes
obligations added with ``context.with_obligations`` and records their retirement
in the report. Declaration annotations and obligations from symbolic operations
are changed at their source. Retiring an exploratory obligation leaves a
method's preconditions in place; ``run`` verifies them again. Identical
obligation records shared by a context and a method reuse one verification, while
conflicting uses of an ID remain definition errors.
The same policy applies to rule premises: identical obligations reuse one
verification, while conflicting conditions need distinct ``name`` values.
All evidence IDs, including nested inference-rule premise paths, must be unique
within a report; a collision is a definition error.

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
rejected as foreign to the context. A defined quantity such as ``doubled = 2*x``
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
checks as property subjects. A refuted native premise makes the context
inconsistent; an unestablished premise blocks methods until supported
parameter facts establish it. The native assumption cannot establish its own
operation's premises.
If upstream CAS source binding, proof, or assumption-consistency checks exhaust
their recursion limit, the report keeps an unresolved obligation and blocks
dependent methods. A branch selector with unsupported source binding also
leaves an inspectable report, including during ``VerificationModel`` declaration.

Checks also bind active SymPy ``assuming(...)`` predicates through observations
and record them in their report, checking the combined premises for consistency.
Cached expression bindings, solution families, and proof attempts remain scoped
to that global assumption context, including its retained native coordinate
scopes. Context-bound methods distinguish those scopes too. Leaving the context
checks the original branch again without retaining context-dependent
simplifications. Use explicit ``context.assume(...)`` branches for premises that
should persist in the study.

Native substitution coordinates keep their own declared domains in cached
bindings, proof attempts, and inference-rule matching. SymPy can consider
two ``Subs`` expressions mathematically equal after renaming their bound
variables; that equality does not transfer a derivative certificate between
their scopes.
Coordinate discovery for component replacement visits every retained scope,
including mathematically equal substitutions with different bound coordinates.
For example, evaluation at zero on a real coordinate does not establish the
same operation on a positive coordinate whose domain excludes zero.
Binding and replacement also retain pending native operations before checking
their premises. If a CAS constructor drops an operation or substitutes a stale
coordinate scope, ``verify()`` reports unresolved evidence and ``run()`` blocks
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

Obligation domains must be established to be nonempty. Empty domains, including
reversed intervals after observing their endpoints, remain unresolved with an
explicit diagnostic. A symbolic interval with unknown endpoint order also stays
unresolved. This policy applies before applying a trusted inference rule and
prevents an empty scope from certifying a computation or supplying a refutation.

Supported proofs cover scalar real functions on real intervals, sign and
polynomial arguments, supported continuity domains, limits, and checked
uniform bounds. Unknown functions, mixed derivative regularity, and unsupported
parameter domains can remain unresolved. ``verify(max_steps=100)`` bounds proof
attempts, not wall-clock time within a CAS call.

Native derivatives introduce differentiability obligations with their order
and evaluation point. They do not imply convexity or continuity of the
derivatives. Divisions introduce definedness obligations before verification-layer
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
feasibility checks even without observations. Inconsistent contexts block method execution.
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
in finite or closed interval domains. This obligation also applies to
``continuously_differentiable(order=1)`` and ``continuously_differentiable(order=2)``. Continuity checks remain relative to
the requested domain.

A uniform-bound hint supplies a sufficient certificate. If its parameter guard
fails, the attempted proof is unresolved; a loose hint cannot refute the bound.
Consumed hint keys, including canonical paths and inference-rule premise hints,
are excluded from unused-hint diagnostics.

Use deferred ``Derivative(named_quantity, coordinate)`` inside declarations.
A named quantity acts as a symbol while the class body is collected, so eager
``diff`` can treat it as independent of the coordinate. Differentiate a resolved
``context.expression`` when performing a calculation outside the declaration.

.. _verification-method-reference:

Methods and inference rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass either a prepared method or a method factory to
``context.run(method, **settings)``. A factory receives the current context and returns
a ``Method``; the runner then checks its premises and relevant automatic
obligations before executing it. Settings go to the computation, not the
factory. A blocked computation raises
``VerificationError`` with ``context`` and ``report``; numerical failures retain the
same context. Unrelated unresolved claims do not block a method.

``Method.execute`` is ordinary Python code. Automatic obligations come from
the model expressions and retained native operands inspected by the verification
engine. Add mathematical premises needed only by the callback to
``Method.preconditions``. The callback implementation is part of the trust
boundary. The study supplies checks connecting its proposed answer to the
model, such as exact identities or optimality certificates; numerical
comparisons remain observations.

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

>>> from modello.contrib.verification import Method
>>> def valuation(current):
...     curve = current.expression("value", wrt="wealth")
...     return Method(
...         "symbolic valuation", lambda inner: curve, uses=("value",),
...     ).bind(current)
>>> positive_context.run(valuation).report.explain("value/strictly_increasing").verdict
'strictly true'

Use ``context.run(valuation)`` after branching: it builds the method from the
current context on every call. The model examples use this form, such as
``context.run(symbolic_method)`` or ``context.run(numerical_method, xatol=1e-10)``.

Factories capturing expressions bind the returned method with
``Method(...).bind(context)``. If a prepared method is retained for later use,
``context.run(prepared)`` rejects a different context or SymPy assumption context
before checking its captured obligations or executing it. Generic methods
that resolve expressions from the supplied ``current`` context can remain unbound.

``InferenceRule`` registers a trusted implication with a conclusion, named premises, source,
and version. ``with_rule`` adds it to a context. The engine recursively checks
the premises and retains them in the conclusion's evidence. Failed premises
make a sufficient rule inapplicable; they do not refute its conclusion.
Circular rules remain unresolved. The inference-rule implementation and CAS form the
trust boundary; a citation or user assertion alone supplies no proof.

``sample`` evaluates a finite set of points and returns a ``Sample``.
This record is labelled numerical observation and cannot discharge a global
obligation. Points must be fully bound finite real scalars in the coordinate's
declared domain, and their evaluated values must be fully bound, finite, and
real. Invalid points and unresolved symbolic values raise
``ModelValidationError``.

.. _verification-api:

API
---

.. automodule:: modello.contrib.verification

.. currentmodule:: modello.contrib.verification

Workflow objects
~~~~~~~~~~~~~~~~~

``Obligation`` is the single type for conditions. ``Method`` describes a
guarded computation; ``InferenceRule`` describes an explicitly trusted implication.

.. autoclass:: Obligation

.. autofunction:: specifies

.. autoclass:: VerificationModel
   :members:

.. autoclass:: VerificationContext
   :members:

.. autoclass:: Method
   :members:

.. autoclass:: InferenceRule

Mathematical conditions
~~~~~~~~~~~~~~~~~~~~~~~

Every function below returns an ``Obligation`` and accepts the same optional
``consumer``, ``reason``, and ``source`` metadata. Sign, order, shape, regularity,
and bounds are different mathematical conditions, rather than different
container types. Their mathematical meanings are explicit:

* ``strictly_increasing`` and ``strictly_decreasing`` require strict
  monotonicity on the domain. A strictly positive derivative is one sufficient
  certificate, rather than the definition of strict monotonicity.
* ``continuously_differentiable(order=n)`` requires :math:`C^n` regularity,
  with continuous derivatives through the finite order n.
  ``differentiable(order=n)`` only requires those derivatives to exist.
* ``supremum_below(f, b)`` states :math:`\sup_{x\in D} f(x)<b`.
  A checked bound hint :math:`M` must establish :math:`f(x)\leq M<b`
  uniformly on the domain.
* ``derivative_ratio_bound(f, bound=b)`` requires a nonzero first derivative
  and :math:`\sup_{x\in D} f(x)f''(x)/(f'(x))^2<b`. The name refers to this
  specific derivative ratio.
* ``equal_to(f, g)`` requires equality on the stated domain, including the
  definedness of both expressions. Algebraically equivalent expressions can
  establish it without having identical syntax.
* ``tends_to_zero`` requires a zero limit as the coordinate tends to
  :math:`+\infty` on the expression's eventual domain.

.. autofunction:: positive

.. autofunction:: nonnegative

.. autofunction:: nonzero

.. autofunction:: strictly_increasing

.. autofunction:: strictly_decreasing

.. autofunction:: convex

.. autofunction:: concave

.. autofunction:: continuously_differentiable

.. autofunction:: differentiable

.. autofunction:: tends_to_zero

.. autofunction:: supremum_below

.. autofunction:: derivative_ratio_bound

.. autofunction:: equal_to

.. autofunction:: predicate

Returned evidence and results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These records are returned by the workflow; they do not introduce additional
ways to declare an obligation.

.. autoclass:: Evidence
   :members:

.. autoclass:: VerificationReport
   :members:

.. autoclass:: VerificationResult

.. autoclass:: Sample

.. autoexception:: VerificationError

.. _verification-draft-migration:

Names changed from the draft research API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The earlier ``modello.contrib.research`` namespace was a draft on this branch.
Use the following names throughout code, reports, and studies. The extension
has one public vocabulary, without compatibility aliases for the draft names.

.. list-table:: Workflow terminology
   :header-rows: 1
   :widths: 45 55

   * - Draft name
     - Current name
   * - ``ResearchModel``
     - ``VerificationModel``
   * - ``ResearchCase``
     - ``VerificationContext``
   * - ``Requirement``
     - ``Obligation``
   * - ``ResearchReport``
     - ``VerificationReport``
   * - ``ResearchResult``
     - ``VerificationResult``
   * - ``ResearchError``
     - ``VerificationError``
   * - ``Theorem``
     - ``InferenceRule``
   * - ``with_theorem``
     - ``with_rule``
   * - ``Experiment``
     - ``Sample``
   * - ``requires``
     - ``specifies``
   * - ``require``
     - ``with_obligations``
   * - ``without_requirements``
     - ``without_obligations``
   * - ``check``
     - ``verify``
   * - ``with_hint``
     - ``with_bound_hint``
   * - ``Method.requirements``
     - ``Method.preconditions``
   * - ``ResearchError.case``
     - ``VerificationError.context``
   * - ``VerificationReport.checks``
     - ``VerificationReport.evidence``
   * - ``portfolio_case`` and other ``*_case`` factories
     - ``portfolio_context`` and ``*_context`` factories
   * - ``from_case``
     - ``from_context``
   * - ``SymbolicChecks``
     - ``SymbolicVerification``
   * - ``NumericalCheck``
     - ``NumericalComparison``
   * - ``checked_upper_bound`` in exported evidence
     - ``bound_hint``

JSON schema version 2 uses ``evidence`` in place of ``checks``, and ``context``
or ``contexts`` in place of ``case`` or ``cases``. The bound hint is exported
as ``bound_hint``. Reports do not emit the earlier draft keys.

.. list-table:: Mathematical property terminology
   :header-rows: 1
   :widths: 45 55

   * - Draft name
     - Current name
   * - ``increasing``
     - ``strictly_increasing``
   * - ``decreasing``
     - ``strictly_decreasing``
   * - ``smooth``
     - ``continuously_differentiable``
   * - ``uniform_bound``
     - ``supremum_below``
   * - ``curvature_bound``
     - ``derivative_ratio_bound``
   * - ``exact_form``
     - ``equal_to``
   * - ``vanishes``
     - ``tends_to_zero``

The numerical extra is now ``modello[numerics]``. The example collection is
``examples.financial_models``; its reports live under ``financial-models``.
