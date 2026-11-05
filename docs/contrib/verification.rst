Deductive verification of model properties
==========================================

A calculation can depend on more than a formula: an optimum may need
concavity, or a local approximation may need continuous second derivatives.
Changing a component can invalidate those premises even when the new formula
still evaluates successfully. ``modello.contrib.verification`` attempts to
establish such properties from a model's definitions, observations, and
assumptions, and records the evidence needed before a method runs.

Start with :doc:`planning record checking <../examples/record-review/index>`.
That worked example builds a workload model from a familiar budget, explains
its calculus, and follows the same decision through assumptions, component
replacement, a failed smoothness claim, and exported evidence. It needs
undergraduate calculus and the ordinary ``modello`` installation. Plotting is
optional.

This page is a reference for readers applying that workflow to their own
models. :doc:`/getting-started` introduces the underlying declarations.

Read an obligation in its context
----------------------------------

An **obligation** is a mathematical claim to establish. The same record can
serve as a declaration's property, a method's precondition, or an
inference-rule premise. ``specifies`` attaches obligations without assuming
their truth. ``VerificationModel`` checks them at definition time; ordinary
``Modello`` classes can be checked explicitly with a ``VerificationContext``.

A context retains definitions, observations, and assumptions for one branch.
Operations such as ``observe``, ``assume``, and ``replace`` return new contexts.
A report interprets every claim under its coordinate domain and those premises:

``strictly true``
   The property is established under the recorded premises.

``conditionally true``
   ``condition`` gives a sufficient parameter condition. It need not be the
   only condition under which the property holds.

``strictly false``
   A checked refutation establishes that the stated property fails.

``unresolved``
   The attempt leaves an open obligation: for example, an unsupported operation,
   a missing proof, an exhausted attempt budget, or an inapplicable inference rule.

Assumption consistency is reported separately; contradictory premises cannot
establish a result. Numerical samples remain observations, rather than proofs
of global claims. A ``Method`` can run once the premises it consumes are
established. Its result retains the report that permitted execution;
``VerificationError`` retains the context and report when execution is blocked.

Reference: contracts, domains, and proof limits
-----------------------------------------------

Use these rules when a replacement or proof attempt needs closer inspection.
The :ref:`API reference <verification-api>` below lists the workflow objects and
property functions.

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

.. _verification-retire-premises:

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
Known finite piecewise values do not need continuity to establish definedness.
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
Undefined evaluation points remain failures. An invalid optional hint supplies
no certificate; it does not refute the requested bound.

Native ``Subs`` bindings retain their declared order. A derivative's local
coordinate also stays independent of symbols introduced by those bindings.
For example, ``Subs(Derivative(Abs(x - y), x), y, x)`` evaluates the original
curve's derivative at a cusp. Simplifying ``Abs(x - x)`` first would change
the function being differentiated. Replacing ``y`` with ``x - 1`` instead
keeps the local derivative well-defined.

When flattening would lose that scope, the automatic obligation retains the
derivative and its native binders in a premise predicate. This checks the
derivative without imposing finiteness on unrelated symbolic terms or
convergence on an enclosing integral. Division obligations likewise preserve
their original substitution scopes: a later replacement of ``x**2`` cannot
erase a pole introduced by an earlier binding of ``x``. Native derivative
premises under compound substitutions, such as replacing ``x**2``, currently
remain unresolved. Overlapping or compound substitutions inside a ranged
denominator also remain unresolved when their coordinate scopes cannot be
established.

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
fails, that certificate remains unestablished; a loose hint cannot refute the
bound. A failed certificate also cannot erase an independently checked failure
of a necessary premise. This includes hints with undefined or unsupported native
operations, whether supplied directly or retained from observations.
If a hint cannot be resolved from the model, the report retains its original
expression and checks necessary premises independently.
An optional certificate that conflicts with the subject's admissible source
conditions stays unresolved; that contradiction does not refute the subject.
A curvature ratio still fails when its subject has a
checked nondifferentiable join, even if the ratio simplifies to a smooth expression.
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

>>> from examples.record_review import ExponentialReview
>>> from modello.contrib.verification import Method, VerificationContext
>>> def response_method(current):
...     curve = current.expression("value", wrt="effort")
...     return Method(
...         "response curve", lambda inner: curve, uses=("value",),
...     ).bind(current)
>>> measured = VerificationContext.for_model(ExponentialReview, rate=1)
>>> measured.run(response_method).report.explain("value/strictly_increasing").verdict
'strictly true'

Use ``context.run(response_method)`` after branching: it builds the method from
the current context on every call. Computation settings can be supplied as
additional keyword arguments to ``run``.

Factories capturing expressions bind the returned method with
``Method(...).bind(context)``. If a prepared method is retained for later use,
``context.run(prepared)`` rejects a different context or SymPy assumption context
before checking its captured obligations or executing it. Generic methods
that resolve expressions from the supplied ``current`` context can remain unbound.

``InferenceRule`` registers a trusted implication with a conclusion, named premises, source,
and version. ``with_rule`` adds it to a context. The engine recursively checks
the premises and retains them in the conclusion's evidence. Failed premises
make a sufficient rule inapplicable; they do not refute its conclusion.
An unregistered rule conclusion remains unresolved; its proof attempt does not
evaluate unsupported native operations.
Circular rules remain unresolved. The inference-rule implementation and CAS form the
trust boundary; a citation or user assertion alone supplies no proof.

``sample`` evaluates a finite set of points and returns a ``Sample``.
This record is labelled numerical observation and cannot discharge a global
obligation. Points must be fully bound finite real scalars in the coordinate's
declared domain, and their evaluated values must be fully bound, finite, and
real. Point expressions also retain their original operand domains and native
derivative premises: an unevaluated ``log(0)**0`` cannot supply a valid point
by simplifying to one. Invalid points and unresolved symbolic values raise
``ModelValidationError``.

.. _verification-example-contract:

Write the record-review method's contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :doc:`record-review walkthrough <../examples/record-review/index>` uses
``review_context()`` to prepare its inputs and mathematical argument. Here is
how the example connects that argument to a computation.

An ``Obligation`` names a claim. ``GLOBAL_MAXIMUM`` states
:math:`G(s)\geq G(t)` for every nonnegative effort :math:`t`, where :math:`s`
is the proposed checking time. ``InferenceRule`` records the supporting-line
argument from the walkthrough as a trusted implication. Its premises ask the
checker to establish concavity, differentiability, a zero derivative at the
candidate, and candidate feasibility:

.. literalinclude:: ../../examples/record_review.py
   :language: python
   :start-at: GLOBAL_MAXIMUM = Obligation(
   :end-before: @dataclass

A registered implication is part of the trust boundary. The engine checks its
premises; it does not prove that they imply the conclusion. A failed premise
makes this argument inapplicable and leaves the conclusion unresolved.

The example's ``_evaluate_plan`` callback evaluates the proposed checking time,
net saving, workload, and workload curvature. A ``Method`` pairs that callback
with the properties it needs:

.. literalinclude:: ../../examples/record_review.py
   :language: python
   :start-at: review_plan = Method(
   :end-before: def review_context(

``uses`` names the fields read by the computation. Following their dependencies
consumes the response's declared increase, concavity, and C2 properties.
``preconditions`` adds the global-maximum conclusion and a positive candidate
workload. The callback executes after those premises are established; its
Python implementation is trusted too.

The convenience function registers the rule and accepts a ``RecordWork``
component containing the entry and repair estimates, plus the checking rate.
Its defaults provide the walkthrough's baseline. Each call starts a fresh
context, leaving the candidate and staffing budget available for later inputs:

.. literalinclude:: ../../examples/record_review.py
   :language: python
   :pyobject: review_context

.. _verification-bound-hints:

Help an open proof with a checked bound
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A bound can need an explicit argument even when each parameter is fixed.
Continue with the same record-review model. Could *any* checking time save
seven minutes per record? There are only six repair minutes to avoid, and
checking itself costs time, so six is a uniform upper bound on net saving.

>>> from sympy import log
>>> from examples.record_review import EFFORT_DOMAIN, review_context
>>> from modello.contrib.verification import supremum_below
>>> proposed = review_context().observe(candidate=2*log(3))
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
refute the original bound. Here ``unresolved`` meant a proof aid was missing.

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
* ``convex`` and ``concave`` allow linear sections. For a twice differentiable
  scalar function on an interval, a nonnegative second derivative is sufficient
  for convexity; a nonpositive second derivative is sufficient for concavity.
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
