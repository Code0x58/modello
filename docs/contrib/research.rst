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
subclass. Every property function returns a ``Requirement``. Attach those
requirements to a declaration with ``requires``:

>>> from sympy import exp, S
>>> from modello import Coordinate, InstanceDummy
>>> from modello.contrib.research import (
...     ResearchCase, ResearchModel, increasing, positive, requires, smooth,
... )
>>> class Utility(ResearchModel):
...     wealth = Coordinate("w", real=True)
...     gamma = InstanceDummy("gamma", real=True)
...     value = requires(-exp(-gamma*wealth),
...         increasing("value", wrt="wealth", consumer="valuation", reason="More wealth must be preferred."),
...         smooth("value", wrt="wealth", consumer="price_curvature", reason="This calculation uses C2 regularity."),
...     )
>>> Utility.definition_report().explain("value/increasing").verdict
'conditionally true'

Requirements use relative paths. Nested components acquire their parent path
when inspected; replacing a component or redefining an inherited formula
rebinds its requirements. The stored annotations describe claims, not old
verdicts. Definition checking does not construct a model or select a solver
candidate. Failed and unresolved requirements remain available for exploration.

``consumer``, ``reason``, and ``source`` are optional metadata accepted by every
property function. They describe why a condition is needed; they do not change
its mathematical meaning. There is no separate claim or declaration-wrapper
type to construct. The same requirement works in ``case.require(...)``,
``Method.requirements``, and ``Theorem.premises``.

``Constrained = requires(Model, *requirements)`` returns a subclass carrying
additional model-wide conditions and its refreshed definition report. The
original model and earlier cases remain unchanged. These conditions follow the
model into nested components and are consumed by methods that read its fields.
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
Replacement cannot split previously aliased fields into independent quantities:
start a new model and case to give those quantities separate observations and
contracts. Named symbolic declarations, such as series and recurrence laws,
follow the same path-based rebinding as fields in observations, claims, and
hints. Recorded declaration paths and argument signatures must remain
compatible; removing or changing that interface requires a new model and case.
Coordinate paths also retain their argument role during replacement. Changing
one into an observed or derived model field requires a new model and case;
replacement rejects that change before it can alter an inherited derivative.

Replacement preserves explicit observations. Copied inferred values whose
definitions depend on the replaced component are invalidated and listed in
the report, so an old calculated intensity cannot constrain its replacement.
Dependencies include named laws and composed series queries, even when they
have no free scalar parameters. Methods also consume the component contracts
reached through those declarations. Selecting one recurrence explicitly does
not consume an alternative law merely because it targets the same series.
Equivalent declarations with renamed local variables share their contracts;
ambiguous series queries remain unresolved until a law is selected explicitly.

``ResearchCase.for_model`` also accepts an ordinary ``Modello`` class, which
then receives checks when explicitly inspected.

Branch assumptions and observations
-----------------------------------

Cases are immutable. Use their actual symbols, preserving identities across
branches:

>>> case = ResearchCase.for_model(Utility)
>>> risk_positive = positive("gamma", name="positive_risk")
>>> case.require(risk_positive).check().explain("positive_risk").verdict
'conditionally true'
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
The same policy applies to theorem premises: identical requirements reuse one
check, while conflicting conditions need distinct ``name`` values.
All evidence IDs, including nested theorem premise paths, must be unique within
a report; a collision is a definition error.

Check a proposed proof
----------------------

The logistic intensity makes a useful research substitution:

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

Independent coordinates with the same name receive distinct scoped labels in
exports, such as ``a.x`` and ``b.x``. Deliberately shared coordinates retain one
identity even when reached through several component paths. Report printers
label the original symbolic tree without substituting or simplifying it, so
formatting cannot turn a difference of independent coordinates into zero.

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
in finite or closed interval domains.

A uniform-bound hint supplies a sufficient certificate. If its parameter guard
fails, the attempted proof is unresolved; a loose hint cannot refute the bound.
Consumed hint keys, including canonical paths and theorem-premise hints, are
excluded from unused-hint diagnostics.

Use deferred ``Derivative(named_quantity, coordinate)`` inside declarations.
A named quantity acts as a symbol while the class body is collected, so eager
``diff`` can treat it as independent of the coordinate. Differentiate a resolved
``case.expression`` when performing a calculation outside the declaration.

Methods and theorems
--------------------

``Method`` records a computation, its explicit requirements, and the field
paths it reads. Pass either a prepared method or a method factory to
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
A locally evaluated expression such as ``Subs(curve, x, 1)`` keeps its own
scope: its obligations concern that evaluation, while the unevaluated source
curve can still have a pole elsewhere.

>>> from modello.contrib.research import Method
>>> def valuation(current):
...     curve = current.expression("value", wrt="wealth")
...     return Method("symbolic valuation", lambda inner: curve, uses=("value",)).bind(current)
>>> positive_risk.run(valuation).report.explain("value/increasing").verdict
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
These checks include active SymPy ``assuming(...)`` predicates as well as
``case.assume(...)``. Parameter-only assumptions remain supported.
Quote-distance and terminal-inventory coordinates are included in the check;
a predicate on a locally substituted argument cannot narrow the solver's domain.

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
