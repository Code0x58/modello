Diagnostic guarantees and reference
===================================

The :doc:`debugging guide <debugging>` introduces the interactive workflow.
The tools separate recorded facts, witnessed feasibility, exact contradictions,
and checks that could not decide. They analyse the recorded SymPy model, with
all declared symbol assumptions retained as background constraints.

What each conclusion establishes
---------------------------------

``consistent``
   A closed, finite numerical assignment has passed every recorded equality and
   every declared domain predicate. This proves existence, without establishing uniqueness or
   enumerating every solution. A solver candidate is checked before it can
   serve as a witness.

``inconsistent``
   An exact consequence contradicts an equality or domain, an expression is
   undefined, a univariate subsystem has an empty solution set, or a rational
   affine subset supplies a contradiction certificate.
   A certificate is a weighted sum of original equation residuals equal to a
   nonzero constant. Its ``verify()`` method checks that identity directly.

``unknown``
   No checked witness or contradiction has been established. Empty legacy ``solve``
   results, unsupported mathematics, undecided predicates, Float residuals,
   conditional external parameters, and exhausted search budgets do not become
   inconsistency proofs.

The analyser propagates ground assignments, checks their necessary consequences,
and extracts the affine equations with exact rational coefficients. Those
equations can prove a contradiction even when other equations are nonlinear.
Otherwise it tries sample assignments and, if needed, SymPy solver candidates
as possible witnesses. Failure to find a witness proves nothing.

``solution_sets`` retains SymPy sets for equations that reduce to one model
field after forced, exact, finite numerical substitutions. Each ``SolutionSet``
records the field, original equations, substitutions, search domain, and returned
set. Sets from equations for the same field are intersected. ``solveset`` can
represent finite roots, infinite families, and unresolved ``ConditionSet``
expressions; these are not flattened into an allegedly exhaustive candidate list.
External symbolic parameters, Float equations, noncommutative expressions, and
equations still involving several unknowns are excluded from this step.

The search domain respects integer, rational, real, sign, and nonzero restrictions
where declared. It can contain extra values: other predicates, such as primality,
are checked separately. These are necessary solutions to a subsystem after
substitution, not a projection of every constraint in the model. A nonempty set
alone proves neither whole-model consistency nor uniqueness. An empty set proves
inconsistency because the domain contains all permitted finite scalar values.
Finite roots are audited as witness proposals against the original problem;
unresolved sets and solver exceptions do not become inconsistency proofs.

``LinearAnalysis`` reports only the rational affine part. Its ``complete`` flag
states whether that part includes every recorded equation. Rank and nullity
ignore domain restrictions and are not general model dimensions: ``x**2 = 0``
has a unique real solution despite a singular Jacobian, while nonnegative
``x + y = 0`` permits only zero despite equality nullity one. Symbolic
coefficients are excluded rather than assigning a generic rank to all parameter
values. The actual included fields and equations remain inspectable.

Exact checks do not infer domain restrictions erased before capture by Python,
SymPy, or model-definition simplification. Declare requirements such as a
nonzero denominator explicitly when they belong to the mathematical model.

Conflicts, probes, and costs
----------------------------

When available, a linear certificate identifies a proven conflict directly.
With ``conflicts=True``, a cached deletion filter removes a constraint only
when its removal leaves a proven contradiction. The remaining set is a proven
conflict. It is labelled ``irreducible`` only when every single-member deletion
admits a checked witness. Each final deletion check is available through
``conflict.removal_checks``. Unknown checks or a limited budget leave
irreducibility unestablished. The filter finds one conflict, without claiming
minimum cardinality, preferences, or enumeration of all conflicts.

With ``relaxations=True``, the original problem is checked after removing each
whole supplied constructor argument. Definitions and all domain assumptions
stay fixed. Component inputs group all copied fields, including derived values.
``input_checks`` records all outcomes; ``relaxations`` includes only witnessed
feasible removals. These are single-input probes, not an enumeration of minimal
diagnoses. Multiple independent conflicts can require several changes.

The shared ``max_checks`` budget counts distinct subset assessments. Conflict
reduction runs before input probes. Individual assessments may simplify and
solve expressions, so this is not a wall-clock timeout. Field queries,
component views, snapshot formatting, and ordinary exception formatting do not
run analysis. Preparation is shared with construction rather than rebuilding
an approximate diagnostic model. Numeric model results are unchanged by analysis.

Research and SymPy foundations
-------------------------------

* Raymond Reiter, *A Theory of Diagnosis from First Principles*, Artificial
  Intelligence 32(1), 57–95 (1987),
  `doi:10.1016/0004-3702(87)90062-2
  <https://doi.org/10.1016/0004-3702(87)90062-2>`_.
  This supplies the distinction between conflicting requirements and alternative
  diagnoses. The current tooling does not implement diagnosis enumeration.
* Ulrich Junker, *QUICKXPLAIN: Preferred Explanations and Relaxations for
  Over-Constrained Problems*, AAAI 2004, 167–172,
  `published paper <https://cdn.aaai.org/AAAI/2004/AAAI04-027.pdf>`_.
  The background/requirement distinction and basic deletion construction guide
  conflict filtering. The implementation uses conservative three-valued checks,
  rather than claiming QuickXplain's complete-oracle guarantees.
* John W. Chinneck and Erik W. Dravnieks, *Locating Minimal Infeasible Constraint
  Sets in Linear Programs*, ORSA Journal on Computing 3(2), 157–168 (1991),
  `doi:10.1287/ijoc.3.2.157 <https://doi.org/10.1287/ijoc.3.2.157>`_.
  Infeasible-subsystem filtering motivates evidence that helps locate formulation
  problems. The current exact affine checker is not a general LP feasibility solver.
* SymPy's `assumptions guide
  <https://docs.sympy.org/latest/guides/assumptions.html>`_ and
  `solution checking reference
  <https://docs.sympy.org/latest/modules/solvers/solvers.html#sympy.solvers.solvers.checksol>`_
  document inconclusive results. Candidate generation disables numerical solver
  checks; diagnostic equality verdicts use exact residuals and preserve unknowns.
* The `solveset reference
  <https://docs.sympy.org/latest/modules/solvers/solveset.html>`_ explains its
  set-valued results and explicit domains: unlike ``solve``, the domain is not
  inferred from the target symbol's assumptions. The `algebraic solving guide
  <https://docs.sympy.org/latest/guides/solving/solve-equation-algebraically.html>`_
  explains why some solution sets cannot be enumerated. Diagnostics retain these
  representations and use an explicit domain rather than assuming complex roots
  satisfy a real field's requirements.
* The `linear equation conversion reference
  <https://docs.sympy.org/latest/modules/solvers/solveset.html#sympy.solvers.solveset.linear_eq_to_matrix>`_
  and `matrix zero-testing discussion
  <https://docs.sympy.org/latest/tutorial/matrices.html#zero-testing>`_ motivate
  restricting rank certificates to exact rational coefficients, retaining the
  original equations, and avoiding universal conclusions from symbolic pivots.

Recorded problems and reports
------------------------------

.. currentmodule:: modello.diagnostics

.. autoclass:: ModelProblem
   :members:

.. autoclass:: ModelSolveError

.. autoclass:: ModelField

.. autoclass:: ModelConstraint
   :members: symbols

.. autoclass:: ComponentView

.. autoclass:: Candidate
   :members: value

.. autoclass:: FieldValue

.. autoclass:: CandidateCheck

.. autoclass:: SolutionSet

.. autoclass:: ResidualCheck

.. autoclass:: AssumptionCheck

.. autoclass:: ProblemAnalysis
   :members: relaxations

.. autoclass:: LinearAnalysis
   :members: nullity

.. autoclass:: LinearCertificate
   :members: verify

.. autoclass:: Conflict

.. autoclass:: RemovalCheck

.. autoclass:: InputCheck
