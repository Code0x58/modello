Research models and changing constraints
========================================

Each sub-example uses :mod:`modello.contrib.research` to turn a model's
mathematical requirements into inspectable obligations. The collection explores
pricing, execution, and allocation with separate components and methods. The
examples use illustrative exact inputs and make their proof boundaries explicit.

Start with :doc:`../../contrib/research` for the API and the
:doc:`continuous-time market-making study <../market-making/index>` for the
original application. The studies below ask what happens when a researcher
changes a component, adds an admissibility restriction, or removes a condition
that a particular shortcut previously needed.

Replay the complete collection with independent numerical references and
figures:

.. code-block:: console

   $ uv run --locked --extra research python -m examples.research_models --output artefacts/research-models

Open ``artefacts/research-models/report.html``. Each model has its own directory
with HTML evidence pages, descriptive JSON, results CSV, and SVG/PNG figures.
The `representative report built with this documentation <study/report.html>`_
uses ``--compact`` to select one illustrative branch per model. The default
runner includes the full branch collection.

Use ``--models black-scholes markowitz`` to select sub-examples, or
``--symbolic-only`` to replay with SymPy alone. The individual model modules
also have their own runners. JSON is an audit export, not an executable case
file; replay uses the Python source and records dependency versions, exact
inputs, source fingerprints, and numerical settings.

.. toctree::
   :maxdepth: 1

   black-scholes
   cir-bond
   perpetual-put
   participation-execution
   markowitz

Three kinds of constraint
--------------------------

Keep these roles visible while exploring:

* A mathematical domain or validity condition gives an expression its meaning:
  positive spot and strike, a positive execution rate, or a positive semidefinite
  covariance matrix.
* An economic restriction defines the admissible decisions: a participation
  cap, a completion deadline, no-short positions, or a return target.
* A method's contract identifies the formulas and properties it actually uses:
  a pure-power execution law, a quadratic variance, or an exact utility form.

Removing an economic restriction need not invalidate the model. Changing a
formula can invalidate a shortcut while leaving a general method applicable.
Dropping strict curvature can preserve minimizers while losing uniqueness.
These outcomes are recorded separately from contradictions and missing proofs.

.. list-table:: What each model contributes
   :header-rows: 1
   :widths: 24 40 36

   * - Model
     - Change to explore
     - Checked consequence
   * - European BSM
     - Add dividends or deterministic coefficient curves.
     - Parity, Greeks, tail-integral boundary conditions, and exact PDE identities.
   * - Constant-participation execution
     - Add or remove a cap and deadline; change the impact law.
     - Constrained KKT, risk-neutral nonattainment, and invalidated closed-form shortcuts.
   * - Markowitz allocation
     - Add a return target or concentration cap; allow shorting.
     - Exact active constraints, infeasibility, and loss of uniqueness under flat curvature.
   * - CIR bonds
     - Add or remove the Feller condition.
     - Changed boundary evidence while affine valuation remains applicable.
   * - Perpetual American put
     - Replace a global C2 requirement by piecewise smoothness and smooth fit.
     - C1 at the exercise join, a checked C2 counterexample, and separately trusted stopping verification.

A repeatable investigation
---------------------------

1. Keep an unobserved model or early case as the common branch point. Parameter
   assumptions and observations accumulate; a contradictory new observation
   does not replace an earlier one. ``without_assumptions`` and
   ``without_requirements`` record why explicit exploratory premises were removed.
2. Make a named branch for the proposed component or restriction change. Retain
   the earlier branch and record which admissible set is being compared.
3. Inspect definition reports and the evidence a chosen method consumes. A
   conditionally true result carries a sufficient parameter condition. An
   unresolved claim is an open obligation rather than a negative result.
4. Generate a candidate and check its certificate independently of the formula
   that proposed it. The sub-examples use exact identities and domain conditions,
   with explicit trusted calculus or convexity rules where required.
5. Compare against an independent numerical calculation. Residuals, quadrature,
   and mesh or tolerance convergence remain numerical observations.
6. Revisit the contract after a replacement or relaxation. Report the conclusions
   that survive, the methods that become blocked, and the conclusions that no
   longer follow from the remaining premises.

Changes to the shared workflow
------------------------------

Reviewing these models exposed several reusable requirements. A method that
captures case-specific expressions now uses ``Method.bind(case)``; a new
assumption branch must rebuild that factory even when its underlying problem
snapshot is shared. Removing a case-added requirement cannot waive a method's
own premise or an automatic denominator/differentiability obligation.

Source identities matter beyond a final formula. A false determinant, shifted
integral, or stale derivative helper can make a valid-looking certificate
misleading. The methods check the helper formulas they consume, including
terminal conditions for deterministic tail integrals. Independent review tests
change helpers while preserving a final scalar value, so a rejected shortcut
must fail for the intended mathematical reason.

Automatic obligations respect intrinsic coordinate domains, and native
substitutions are evaluated in their original derivative scopes. A limited
constant-join rule now checks actual assigned values and one-sided derivatives
for C0, C1, and C2. It establishes the put's interior smooth fit and refutes C2
at its exercise boundary. An included join endpoint with only one active side
does not supply an ordinary derivative certificate.

Exports retain the declared requirement and expected formula before numerical
observations. A successful equality therefore remains inspectable when its
specialized form has reduced to zero. Repeated scalar proof attempts within a
report share resolved CAS work while retaining each consumer's evidence and
the proof-attempt budget.

The scalar checker does not establish stochastic verification theorems,
multivariate positive semidefiniteness, or infinite-dimensional variational
existence by itself. Source citations identify trusted theory; they do not
replace checked premises or register new proof rules automatically.

Further work can build on these concrete gaps: coordinate scopes for mixed
derivatives, parameter-dependent piecewise joins, more analytic composition
rules, and reusable certificate builders for broader constrained problems.
The current examples keep those limits explicit instead of treating numerical
agreement or a citation as a missing proof.
