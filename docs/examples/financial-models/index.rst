Financial models with changing constraints
==========================================

A researcher often needs to compare several versions of a model: a new cost
law, a tighter admissible set, or a method with weaker regularity obligations.
These studies show how to keep an earlier context, make the change, and inspect
which computations still have the evidence they need.

The examples cover pricing, execution, and allocation with illustrative exact
inputs. They assume familiarity with Python and basic SymPy expressions. Start
with :doc:`../../contrib/verification` for the declaration, context, evidence, and
method workflow; :doc:`../../getting-started` introduces Modello itself.

Choose a study
---------------

Follow the order below for a progression from one scalar decision to
stochastic and piecewise models. Each page also states its own problem and
can be read independently.

1. :doc:`markowitz`: add a return target or concentration cap, or allow
   shorting. Inspect exact active constraints, infeasibility, and the loss of
   uniqueness under flat curvature.
2. :doc:`participation-execution`: add or remove a cap and deadline, then
   replace the impact law. Inspect constrained optimality, risk-neutral
   nonattainment, and a blocked closed-form shortcut.
3. :doc:`black-scholes`: add dividends or deterministic coefficient curves.
   Inspect parity, Greeks, tail-integral boundary conditions, and exact PDE
   identities.
4. :doc:`cir-bond`: add or remove the Feller condition. Inspect boundary
   evidence that changes while affine valuation remains applicable.
5. :doc:`perpetual-put`: compare global C2 with piecewise smoothness and smooth
   fit. Inspect C1 at the exercise join, a C2 counterexample, and explicitly
   trusted stopping verification.

.. toctree::
   :hidden:
   :maxdepth: 1

   markowitz
   participation-execution
   black-scholes
   cir-bond
   perpetual-put

Continue with :doc:`continuous-time market making <../market-making/index>`
to connect utility, risk, and liquidity obligations to a coupled numerical
terminal-value problem.

Separate the roles of constraints
---------------------------------

The same mathematical condition can play different roles in an investigation:

* A validity condition gives an expression its meaning: positive spot and
  strike, a positive execution rate, or a positive semidefinite covariance.
* An economic restriction defines admissible decisions: a participation cap,
  a completion deadline, no-short positions, or a return target.
* A method contract states what a computation uses: a pure-power execution
  law, a quadratic variance, or an exact utility form.

Removing an economic restriction can preserve a valid model. Replacing a
formula can block a shortcut while leaving a general method applicable.
Dropping strict curvature can preserve minimizers while losing uniqueness.
The studies retain these outcomes in separate branches.

Work through a model change
---------------------------

1. Keep an unobserved model or early context as the common branch point.
   Assumptions and observations accumulate, so branch before an observation
   when you want to compare another value.
2. Make a named branch for the component or restriction change. State the
   admissible set and the calculation you intend to compare.
3. Run the method through the current context. Inspect its report for established
   premises, sufficient parameter conditions, refutations, or open obligations.
4. Inspect the candidate's certificate. The examples check exact identities,
   domain conditions, and the premises of any explicitly trusted rule.
5. Compare with an independent numerical calculation. Residuals, quadrature,
   and tolerance convergence provide numerical observations.
6. Revisit the contract after a replacement or relaxation. Keep both the
   successful and blocked computations, recording why a premise was removed.

``context.run(method_factory)`` prepares a method for the current branch. A
prepared method that captures context-specific expressions uses
``Method.bind(context)``. Retiring an exploratory obligation leaves the method's
preconditions and the obligations from symbolic operations in place.

Read and compare the reports
----------------------------

Each study report keeps context inputs, completed computations, and blocked
attempts together. Open a computation's premises link to see the obligation,
coordinate domain, conditions, and any counterexample or nested inference-rule
premises. The report retains the declared formula alongside the specialized
expression, so an identity remains understandable after numerical substitution.

Check the kind of result being reported. An exact scalar identity, a conclusion
under an explicitly trusted stochastic rule, and a numerical comparison answer
different questions. The scalar checker handles supported algebra and
regularity rules; stochastic verification uses the stated external theory.
The :doc:`verification guide's reference sections <../../contrib/verification>` describe
the proof scopes and how unresolved results are preserved.

Replay the collection
---------------------

Install the optional numerical and plotting dependencies, then generate the
reports:

.. code-block:: console

   $ pip install 'modello[numerics]'
   $ python -m examples.financial_models --output artefacts/financial-models

Open ``artefacts/financial-models/report.html``. Each model has its own directory
with HTML evidence pages, descriptive JSON, results CSV, and SVG/PNG figures.
The `representative report built with this documentation <study/report.html>`_
uses ``--compact`` to select one illustrative branch per model. The default
runner includes the full branch collection.

Use ``--models markowitz participation-execution`` to select sub-examples, or
``--symbolic-only`` to replay with SymPy alone. The individual model modules
also have their own runners. From a locked repository checkout, the full
numerical replay is:

.. code-block:: console

   $ uv run --locked --extra numerics python -m examples.financial_models --output artefacts/financial-models

Replay uses the Python source and records dependency versions, exact inputs,
source fingerprints, and numerical settings. JSON is a descriptive audit
export. For a custom study, keep the Python code that constructed its contexts
and computations. The reusable ``write_reports`` writer accepts a
``replay_command`` for that code's entry point; a study key labels its output
directory.
