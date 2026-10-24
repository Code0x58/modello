Scenario contracts
==================

These models are independent of the GUI gallery. They exercise public APIs with
small, independently calculated answers and equivalent formulations. Most joint
problems run through both construction and exact solution families. A finite
suite cannot prove that arbitrary symbolic mathematics will be solved.

Run ``uv run --locked --dev pytest tests/scenario``. The regular CI suite also
collects these tests.

Coverage map
------------

===============================  ==============================================
Module                           Contracts
===============================  ==============================================
``test_vectors``                 Empty, singleton, real/complex and rational
                                 data; bilinear/Hermitian contraction; partial
                                 observations; several shape parameters;
                                 equivalent shapes; observed/negative indices;
                                 zero-product nonidentifiability.
``test_calculus``                Gradients, Hessians, mixed partials; named
                                 dependencies; integral/sum composition; moving
                                 endpoints; exceptional limits; implicit Euler;
                                 family response versus partial derivatives;
                                 free bounds sharing a bound symbol.
``test_operator_composition``    Split/whole, nested and multidimensional
                                 reductions; reversed intervals; Piecewise
                                 boundaries; improper/nested integrals; inverse
                                 integration parameters; higher derivatives
                                 at removable-singularity points.
``test_recurrences``             Shifted origins; coefficient branches; arbitrary
                                 query order; integer domains; composed and
                                 mutually driven series; native aggregate and
                                 family queries; inverse/invalid histories.
``test_series_inference``        Arbitrary-order finite autoregressions; sums of
                                 relative lags; observation-driven horizons;
                                 bidirectional states/innovations; nonlinear and
                                 time-varying forcing; affine coefficient cases.
``test_model_vectors``           Whole-model versus component equations;
                                 current-step and implicit updates; terminal
                                 boundaries; sliding windows in equivalent
                                 notation; nested paths; invalid indices;
                                 empty/start-zero laws; disjoint state laws.
``test_sequence_parity``         Shared finite causal answers, forward/inverse
                                 solving and domains; moving input bounds;
                                 native binder scopes; symbolic input sizes;
                                 queried-prefix versus whole-window validity.
``test_composition_contracts``    Explicit nested laws; lexical binders; centered
                                 boundary equations; source denominator domains;
                                 unused branches/reductions; persistent process
                                 dependencies; local versus path calculations;
                                 exact endpoint state validation.
``test_solver_composition``      Typed members and schema equations; nonlinear
                                 joint alternatives; observation order and
                                 release; inheritance/refinement; shared inverse
                                 parameters; explicit approximate/exact boundary.
``test_stochastic``              Arithmetic/geometric/affine-additive values and
                                 moments; nonlinear/time-dependent local
                                 calculus; standalone/native/finite bodies;
                                 zero-time identities and domains; inverse
                                 coefficient fitting; named model definitions.
``test_symbolic_protocol``       SymPy reconstruction, copying/serialization,
                                 alpha-renaming, scopes, assumptions, and native
                                 operators surrounding custom declarations.
``test_capability_boundaries``   Unsupported causality, closed forms, tensor
                                 states, cyclic queries, exact SDE classes,
                                 ambiguous laws, unbound indices, schema and
                                 dimensional errors. These are explicit library
                                 limits, never upstream xfails.
``test_upstream_sympy``          Standalone upstream-only regression reproducers.
===============================  ==============================================

Regression policy
-----------------

A failing Modello scenario is a library defect or a documented capability gap.
Only an independently reproduced failure without Modello imports belongs in
``test_upstream_sympy``. Its strict xfail names the failure, pins the exception
kind, and includes the tested version, minimal program, actual/expected answers,
likely source location, and any known workaround in its docstring. Diagnostics
include expression representations. An unexpected pass fails CI, prompting
removal of the xfail after an upstream fix.

Keep a failing equivalence case beside its passing counterpart. Prefer exact
small answers or independent iteration over repeating the implementation in an
oracle. Parameterize meaningful domains and boundaries rather than duplicating
near-identical examples. The shared solver fixture deliberately exercises both
public solve paths.
