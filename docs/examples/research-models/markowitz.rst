Markowitz allocation with changing restrictions
===============================================

This sub-example optimizes a two-asset portfolio and retains an exact scalar
optimality certificate. It complements the portfolio GUI, which displays the
relationships without optimizing them. The input moments are illustrative;
there is no estimation from market data.

The model follows the quadratic portfolio problem in `Boyd and Vandenberghe,
Convex Optimization, section 4.4
<https://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>`_. Section 5.5.3 supplies
the trusted convex KKT sufficiency rule used below. The implementation checks
that rule's algebraic premises; it is not a general matrix optimization engine.

.. code-block:: console

   $ python -m examples.research_models.markowitz

The budget eliminates one variable
----------------------------------

Write the two weights as :math:`w` and :math:`1-w`. For means
:math:`\mu_1,\mu_2`, variances :math:`v_1,v_2`, and covariance :math:`c`,

.. math::

   m(w) &= w\mu_1+(1-w)\mu_2,\\
   f(w) &= v_1w^2+v_2(1-w)^2+2cw(1-w).

The ``Covariance`` component requires :math:`v_1,v_2\geq0` and
:math:`v_1v_2-c^2\geq0`. These are the principal-minor conditions for a symmetric
two-by-two covariance matrix to be positive semidefinite. The reduced curvature
is :math:`a=v_1+v_2-2c`. Strict positivity of :math:`a` gives uniqueness along
the budget line, even when the covariance matrix itself is singular.

``weight`` quantifies the variance curve over the real line. A separate
``candidate`` quantity records one proposed solution without replacing the
quantified coordinate. The live variance annotation checks convexity and C2
regularity; the optimizer additionally checks the exact quadratic and linear
return formulas that it consumes.

>>> from sympy import Rational, S
>>> from examples.research_models.markowitz import (
...     PortfolioRestrictions, allocate, portfolio_case,
... )
>>> case = portfolio_case()
>>> interior = allocate(case)
>>> interior.data.weight, interior.data.variance
(1/5, 4/5)
>>> interior.report.explain("global_minimum").verdict
'strictly true'

The default variances 4 and 1 are normalized algebraic units. The means are
0.08 and 0.04. Multiplying the entire covariance matrix by a positive constant
changes the variance and multipliers, while preserving the minimizing weight.

Add and remove economic restrictions
-------------------------------------

``PortfolioRestrictions`` is an immutable example-level specification. No-short
positions impose :math:`0\leq w\leq1`; a concentration cap imposes
:math:`w\leq u`; a minimum return imposes
:math:`(\mu_1-\mu_2)w\geq m_{\min}-\mu_2`.

These conditions reduce to a feasible interval :math:`[L,U]`. The code handles
either sign of the return difference, equal means, unbounded intervals, and
singleton feasible sets. An infeasible interval blocks the method and retains
the case and its mathematical report in ``ResearchError``.

The optimizer rejects assumptions restricting the quantified weight and scalar
observations of weight-dependent outputs. Put those restrictions in
``PortfolioRestrictions`` so that the candidate and its KKT certificate use the
same admissible interval. Observing market moments remains supported.
The same check applies to ``case.assume(...)`` and active SymPy
``assuming(...)`` contexts; ambient predicates cannot silently narrow the
weight interval used by the certificate.

>>> restrictions = PortfolioRestrictions(target=Rational(7, 100), cap=Rational(4, 5))
>>> targeted = allocate(case, restrictions)
>>> targeted.data.weight, targeted.data.active
(3/4, ('minimum_return',))
>>> relaxed = restrictions.without("target")
>>> allocate(case, relaxed).data.weight
1/5
>>> restrictions.target, relaxed.target, relaxed.changes
(7/100, None, ('Removed target.',))

``without`` changes this economic admissible set. It does not remove covariance
validity or a solver's need for the exact objective law. A class that replaces
the quadratic variance by a quartic cannot use this analytic shortcut; its
exact-form requirement is rechecked and blocks execution.

The determinant, reduced curvature, derivative, candidate values, and complete
objective-difference expression also have live source identities. Renaming an
unchecked helper to a valid-looking value cannot make a false covariance or an
incorrect certificate pass after subclassing.

The checked certificate
-----------------------

For :math:`a>0`, the unconstrained minimizer is
:math:`(v_2-c)/a`; the constrained candidate is its projection onto the feasible
interval. Each inequality is represented as :math:`a_iw\geq b_i`. The code
derives multipliers and checks:

.. math::

   a_iw_* - b_i &\geq0, & \lambda_i&\geq0,\\
   \lambda_i(a_iw_*-b_i)&=0, & f'(w_*)&=\sum_i\lambda_i a_i.

It also verifies the global polynomial identity

.. math::

   f(w)-f(w_*)=f'(w_*)(w-w_*)+a(w-w_*)^2.

The versioned ``scalar_convex_KKT`` rule retains these premises recursively.
Their truth, together with nonnegative curvature, certifies a global minimum
over the selected interval. Numerical agreement never discharges these claims.

>>> targeted.report.explain("global_minimum/quadratic_completion").verdict
'strictly true'
>>> targeted.data.stationarity_residual, targeted.data.completion_residual
(0, 0)

Loss of uniqueness is a separate outcome
-----------------------------------------

With identical perfectly correlated asset returns, :math:`v_1=v_2=c=1`, the
variance is constant along the budget line. The strict-curvature shortcut is
blocked. The broader convex method retains every feasible minimizer:

>>> flat = portfolio_case(first_variance=1, second_variance=1, covariance=1)
>>> result = allocate(flat, unique=False)
>>> result.data.variance, result.data.minimizers
(1, Interval(0, 1))
>>> result.report.explain("global_minimum").verdict
'strictly true'

``numerical_reference`` performs an independent bounded scalar minimization
using the optional ``research`` dependencies. It compares objective values and
weights with the exact certificate and is labelled a numerical observation.
The example does not establish multivariate convexity for larger covariance
matrices, estimate moments, or prove a stochastic investment policy optimal.
