CIR bonds and a removable Feller condition
==========================================

This study values a unit zero-coupon bond and keeps scalar identities separate
from the stochastic rules needed to interpret the formula. The dynamics are
explicitly risk-neutral:

.. math::

   dR_s=\kappa(\theta-R_s)\,ds+\sigma\sqrt{R_s}\,dW_s^{\mathbb Q},
   \qquad \kappa,\theta,\sigma>0,\quad R_0\geq0.

The source is `Cox, Ingersoll and Ross (1985), A Theory of the Term Structure
of Interest Rates, section 3
<https://roycheng.cn/files/papers/interestRate/paper_Cox%26Ingersoll%26Ross_1985.pdf>`_.
Page 391 discusses the zero boundary; pages 392--393 give the pricing equation
and affine bond formula. The parameters here already describe the risk-neutral
drift. They are not physical-measure estimates with an omitted risk premium.

.. code-block:: console

   $ python -m examples.research_models.cir_bond
   $ python -m examples.research_models.cir_bond --numerical

Importing the example requires only SymPy and Modello. The second command
imports the optional SciPy dependency inside its numerical method.

One formula on the nonnegative state domain
-------------------------------------------

For remaining time :math:`\tau=T-t\geq0`, the declaration uses an algebraically
equivalent hyperbolic representation of the source formula:

.. math::

   h&=\sqrt{\kappa^2+2\sigma^2},\\
   D(\tau)&=h\cosh(h\tau/2)+\kappa\sinh(h\tau/2),\\
   B(\tau)&=\frac{2\sinh(h\tau/2)}{D(\tau)},\\
   \log A(\tau)&=\frac{2\kappa\theta}{\sigma^2}
      \left[\log\frac{h}{D(\tau)}+\frac{\kappa\tau}{2}\right],\\
   P(\tau,r)&=\exp\bigl(\log A(\tau)-B(\tau)r\bigr).

The positive parameters and nonnegative time keep :math:`D` positive. The
implementation checks definedness and coefficient regularity before using
reduced residuals. Exact-form requirements connect the selected-horizon
coefficients, the rate curve, and its initial-rate evaluation to this formula.
Changing a price definition cannot inherit an old valuation certificate.

``rate`` and ``time`` are quantified nonnegative coordinates. ``initial_rate``
and ``horizon`` select one evaluation without fixing those coordinates in a
global shape claim. All five scalar inputs have their intrinsic CIR domains;
the Feller inequality is an additional, optional assumption. Python float
inputs are interpreted through their decimal spelling; exact rational inputs
make the audit reproducible.

>>> from examples.research_models.cir_bond import cir_case, price, verification_method
>>> case = cir_case()
>>> result = price(case)
>>> result.report.explain("rate_decreasing").verdict
'strictly true'
>>> result.report.explain("rate_convex").verdict
'strictly true'

At positive maturity the price is strictly decreasing and convex in current
rate, including a curve whose state domain contains zero. At maturity zero the
price is identically one; its contract records nonincrease rather than strict
decrease. An initial rate of zero is also valid for valuation. The example
does not require two derivatives of :math:`\sqrt r` at zero: the pricing
generator consumes the variance rate :math:`\sigma^2r`.

Checked scalar identities
-------------------------

The verification method differentiates the actual, parameter-bound coefficient
expressions and checks

.. math::

   B'&=1-\kappa B-\tfrac12\sigma^2B^2,\\
   (\log A)'&=-\kappa\theta B,\\
   P(0,r)&=1,\\
   \partial_\tau P
      &-\kappa(\theta-r)\partial_rP
       -\tfrac12\sigma^2r\partial_{rr}P+rP=0.

For the last identity it uses the normalized residual obtained from
:math:`\log P=\log A-Br`. Positivity and the exponential price-form contract
justify that reduction. This avoids a large native mixed-derivative
declaration and preserves the distinction between a scalar identity and a
stochastic valuation theorem.

>>> verification = case.run(verification_method(case))
>>> verification.data.riccati_b, verification.data.riccati_log_a
(0, 0)
>>> verification.data.generator, verification.data.terminal_price
(0, 1)

``CIR_affine_discounted_expectation`` is an explicit, versioned trusted rule
for the source formula under the stated risk-neutral CIR dynamics. Its
recursively checked premises include the parameter domains, exact drift and
diffusion, coefficient formulas, and the nonnegative-path rule. Solving the
pricing equation alone is not presented as a proof of stochastic uniqueness
or of the discounted-expectation representation.

Feller is a boundary requirement
--------------------------------

The separately registered ``CIR_unattainable_zero_boundary`` rule requires
:math:`R_0>0` and

.. math::

   2\kappa\theta\geq\sigma^2.

It concludes that the standard CIR solution does not hit zero at finite time.
Equality is included. The weaker ``CIR_nonnegative_solution`` rule allows
:math:`R_0=0` and does not need the Feller inequality. These are sourced
stochastic rules, not properties inferred by differentiating a square root or
sampling a simulated path.

>>> from sympy import Rational
>>> from modello.contrib.research import ResearchError
>>> from examples.research_models.cir_bond import strict_positive_boundary_method
>>> weak = cir_case(sigma=Rational(2, 5))
>>> weak.expression("feller_margin") < 0
True
>>> price(weak).report.explain("affine_valuation").verdict
'strictly true'
>>> try:
...     weak.run(strict_positive_boundary_method(weak))
... except ResearchError as error:
...     boundary_report = error.report
>>> boundary_report.explain("unattainable_zero").verdict
'unresolved'
>>> boundary_report.explain("unattainable_zero/feller_condition").verdict
'strictly false'

A failed sufficient premise makes this boundary rule inapplicable. It does
not refute the pricing formula and does not imply that rates become negative.

To add and remove only the Feller assumption, use an immutable branch and
record the reason for its removal. Leaving ``theta`` symbolic makes its role
visible:

>>> from examples.research_models.cir_bond import with_feller
>>> earlier = cir_case(theta=None)
>>> bounded = with_feller(earlier)
>>> bounded.run(strict_positive_boundary_method(bounded)).data.kind
'trusted stochastic boundary rule'
>>> feller_condition = bounded.assumptions[-1]
>>> relaxed = bounded.without_assumptions(
...     feller_condition, reason="Compare pricing without the optional boundary condition.")
>>> len(earlier.assumptions), len(bounded.assumptions), len(relaxed.assumptions)
(0, 1, 0)
>>> "Removed assumption" in relaxed.changes[-1]
True
>>> try:
...     relaxed.run(strict_positive_boundary_method(relaxed))
... except ResearchError as error:
...     relaxed_report = error.report
>>> relaxed_report.explain("unattainable_zero").verdict
'conditionally true'
>>> price(relaxed).report.explain("affine_valuation").verdict
'strictly true'

The earlier and bounded cases retain their original assumptions. The relaxed
case records the removal while preserving the positive parameter and
nonnegative initial-state domains. Pricing and nonnegative-path evidence
survive; the stronger boundary conclusion again needs its sufficient condition.
Each case-aware method factory uses ``Method.bind(case)``. Rebuild a method
after changing assumptions, observations, or any other part of its immutable
case context.
``run_demo()`` returns these raw cases, results, and blocked-method reports
alongside the two fully observed parameter regimes, zero-rate valuation, and
the terminal-payment case.

Independent numerical observations
----------------------------------

``numerical_method`` integrates the two Riccati equations from zero using
SciPy's DOP853 solver. It also quadratures the numerical :math:`B` curve to
check :math:`\log A=-\kappa\theta\int_0^\tau B(s)\,ds`. The ODE right-hand
side does not evaluate the closed coefficient formula. Diagnostics retain
tolerances, function evaluations, SciPy's version, coefficient/price
differences, and the quadrature error estimate.

These comparisons are labelled numerical observations. They do not prove
error bounds, pathwise positivity, the Feller theorem, or a general
Feynman--Kac theorem. Zero volatility is outside this example's positive
diffusion-parameter domain; no deterministic-limit shortcut is silently used.
