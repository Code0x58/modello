Perpetual American put and optimal stopping
===========================================

This example uses the constant-coefficient Black--Scholes--Merton economic
setting with positive interest, volatility, and strike, and zero dividends.
It follows Robert C. Merton, `Theory of Rational Option Pricing (1973),
Section 8, pages 173--174, equations (49)--(52)
<https://finance.martinsewell.com/option-pricing/Merton1973.pdf#page=34>`_.
The scalar calculations, numerical barrier search, and explicitly trusted
stochastic verification rule are separate results.

.. code-block:: console

   $ uv run --locked python -m examples.research_models.perpetual_put
   $ uv run --locked --dev python -m examples.research_models.perpetual_put --numerical

Importing the module requires only SymPy. The numerical method imports SciPy
inside its execution; it is available through ``modello[research]``.

Normalize the exercise boundary
-------------------------------

Under the risk-neutral measure, assume
:math:`dS_t=rS_t\,dt+\sigma S_t\,dW_t`, with :math:`r,\sigma,K>0` and no
dividends. Define

.. math::

   \beta=-\frac{2r}{\sigma^2}<0,\qquad
   B=\frac{\beta K}{\beta-1}=\frac{2rK}{2r+\sigma^2},\qquad x=S/B.

The proposed value is

.. math::

   P(Bx)=\begin{cases}
       K-Bx,&0<x\leq1,\\
       (K-B)x^\beta,&x>1.
   \end{cases}

Merton uses :math:`\gamma=2r/\sigma^2=-\beta` and names the exercise boundary
:math:`C^*`. The implementation retains the fixed domains ``(0,1]`` and
``(1,infinity)`` instead of asking the checker to quantify over a moving,
parameter-dependent boundary.

The defaults are :math:`r=1/25,\sigma=1/5,K=100,S_0=100`:

>>> from examples.research_models.perpetual_put import (
...     perpetual_put_case, algebra_method, c2_consumer_method,
...     trust_stopping_theorem, verification_method,
... )
>>> case = perpetual_put_case()
>>> algebra = case.run(algebra_method)
>>> algebra.data.beta, algebra.data.boundary, algebra.data.value_at_initial_spot
(-2, 200/3, 400/27)
>>> algebra.report.explain("continuation_pricing_generator").verdict
'strictly true'

The checked scalar conditions
-----------------------------

The live contracts protect the exponent, boundary, normalized spot, branch
values, derivatives, piecewise assembly, and generator formulas. Replacing a
helper by a constant cannot silently provide a stale certificate. The model's
discounted pricing generator is

.. math::

   (\mathcal L-r)f(x)
       =\tfrac12\sigma^2x^2f''(x)+(r-q)xf'(x)-rf(x).

The zero-dividend contract requires :math:`q=0`. The continuation residual is
zero; the exercise residual is :math:`-rK<0`. The method checks
:math:`0\leq P\leq K`, equality with the exercise payoff below the boundary,
and continuation dominance over both zero and :math:`K-S` above it. It also
checks decay at infinity and the deterministic transversality envelope
:math:`K e^{-rt}\to0`.

For the continuation obstacle inequality, a separate, elementary trusted
calculus lemma uses the gap :math:`h(x)=P(Bx)-(K-Bx)`. Its checked premises are
C2 regularity on :math:`[1,\infty)`, nonnegative curvature, and
:math:`h(1)=h'(1)=0`. The convex tangent inequality then gives :math:`h(x)\geq0`
for :math:`x>1`. These premises are recursive evidence in the report; no grid
of numerical samples substitutes for the global inequality.

Smooth fit does not give C2
---------------------------

The certificate takes actual one-sided limits of the declared analytic
branches and converts derivatives in :math:`x` to spot derivatives by dividing
by :math:`B` or :math:`B^2`:

>>> certificate = algebra.data
>>> certificate.one_sided["value_left"], certificate.one_sided["value_right"]
(100/3, 100/3)
>>> certificate.one_sided["delta_left"], certificate.one_sided["delta_right"]
(-1, -1)
>>> certificate.one_sided["gamma_left"], certificate.one_sided["gamma_right"]
(0, 9/200)

Thus the two smooth branches join with a continuous first derivative, while
the second derivative jumps. In general the right gamma at the boundary is
:math:`2rK/(\sigma^2B^2)>0`, whereas the left gamma is zero.

``certificate.boundary_report`` retains the checked positive jump.
``certificate.regularity`` now establishes C1 and refutes C2 through a narrow
scalar rule for two branches with one finite constant join:

>>> certificate.regularity.explain("global_c1").verdict
'strictly true'
>>> c2 = certificate.regularity.explain("global_c2")
>>> c2.verdict, tuple(value for _, value in c2.witness)
('strictly false', (1,))

The witness is normalized moneyness :math:`x=1`, corresponding to :math:`S=B`.
The rule checks regularity on each active open side, the value actually
assigned at the join, and finite one-sided derivative limits. A matched first
derivative and an unequal second derivative give the two different verdicts;
they do not use a registered stochastic theorem. Unsupported multiple or
parameter-dependent joins remain unresolved. At an included domain endpoint,
one-sided continuity alone cannot discharge an ordinary derivative premise.
The separate exact certificate still exports derivatives in spot units.
``c2_consumer_method()`` demands global C2 and is blocked. ``algebra_method``
removes that excessive consumer restriction and instead checks piecewise
regularity, value matching, and smooth fit.

Where stochastic verification enters
------------------------------------

A PDE residual alone does not prove optimal stopping. The ordinary case has
no rule establishing ``optimal_stopping_under_GBM``. Adopt it explicitly:

>>> trusted = trust_stopping_theorem(case)
>>> verification = trusted.run(verification_method)
>>> verification.report.explain("optimal_stopping_under_GBM").kind
'theorem'

The trusted setting is fully retained in
``verification.data.trusted_assumptions`` and in the report's conclusion reason.
The stopping objective is
:math:`\sup_\tau\mathbb E_Q[e^{-r\tau}(K-S_\tau)^+]`:

* The filtered probability space satisfies the usual conditions; under the
  risk-neutral measure, the stock solves the stated GBM equation with standard
  Brownian motion, positive initial stock, and constant positive parameters.
* Exercise is immediate and has no fees. Admissible decisions are all stopping
  times in :math:`[0,\infty]` for that filtration. An infinite stopping time has
  discounted payoff zero.
* The trusted proof uses generalized Itô for a C1, piecewise C2 candidate with
  locally bounded second derivatives, and localized optional sampling. Smooth
  fit removes the boundary local-time term.
* The checked bound :math:`0\leq P\leq K` gives integrability and bounded
  stopped discounted values. The envelope :math:`K e^{-rt}\to0` supplies the
  terminal limit on paths that have not exercised.

Under that rule, the generator inequality and obstacle dominance give an
upper bound for every admissible stopping time. Equality is attained by
:math:`\tau_B=\inf\{t\geq0:S_t\leq B\}`, with infinity allowed when the barrier
is never reached. Brownian dynamics, Itô, and optional sampling are trusted
mathematical inputs, rather than discoveries made by the scalar CAS checker.

Change a parameter or the consumer
----------------------------------

``perpetual_put_case(rate=0)`` cannot reuse the positive-parameter shortcut:
the positive-rate and positive-boundary premises fail, and the envelope no
longer vanishes. The example records this rejection; it does not obtain a
zero-rate price by substituting into a formula whose normalization requires
:math:`B>0`. Nonzero dividends likewise invalidate this exponent and are
blocked by the exact zero-dividend and generator checks.

``case.run(algebra_method)`` builds the method for the current immutable branch.
Observe ``initial_spot`` to value an initial state;
observations or assumptions fixing the normalized curve itself are rejected
because verification traverses both regions.
The restriction check includes active SymPy ``assuming(...)`` contexts as well
as explicitly recorded ``case.assume(...)`` predicates.
Both moneyness and the time coordinate of the transversality envelope must
remain unrestricted.

Independent stopping-barrier optimization
-----------------------------------------

For a down-crossing barrier :math:`0<b\leq\min(S_0,K)`, the discounted hitting
payoff in the stated GBM setting is

.. math::

   H(b)=(K-b)(S_0/b)^\beta.

``numerical_method(case)`` maximizes this function with bounded SciPy scalar
optimization in :math:`b/K`. The bracket and objective use the admissible
barrier domain; the analytic boundary is used only afterward for comparison.
The method also checks the immediate-exercise endpoint when :math:`S_0<B`.
It reports barrier and value differences, with default ``xatol=1e-12``.
These are numerical observations over barrier strategies. They neither prove
an error bound nor replace verification over all stopping times.

``run_demo(numerical=True)`` returns the raw cases, scalar and trusted results,
blocked-method reports, and numerical comparisons for continuation and
immediate exercise.

API
---

.. automodule:: examples.research_models.perpetual_put
   :members: perpetual_put_case, symbolic_case, algebra_method, c2_consumer_method, trust_stopping_theorem, verification_method, numerical_method, regularity_report, run_demo, PerpetualPut, PutCertificate, StoppingVerification, BarrierComparison
   :member-order: bysource
