Black--Scholes--Merton: carry, Greeks, and a nonsmooth boundary
===============================================================

This study composes a European call and put with deterministic integrated
interest, dividend yield, and variance. It checks exact identities, replaces a
constant schedule with an affine schedule, rejects an obsolete dividend-free
shortcut, and treats the zero-variance payoff separately.

Run the complete example from the checkout:

.. code-block:: console

   $ uv run --locked --dev python -m examples.research_models.black_scholes
   constant: 18 exact zero residuals; report {'strictly true': 56}
   affine: 18 exact zero residuals; report {'strictly true': 56}
   constant: call=9.3197384552, put=6.3936989955, delta=0.5927490982, gamma=0.0191410352
   ...
   Dividend-free shortcut after adding yield: strictly false

``--symbolic-only`` skips the independent SciPy quadrature. Importing the module
uses SymPy alone; the numerical method imports SciPy when executed. The exact
inputs are rational numbers, so the symbolic results do not depend on floating
point tolerances.

The reusable kernel
-------------------

``IntegratedCarry`` supplies three real scalar inputs:

.. list-table:: Deterministic tail integrals
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Symbol
     - Interpretation
   * - ``rate_total``
     - :math:`R`
     - :math:`\int_t^T r(u)\,du`
   * - ``yield_total``
     - :math:`Q`
     - :math:`\int_t^T q(u)\,du`
   * - ``variance_total``
     - :math:`W`
     - :math:`\int_t^T \sigma(u)^2\,du`

``EuropeanOptions`` shares these inputs, spot :math:`S`, and strike :math:`K`
between both options. With :math:`\Phi` the standard normal CDF, define

.. math::

   d_1 = \frac{\log(S/K)+R-Q+W/2}{\sqrt W},\qquad d_2=d_1-\sqrt W,

   C=S e^{-Q}\Phi(d_1)-K e^{-R}\Phi(d_2),\qquad
   P=K e^{-R}\Phi(-d_2)-S e^{-Q}\Phi(-d_1).

The continuous-dividend formula and parity are described in
`Martin Haugh's Columbia lecture notes, sections 1.2 and 3
<https://www.columbia.edu/~mh2078/FoundationsFE/BlackScholes.pdf>`_. The present
example extends the scalar carry inputs to deterministic tail integrals and
checks the resulting algebra explicitly. It does not cover discrete cash
dividends, stochastic rates or volatility, early exercise, or transaction costs.

All input symbols are declared real. ``Requires`` records :math:`K>0` and
:math:`W>0` as live premises, rather than building positivity into their symbol
declarations. Spot claims quantify the explicit open domain
:math:`S\in(0,\infty)`. Observing ``spot=100`` selects a valuation state;
``case.expression("pricing.call", wrt="pricing.spot")`` still retains the
entire spot curve. The method also checks that the observed spot is positive.
No interior contract includes :math:`S=0` or :math:`W=0`.

Exact checks and their trust boundary
-------------------------------------

``symbolic_method(case)`` differentiates the current declared formulas and
checks zero residuals for parity and both options' delta and gamma:

.. math::

   C-P=S e^{-Q}-K e^{-R},\qquad
   C_S=e^{-Q}\Phi(d_1),\qquad P_S=e^{-Q}(\Phi(d_1)-1),

   C_{SS}=P_{SS}=\frac{e^{-Q}\phi(d_1)}{S\sqrt W}>0.

The audit first restricts the real scalar spot input to a positive symbolic
variable. This implements the stated open spot domain and discharges the
retained nonzero-spot division guards before differentiation. It does not
remove the original guards or extrapolate through zero. The audit also derives
the Gaussian density balance
:math:`S e^{-Q}\phi(d_1)=K e^{-R}\phi(d_2)` and checks gamma positivity.

The research checker currently cannot establish the continuity domain of
``erf`` compositions by itself. Without an added rule, the C2 claim is
``unresolved``. Each study case therefore registers a narrowly scoped trusted
composition theorem. Its checked premises include the exact declared option
formula, positive strike and total variance, and C2 regularity of both Gaussian
arguments and elementary discount factors on the spot domain. Separate source
contracts link those arguments and factors to the actual primitive inputs;
smoothness of a field with a familiar name would not establish that identity.
`NIST DLMF section 7.2(i) <https://dlmf.nist.gov/7.2#i>`_ supplies the mathematical
fact that the error function is entire. The theorem and SymPy are part of the
trust boundary; the source citation alone is not a proof. The rule supplies no
regularity conclusion for arbitrary unknown functions or for the maturity
payoff.

The formulas, Greek forms, and parity are live requirements. ``d1``, ``d2``,
``discounted_spot``, and ``discounted_strike`` each have an exact source
contract, and ``call_minus_put`` must equal the actual declared call minus put.
Both adapters consume these contracts explicitly. A subclass that compensates
for several incorrect helpers to retain correct final prices still fails the
helper contracts. Redefining an inherited field refreshes its requirements; a
copied earlier verdict cannot certify the replacement.

A changing deterministic schedule
---------------------------------

``OptionStudy`` keeps a scalar pricing snapshot and a separate schedule
component. Each method checks that :math:`R,Q,W` match the schedule integrals at
the observed ``now``. Keeping the schedule separate also makes the quantified
domains explicit: spot differentiation uses :math:`(0,\infty)`, whereas the
schedule uses :math:`t\in[0,1)` and has normalized maturity :math:`T=1` year.
Instantaneous variance is required to be nonnegative on :math:`[0,1]`, and
integrated variance is required to be positive before maturity.

Both the symbolic and numerical adapters check a common sufficient fundamental
theorem of calculus scope: the three coefficient curves and their proposed
tail integrals must be C1 on the closed interval :math:`[0,1]`. They also check
the exact derivative identities on :math:`[0,1)` and the terminal conditions
:math:`R(1)=Q(1)=W(1)=0`. These conditions identify the tail integrals, including
their integration constants. The adapter accepts other supported C1 schedules;
it does not require their coefficients to be affine.

The baseline has :math:`r=0.04`, :math:`q=0.01`, and
:math:`\sigma^2=0.04`. Replacing ``ConstantSchedule`` with ``AffineSchedule``
exposes previously zero slopes as parameters. The example uses

.. math::

   r(t)=0.04+0.01t,\qquad q(t)=0.01+0.005t,\qquad
   \sigma(t)^2=0.04+0.02t.

The integrals are exact polynomials, for example
:math:`R(t)=0.04(1-t)+0.005(1-t^2)`. The checker differentiates each live
integral, verifies :math:`R'=-r`, :math:`Q'=-q`, and :math:`W'=-\sigma^2`, and
evaluates all three totals at maturity. A constant shift preserves those
derivatives and can match a shifted pricing snapshot, but it fails the zero
terminal conditions. Patching only the terminal point to zero does not repair
the resulting discontinuity: the closed-interval regularity contract blocks
that schedule as well.
For each option :math:`V`, an independent integrated-kernel calculation checks

.. math::

   -r V_R-q V_Q-\sigma^2 V_W
   +(r-q)S V_S+\tfrac12\sigma^2 S^2 V_{SS}-rV=0.

Combining these identities with the ordinary deterministic chain rule gives
the PDE residual :math:`V_t+(r-q)SV_S+\frac12\sigma^2S^2V_{SS}-rV=0` for both
the constant and affine schedules. The implementation reduces the integrated
identity before applying the schedule. It avoids asking the CAS to simplify
one enormous expanded time-dependent derivative tree. This is an exact
chain-rule derivation on the interior; it is not an automatic certificate of
joint regularity on a rectangle containing maturity.

For the financial theory behind the PDE and discounted expectations, see
`David Yao's Black--Scholes--Merton notes
<https://www.columbia.edu/~yao/bsnotes04b.pdf>`_ and
`Steven Lalley's Black--Scholes lecture
<https://www.stat.uchicago.edu/~lalley/Courses/390/Lecture7.pdf>`_. Stochastic
integration, self-financing replication, admissibility, and uniqueness of the
pricing measure are external mathematical theory here. A zero PDE residual
does not machine-prove those results.

Branching and replacement
-------------------------

.. code-block:: python

   from sympy import Rational
   from examples.research_models.black_scholes import (
       AffineSchedule, study_case, symbolic_method, valuation_method,
   )

   constant = study_case()
   changing = study_case(schedule=True, now=Rational(1, 4))
   exact = changing.run(symbolic_method)
   reference = changing.run(valuation_method)
   assert all(value == 0 for value in exact.data.residuals.values())
   assert reference.data.call_error < 1e-9  # a numerical comparison

   # This immutable replacement leaves the explicit pricing snapshot in place.
   stale = constant.replace("schedule", AffineSchedule).observe(
       schedule=AffineSchedule("schedule", rate_slope=Rational(1, 100)),
   )
   # stale.run(symbolic_method) raises ResearchError: its totals disagree.

``case.run(symbolic_method)`` and ``case.run(valuation_method)`` build a fresh
method for the current branch, evaluating its schedule at ``now``. Prepared
methods use ``Method.bind(case)`` and reject reuse on a different case, even
when it shares the same problem object. Earlier branches remain unchanged.

``dividend_free_method()`` has the additional exact contract
:math:`C-P=S-Ke^{-R}`. It runs when cumulative dividend yield is zero. Adding the
positive yield in the baseline produces a checked counterexample to that
identity and blocks the method. General discounted parity still holds. Shape
properties or a successful earlier run do not preserve this shortcut.

The boundary is a different calculation
---------------------------------------

At :math:`W=0`, the deterministic discounted call payoff is
:math:`\max(S e^{-Q}-K e^{-R},0)`, with the analogous put. At maturity
:math:`R=Q=0`. ``PayoffBoundary`` uses discounted forward moneyness
:math:`x=S e^{-Q}-K e^{-R}` and writes the two payoffs as
:math:`(x+|x|)/2` and :math:`(-x+|x|)/2`.

``boundary_checks()`` computes exact one-sided slopes at :math:`x=0`: the call
has slopes 0 and 1, and the put has slopes -1 and 0. Consequently neither payoff
is differentiable there, and neither is C2. The research check for the absolute
value kink also refutes C2 regularity. Away from the kink the payoff is affine.
The interior method rejects zero or negative total variance and maturity; it
never substitutes :math:`W=0` into the expressions for :math:`d_1,d_2`.
For singular observations the interior model has no admissible defined state;
the report records inconsistency and leaves claims unresolved instead of
proving them from impossible premises. Scalar aliases capture these exploratory
inputs without first constructing and solving a singular option instance.

The numerical reference
-----------------------

``valuation_method(case)`` integrates the call and put payoffs against a
standard Gaussian density, using
:math:`S_T=S\exp(R-Q-W/2+\sqrt W Z)` and discount :math:`e^{-R}`. It splits the
integrals at the exercise threshold and does not call a normal CDF or reuse
the closed-form option formula in the reference integrand. Numerical results
include the two quadrature differences and the integrator's error estimates.
These finite observations do not establish a global approximation bound or
discharge an unresolved mathematical requirement.

``run_demo()`` retains raw cases, symbolic and numerical method results,
the accepted and rejected shortcut reports, and the boundary slopes. It is
suitable for the collection runner as well as direct interactive inspection.

API
---

.. automodule:: examples.research_models.black_scholes
   :members: IntegratedCarry, EuropeanOptions, ConstantSchedule, AffineSchedule, OptionStudy, PayoffBoundary, SymbolicChecks, NumericalCheck, symbolic_case, study_case, symbolic_method, valuation_method, dividend_free_method, boundary_checks, run_demo
   :member-order: bysource
