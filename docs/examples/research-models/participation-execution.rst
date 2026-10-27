Constant participation execution
================================

``examples.research_models.participation_execution`` reproduces the scalar
execution problem in Olivier Guéant, `Execution and block trade pricing with
optimal constant rate of participation <https://arxiv.org/html/1210.7608>`_, Section 2.1,
Propositions 1--4. It varies participation caps, completion deadlines, risk
aversion, and the execution-cost component while retaining evidence for every
method's requirements.

Run the exact examples with the core SymPy dependency:

.. code-block:: console

   $ uv run --locked python -m examples.research_models.participation_execution

Install the research extra to compare independent SciPy minimization and
inventory-risk quadrature:

.. code-block:: console

   $ pip install 'modello[research]'
   $ python -m examples.research_models.participation_execution --numerical

The Python ``compare(numerical=True)`` function returns the original cases,
``ResearchResult`` objects, and reports explaining blocked computations. It
does not replace exact certificates with numerical samples. Importing the
module loads no NumPy, SciPy, or plotting packages.

The precise optimization problem
--------------------------------

The paper assumes a positive initial inventory :math:`Q`, continuous positive
deterministic market volume, Brownian price volatility :math:`\sigma>0`, CARA
risk aversion :math:`\gamma>0`, and an increasing, strictly convex execution
function :math:`L` with :math:`L(0)=0` and superlinear growth. Its POV strategy
trades at :math:`v_t=\rho V_t` until liquidation, with one constant
:math:`\rho>0`. Market volume excludes the trader's own trades, so participation
need not be bounded by 1.

Proposition 4 specializes to constant :math:`V_t=V>0` and
:math:`L(\rho)=\eta\rho^{1+\phi}`, where :math:`\eta,\phi>0`. The completion time,
remaining inventory before completion, and certainty-equivalent cost are

.. math::

   T(\rho)&=\frac{Q}{\rho V},\qquad
   q_t=Q-\rho Vt,\quad 0\leq t\leq T(\rho),\\
   J(\rho)&=Q\frac{L(\rho)}{\rho}
       +\frac{\gamma\sigma^2}{2}\int_0^{T(\rho)}q_t^2\,dt
     =\eta Q\rho^\phi+\frac{\gamma\sigma^2Q^3}{6V\rho}.

This example omits permanent impact and spread. The paper's linear permanent
impact :math:`k` and fixed per-share spread :math:`\psi` add
:math:`kQ^2/2+\psi Q`, independent of participation; Propositions 1--2 explain
why they do not alter the minimizing rate. The example reports the variable
cost :math:`J`, not total cash proceeds or a full block-trade price.

Units are shares for :math:`Q`, shares/time for :math:`V`, currency/share for
:math:`\eta`, currency/share/sqrt(time) for :math:`\sigma`, and inverse currency
for :math:`\gamma`. Participation and :math:`\phi` are dimensionless. Rates,
volumes, and volatility are constant over the liquidation horizon.

The unconstrained solution is

.. math::

   \rho^*=\left(\frac{\gamma\sigma^2Q^2}
                        {6\eta\phi V}\right)^{1/(1+\phi)}.

This is optimal among the scalar family of constant-participation POV
strategies. It does not establish optimality among arbitrary time-dependent
execution schedules. The existing
:doc:`portfolio execution example <../portfolio/index>` studies
a different, quadratic fixed-horizon scheduling problem.

Check a solution and its evidence
---------------------------------

The defaults are exact: :math:`Q=3,V=4,\eta=1,\phi=1/2,\gamma=1/6,\sigma=1`.

>>> from examples.research_models.participation_execution import (
...     participation_case, analytic_method, numerical_method, with_limits,
... )
>>> from sympy import S
>>> case = participation_case()
>>> result = case.run(analytic_method(case))
>>> result.data.rate, result.data.duration, result.data.cost
(1/4, 3, 9/4)
>>> result.report.explain("inventory_risk_integral").verdict
'strictly true'
>>> result.data.certificate.explain("pov_kkt_certificate").verdict
'strictly true'

The method checks the live objective, inventory, completion-time, and risk
integral identities. The returned ``ParticipationSolution.certificate`` checks
the scaled first-order condition, completion boundary, feasibility, multiplier
signs, stationarity, and complementary slackness at the proposed rate.

For :math:`0<\phi<1`, :math:`J` need not be convex in :math:`\rho`. In the
default case :math:`J''(1)=-3/8`. Two valid uniqueness arguments are retained:

.. math::

   \rho^2J'(\rho)
       &=\eta Q\phi\rho^{\phi+1}-\frac{\gamma\sigma^2Q^3}{6V},\\
   \frac{d^2}{dx^2}J(e^x)
       &=\eta Q\phi^2e^{\phi x}
                      +\frac{\gamma\sigma^2Q^3}{6V}e^{-x}>0.

The scaled slope is strictly increasing, and the log-rate objective is
strictly convex on the real line. The registered ``uniqueness_theorem`` checks
the transformation and curvature identities before its C2 and
positive-curvature premises. Its conclusion is at most
one attained minimum; the candidate's checked KKT certificate establishes
attainment in each feasible minimum case. A citation alone supplies no proof.

Add and remove constraints
--------------------------

A cap restricts :math:`\rho\leq\rho_{\max}`. A completion deadline imposes
:math:`\rho\geq\ell=Q/(VT_{\max})>0`. If both are active, feasibility requires
:math:`\ell\leq\rho_{\max}`. For positive risk aversion the solution is

.. math::

   \widehat\rho=\min\{\rho_{\max},\max\{\ell,\rho^*\}\},

with the missing bounds omitted. The lower endpoint remains open at zero
when there is no deadline. Limits are live model components, and changing a
component rechecks the same method's requirements.

>>> capped = with_limits(case, cap=S.One/8)
>>> answer = capped.run(analytic_method(capped)).data
>>> answer.rate, answer.duration, answer.active_bounds
(1/8, 6, ('cap',))
>>> deadline = with_limits(case, deadline=2)
>>> answer = deadline.run(analytic_method(deadline)).data
>>> answer.rate, answer.duration, answer.active_bounds
(3/8, 2, ('deadline',))
>>> restored = with_limits(capped)
>>> restored.run(analytic_method(restored)).data.rate
1/4
>>> capped.run(analytic_method(capped)).data.rate
1/8

Removal disables the bound and preserves the earlier observations. Cases are
immutable, so the original constrained case remains available. To change an
already observed numeric cap or deadline, branch from the earlier case;
observations accumulate rather than overwrite one another. A cap of ``1/8``
with deadline ``2`` is infeasible and blocks both methods with a checked
``feasible_limits`` failure. Equal lower and upper bounds are supported.

The certificate uses constraints :math:`\ell-\rho\leq0` and
:math:`\rho-\rho_{\max}\leq0`, with stationarity
:math:`J'(\widehat\rho)-\lambda_\ell+\lambda_u=0`. A missing bound has zero
multiplier. ``active_bounds`` records equality with an enabled bound; an equal
bound can have a zero multiplier.

Multiplying stationarity by the positive candidate rate converts it to
stationarity for :math:`x=\log\rho`, with multipliers
:math:`\widehat\rho\lambda_\ell` and :math:`\widehat\rho\lambda_u`.
The active endpoints preserve complementary slackness. Convex KKT sufficiency
therefore applies in log-rate even when :math:`J` is not convex in rate.

Risk-neutral endpoint behavior
------------------------------

``gamma=0`` extends the model to risk-neutral expected-cost minimization. It
does not maximize the original exponential utility at zero, which would be
constant. Without a deadline, :math:`J(\rho)=\eta Q\rho^\phi>0` has infimum
zero as :math:`\rho\downarrow0`; no admissible finite optimizer attains it.
A participation cap alone does not repair nonattainment.

>>> neutral = participation_case(gamma=0)
>>> answer = neutral.run(analytic_method(neutral)).data
>>> answer.status, answer.rate, answer.duration, answer.cost
('infimum_not_attained', None, None, 0)
>>> urgent = participation_case(gamma=0, deadline=2)
>>> urgent.run(analytic_method(urgent)).data.rate
3/8

``symbolic_case()`` exposes the unobserved declaration. Parameter assumptions
can establish its positive-risk or zero-risk branch. The methods reject
assumptions restricting the rate itself; express a cap or deadline with
``with_limits`` so the feasible interval and KKT conditions stay explicit.
Unsupported symbolic comparisons leave the certificate unestablished and
raise ``ResearchError`` with the case and report.

Replace the impact function
---------------------------

``SurchargedPowerImpact`` changes the execution function to
:math:`L(\rho)=\eta\rho^{1+\phi}+c\rho^2`, with :math:`c\geq0`. The inherited
shape requirements follow the replacement, and strict log-rate convexity
still holds. The pure-power identity fails when :math:`c>0`, blocking the
analytic shortcut; the independent numerical method remains applicable.

.. code-block:: python

   from examples.research_models.participation_execution import SurchargedPowerImpact

   changed = case.replace("impact", SurchargedPowerImpact).observe(
       impact=SurchargedPowerImpact("surcharge", surcharge=S.One/4)
   )
   numerical = changed.run(numerical_method(changed))
   # rate approximately 0.2174036176; the earlier case still has rate 1/4.

The numerical method minimizes the actual declared objective in log-rate. It
finds a derivative sign bracket without using the closed-form optimum, then
uses bounded scalar minimization or the appropriate feasible endpoint. It
independently integrates squared inventory with SciPy quadrature, compares the
resulting cost, and reports a numerical KKT defect. The default ``xatol`` is
``1e-12``. These diagnostics are observations, not certified error bounds.
The risk-neutral open-endpoint case is classified by checked symbolic limits
instead of returning a small positive rate as a supposed optimizer.

API
---

.. automodule:: examples.research_models.participation_execution
   :members: participation_case, symbolic_case, with_limits, analytic_method, numerical_method, compare, ParticipationExecution, PowerLawImpact, SurchargedPowerImpact, ParticipationSolution, NumericalParticipation
   :member-order: bysource
