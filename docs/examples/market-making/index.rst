Market making with component requirements
=========================================

When a model consumes utility, risk, and liquidity components, their shapes
can matter as much as their values. An optimizer might need decreasing trade
intensity to obtain a unique quote, or two derivatives to evaluate curvature.
This example makes those obligations explicit before solving a trading model.

The starting point is the continuous-time market-making framework near the
end of Olivier Guéant's *The Financial Mathematics of Market Liquidity: From
Optimal Execution to Market Making* (2016), chapter 11, especially sections
11.2--11.3. See the `publisher's contents and preview
<https://www.routledge.com/The-Financial-Mathematics-of-Market-Liquidity-From-Optimal-Execution-t/Gueant/p/book/9781498725477>`_.
For publicly accessible equations, this implementation follows the single-asset
CARA formulation, Model A, in Guéant's `Optimal market making
<https://arxiv.org/html/1605.01862>`_, sections 2--3 and Lemma 3.1.

Time is continuous; inventory changes in discrete lots. The lightweight runner
represents one state of the reduced HJB and its exact exponential-intensity
quote optimizer. The complete :doc:`research workflow </contrib/research>` adds
live contracts, checked theorem premises, the coupled terminal-value ODEs, and
an exponential/logistic comparison.

Run it from the repository:

.. code-block:: console

   $ uv run --locked --dev python -m examples.market_making

Run the complete study and open its report:

.. code-block:: console

   $ uv run --locked --extra research python -m examples.market_making --study --output artefacts/market-making

An HTML documentation build also generates a `complete study report
<study/report.html>`_. Its evidence pages, CSV tables, JSON audit, and figures
are available offline from the same directory.

The model and its requirements
-----------------------------------

The reference price follows :math:`dS_t=\sigma\,dW_t`. Buying or selling one
lot changes inventory by :math:`\Delta`, and cash by the corresponding traded
value. Quotes are distances from the reference price:
:math:`S^b=S-\delta^b`, :math:`S^a=S+\delta^a`.
The control objective is

.. math::

   \sup_{\delta^b,\delta^a}
   \mathbb{E}\!\left[-e^{-\gamma(X_T+q_TS_T-\ell(q_T))}\right].

There is no additional running risk penalty in this objective. The quadratic
term below comes from Brownian price risk after the CARA change of variables.
Here :math:`\ell(q)=\eta q^2` specializes the paper's :math:`\ell(|q|)`.

``CARAUtility``: :math:`U(w)=-e^{-\gamma w}`
   Requires :math:`\gamma>0`, positive slope, concavity, and :math:`C^2` on
   :math:`\mathbb R`; the reduction uses price curvature.

``ExponentialIntensity``: :math:`\Lambda(\delta)=Ae^{-k\delta}`
   Requires a positive :math:`C^2` rate with :math:`\Lambda'<0` and a
   vanishing right tail; :math:`A,k>0`. A further uniform bound,
   :math:`\sup_\delta\Lambda\Lambda''/(\Lambda')^2<2`, is used to establish
   the unique optimizer. The ratio is exactly 1 for exponential intensity.

``QuadraticInventoryRisk``: :math:`R(q)=\gamma\sigma^2q^2/2`
   Nonnegative, convex, and :math:`C^2` in signed inventory. This reuses
   ``InventoryRisk`` from the optimal-execution example.

``QuadraticTerminalPenalty``: :math:`\ell(q)=\eta q^2`, :math:`\eta\geq0`
   Nonnegative, convex, and :math:`C^2`. It need not increase on all signed
   inventories: both long and short positions carry a cost.

``ContinuationValue``: :math:`\theta(t,q)`, :math:`\theta(t,q\pm\Delta)`
   Also supplies :math:`\partial_t\theta(t,q)`. The theoretical solution is
   :math:`C^1` in time. Neighboring inventory values enter through finite
   differences, not inventory derivatives.

Let :math:`q\in\{-Q,-Q+\Delta,\ldots,Q\}`, with :math:`Q/\Delta` a positive
integer. Inserting
:math:`u(t,x,q,S)=-e^{-\gamma(x+qS+\theta(t,q))}` into the HJB gives

.. math::

   0 = -\partial_t\theta(t,q)+R(q)
       -\mathbf 1_{q<Q}H^b(p^b)-\mathbf 1_{q>-Q}H^a(p^a),
   \qquad \theta(T,q)=-\ell(q),

   p^b=\frac{\theta(t,q)-\theta(t,q+\Delta)}{\Delta},\qquad
   p^a=\frac{\theta(t,q)-\theta(t,q-\Delta)}{\Delta}.

Each side has its own intensity parameters. For exponential intensity, its
Hamiltonian and maximizing quote distance are

.. math::

   H(p)=\sup_{\delta\in\mathbb R}
         \frac{Ae^{-k\delta}}{\gamma}
         \left(1-e^{-\gamma\Delta(\delta-p)}\right),
   \qquad
   \delta^*(p)=p+\frac{\log(1+\gamma\Delta/k)}{\gamma\Delta}.

``ExponentialQuote.value`` substitutes this optimizer into the gain function;
``MarketMaker`` connects the two sides, shared risk aversion, lot size, and
continuation values. The two inventory gates suppress trades that would
increase inventory past a boundary. ``terminal_value`` evaluates the negative
terminal penalty at the selected state inventory.

.. literalinclude:: ../../../examples/market_making.py
   :pyobject: QuoteHamiltonian
   :language: python

.. literalinclude:: ../../../examples/market_making.py
   :pyobject: ExponentialQuote
   :language: python

These formulas use real quote distances, including negative ones. Adding
nonnegative quote constraints or switching to risk-neutral utility requires
changing the optimizer. The example assumes the stated inventory lattice;
it does not enforce lattice membership or supply a stochastic simulator.

Check the component definitions
-----------------------------------

``checked`` is an example decorator: after the class body, it constructs the
component with symbolic parameters and calls an ordinary requirement function.
It stores a ``Requirements`` annotation through Modello's existing metadata
API. Nested composition carries the reports to their component paths:

>>> from examples.market_making import MarketMaker, Requirements
>>> reports = Requirements.for_model(MarketMaker)
>>> sorted(reports)
['ask.intensity', 'bid.intensity', 'risk', 'terminal', 'utility']
>>> all(check.verdict == "strictly true" for report in reports.values() for check in report.checks)
True

A check has one of the requested three truth classifications, or an explicit
unresolved result when no proof or parameter condition has been established:

``strictly true``
   Established over the declared real domain and parameter assumptions.

``conditionally true``
   The check reduces to an explicit condition on parameters, such as
   :math:`\gamma>0`; the condition is retained in ``condition``.

``strictly false``
   The stated global certificate fails for this definition.

``unresolved``
   The checker has not established a result; ``condition`` is ``None``.

For example, relax the declaration of risk aversion before connecting it to
any consumer. Reapply the decorator after changing a component definition:

>>> from examples.market_making import CARAUtility, checked, utility_requirements
>>> from modello import InstanceDummy
>>> @checked(utility_requirements)
... class UnrestrictedUtility(CARAUtility):
...     gamma = InstanceDummy("gamma", real=True)
>>> checks = Requirements.for_model(UnrestrictedUtility)[""].checks
>>> checks[0].verdict, checks[2].verdict
('conditionally true', 'conditionally true')
>>> condition = checks[0].condition
>>> gamma = next(iter(condition.free_symbols))
>>> condition == (gamma > 0)
True
>>> utility_requirements(UnrestrictedUtility("constant", gamma=0))[2].verdict
'strictly false'

The functions check whole real domains. A positive derivative at one sampled
wealth does not establish monotonicity. ``positive_slope`` requires a :math:`C^1`
function with positive derivative everywhere, a sufficient condition for strict
increase. It is deliberately stronger than strict increase itself: :math:`w^3`
fails this certificate at zero. ``convex`` uses a nonnegative second derivative
and :math:`C^2` regularity, so it is also a smooth-function certificate.

``smooth`` asks SymPy for the continuity domains of the expression and its
derivatives. Poles fail; unsupported functions and some nonsmooth expressions
remain unresolved. Parameter-only subexpressions must also be known finite:
for example, :math:`e^{w/a}` with unrestricted real :math:`a` remains unresolved
because :math:`a=0` is allowed. ``uniform_curvature_bound`` retains the condition
that the derivative never vanishes before simplifying the ratio. It only
certifies ratios independent of the quote coordinate; a pointwise inequality
does not establish a uniform supremum bound. In particular, the existence of a
formal derivative does not establish differentiability or continuity:

>>> from sympy import Function, Symbol
>>> from examples.market_making import smooth
>>> t = Symbol("t", real=True)
>>> smooth(Function("theta")(t), t, 1) is None
True

Reports describe each declared component family independently. They are not
an enforcement hook, do not add assumptions to the solver, and do not refresh
after subclassing or observations. Call the requirement function again for a
new case; redecorate a changed subclass. Conditional and failed definitions
remain available for exploration. Substituting another utility family would
also require reconsidering the CARA ansatz, not just its shape requirements.

Evaluate a state of the HJB
-----------------------------------

With unit risk aversion, lot size, decay, and volatility, and arrival scale 2
on both sides, zero neighboring continuation values give :math:`p^b=p^a=0`.
The optimizer is :math:`\log 2` and each Hamiltonian is :math:`1/2`:

>>> from examples.market_making import (
...     ContinuationValue, ExponentialIntensity, ExponentialQuote, QuadraticInventoryRisk,
... )
>>> quote = ExponentialQuote(
...     "quote", intensity=ExponentialIntensity("liquidity", scale=2, decay=1),
...     gamma=1, step=1, cost=0,
... )
>>> quote.optimal_distance, quote.value
(log(2), 1/2)
>>> maker = MarketMaker(
...     "interior", gamma=1, step=1, capacity=1, inventory=0, residual=0,
...     risk=QuadraticInventoryRisk("risk", sigma=1),
...     continuation=ContinuationValue("theta", current=0, after_bid=0, after_ask=0),
...     bid=quote, ask=quote,
... )
>>> maker.continuation.time_derivative
-1

Observing ``residual=0`` infers the time derivative from the other quantities.
This calculation verifies one equation; it does not establish that arbitrary
continuation values solve the terminal-value problem.

Continue with :doc:`/contrib/research` for immutable cases and live evidence.
The full study preserves this HJB declaration, replaces the specialized quote
components with general Hamiltonian inputs, and integrates forward in remaining
time. The exponential case has an independent matrix-exponential reference;
both families have tighter-tolerance and independent time-difference checks.
The default comparison matches the rate and slope of the two intensities at
zero distance, so their differences reflect the changed shape.
