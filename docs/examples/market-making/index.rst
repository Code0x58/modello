.. _market-making-with-component-requirements:

Market making with component obligations
=========================================

Choose bid and ask quotes for a single asset to maximize expected utility of
terminal wealth after an inventory penalty. Connect utility, inventory risk,
and liquidity components, then inspect the properties a quote optimizer needs
from them. This page first checks the components and evaluates one state of
the pricing equation. It then solves the coupled inventory equations over
time and compares exponential and logistic trade-arrival intensities.

Time is continuous and inventory changes in discrete lots. The pricing
equation is a Hamilton--Jacobi--Bellman (HJB) equation for continuation values.
The utility has constant absolute risk aversion (CARA), which supplies the
reduction used below. For the context, evidence, and method API, first work through
:doc:`/contrib/verification`; :doc:`../financial-models/markowitz` offers a smaller
scalar example of changing constraints.

With Modello installed, run the one-state symbolic calculation:

.. code-block:: console

   $ python -m examples.market_making

For the full numerical study, install the numerical dependencies and generate
an offline report:

.. code-block:: console

   $ pip install 'modello[numerics]'
   $ python -m examples.market_making --study --output artefacts/market-making

Open ``artefacts/market-making/report.html``. An HTML documentation build also
generates a `complete study report <study/report.html>`_, with evidence pages,
CSV tables, JSON audit, and figures in the same directory.

.. _the-model-and-its-requirements:

The model and its obligations
-----------------------------------

The starting point is the continuous-time market-making framework near the
end of Olivier Guéant's *The Financial Mathematics of Market Liquidity: From
Optimal Execution to Market Making* (2016), chapter 11, especially sections
11.2--11.3. See the `publisher's contents and preview
<https://www.routledge.com/The-Financial-Mathematics-of-Market-Liquidity-From-Optimal-Execution-t/Gueant/p/book/9781498725477>`_.
For publicly accessible equations, this implementation follows the single-asset
CARA formulation, Model A, in Guéant's `Optimal market making
<https://arxiv.org/html/1605.01862>`_, sections 2--3 and Lemma 3.1.

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
   ``InventoryRisk`` from the
   :ref:`fixed-horizon execution study <portfolio-continuous-execution>`.

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
changing the optimizer. The one-state calculation assumes the stated inventory
lattice. The numerical study below checks that capacity is an integral number
of lots. Neither runner supplies a stochastic simulator.

Check the component definitions
-----------------------------------

The components use the shared ``modello.contrib.verification`` interface.
``VerificationModel`` checks their declarations after each class body, including
subclasses. ``specifies`` attaches mathematical obligations to a field or a
model. Nested composition scopes those obligations to their component paths:

>>> from examples.market_making import MarketMaker
>>> report = MarketMaker.definition_report()
>>> report.explain("utility.value/strictly_increasing").verdict
'strictly true'
>>> report.explain("bid.intensity.rate/derivative_ratio_bound").verdict
'strictly true'
>>> all(check.verdict == "strictly true" for check in report.evidence)
True

A check reports one of four verdicts:

``strictly true``
   Established over the declared domain and parameter assumptions.

``conditionally true``
   An explicit sufficient parameter condition is retained in ``condition``.

``strictly false``
   A checked refutation of the stated property.

``unresolved``
   The checker has not established a result; ``condition`` is ``None``.

For example, relax the declaration of risk aversion before connecting it to
any consumer. The subclass automatically refreshes the inherited obligations:

>>> from examples.market_making import CARAUtility
>>> from modello import InstanceDummy
>>> from modello.contrib.verification import VerificationContext
>>> class UnrestrictedUtility(CARAUtility):
...     gamma = InstanceDummy("gamma", real=True)
>>> checks = UnrestrictedUtility.definition_report()
>>> checks.explain("gamma/positive").verdict, checks.explain("value/strictly_increasing").verdict
('conditionally true', 'conditionally true')
>>> condition = checks.explain("gamma/positive").condition
>>> gamma = next(iter(condition.free_symbols))
>>> condition == (gamma > 0)
True
>>> context = VerificationContext.for_model(UnrestrictedUtility)
>>> context.observe(gamma=0).verify().explain("value/strictly_increasing").verdict
'strictly false'

Checks concern whole declared domains; sampled slopes are numerical
observations. The :doc:`verification guide </contrib/verification>` explains supported
regularity arguments, unresolved derivatives, and checked hints for uniform
curvature bounds. Failed or unresolved definitions remain available for
exploration, while methods require the evidence they consume. Substituting
another utility family also requires reconsidering the CARA ansatz and its
exact-form contract.

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

The next step is to preserve this HJB declaration, replace the specialized quote
components with general Hamiltonian inputs, and integrate in remaining time.

Run the time-dependent study
----------------------------

``study_context`` constructs that branch. ``run_quotes`` returns continuation
values, quotes, and diagnostics on a time and inventory grid. Boundary gates
leave the inventory-increasing quotes absent throughout the grid:

>>> from examples.market_making_study import study_context
>>> from examples._market_making_numerics import run_quotes
>>> result = run_quotes(study_context(logistic=True, capacity=1), points=5)
>>> result.data.bid[0][-1] is None, result.data.ask[0][0] is None
(True, True)
>>> result.data.diagnostics["terminal_error"]
0.0

The full runner compares exponential :math:`A=2,k=1` with logistic
:math:`A=4,k=2`, matching rate and slope at zero. Defaults are
:math:`T=1`, :math:`\Delta=1`, :math:`Q=5`, :math:`\gamma=0.1`,
:math:`\sigma=1`, and :math:`\eta=0.05`.

Both sides consume the general Hamiltonian interface. A separate exact-form
contract protects the exponential shortcut; changing intensity does not carry
that formula into the new model. The CARA reduction has its own exact-utility
contract. Additional contracts protect the nearest-neighbor continuation
differences, CARA gain, boundary gates, and HJB structure used by the numerical
adapter. Shape properties alone do not establish these identities.

The solver integrates in remaining time using SciPy Radau with an analytic
tridiagonal Jacobian. The general quote method brackets the unique
first-order-condition root; exponential and logistic expressions use stable
numerical evaluations. Inventory boundary gates come from the shared HJB.
Inactive quotes are absent values in reports. Invalid lattice membership,
nonfinite values, failed bracketing, and solver failures retain diagnostics.
The full ODE requires independent state inputs and parameter observations;
observations or assumptions fixing a state are rejected because the study
traverses the entire inventory and time grid.
These checks include active SymPy ``assuming(...)`` predicates as well as
``context.assume(...)``. Parameter-only assumptions remain supported.
Quote-distance and terminal-inventory coordinates are included in the check;
a predicate on a locally substituted argument cannot narrow the solver's domain.

``run_quotes`` requires at least three output times, finite positive ``rtol``
and ``atol``, and a finite positive ``max_step`` when supplied. Invalid settings
raise ``VerificationError`` with the originating context and report before integration
starts.

Default tolerances are ``rtol=1e-8``, ``atol=1e-10``, and a maximum time step
of ``T/100``. The report compares a tighter run, measures independent central
time-difference defects, and checks the exponential context against its linear
matrix-exponential transformation. Both calculations use the context's declared
risk and terminal laws, including supported component replacements. The
comparisons record numerical differences and residuals; they do not prove
error bounds. Source fingerprints, exact parameters, solver settings, and
dependency versions accompany the results.
