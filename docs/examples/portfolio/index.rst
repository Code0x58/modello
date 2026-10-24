An account across executions
============================

A fill changes shares and cash. Represent the account as a finite sequence of
typed states, then supply opening balances, execution terms, or a closing
observation according to the question. This walkthrough reuses the recurrent
model from the :doc:`fundamentals gallery <../gallery/index>`; the equations live
in ``examples.portfolio.models``, independently of the GUI.

Post executions and reconcile a balance
---------------------------------------

``AccountSequence`` applies the same relative equations at every execution:

.. math::

   Q_n = Q_{n-1} + u_{n-1},\qquad
   C_n = C_{n-1} - u_{n-1}p_{n-1} - f_{n-1}.

Positive quantities buy shares; negative quantities sell them. The execution
model owns its notional equation. The :doc:`model-vector guide </model-vectors>`
shows the sequence declaration and its boundary semantics.

>>> from modello import InstanceDummy, Modello, ModelObservations, component
>>> from examples.portfolio.models import AccountSequence, Execution, HoldingState
>>> opening = HoldingState("opening", quantity=3, cash=1000)
>>> buy = Execution("buy", quantity=2, price=100, fee=1)
>>> bought = AccountSequence("buy", fills=[buy], states=ModelObservations({0: opening}))
>>> bought.terminal.quantity, bought.terminal.cash
(5, 799)

Add a sale, leave its price unknown, and observe the terminal cash instead:

>>> sale = AccountSequence(
...     "buy_then_sell", fills=[buy, Execution("sale", quantity=-1, fee=1)],
...     states=ModelObservations({0: opening}),
...     terminal=HoldingState("statement", cash=908),
... )
>>> sale.terminal.quantity, sale.terminal.cash, sale.fills[1].price
(4, 908, 110)
>>> tuple(state.cash for state in sale.states)
(1000, 799, 908)

The same law handles both questions. Execution count determines ``N``; there
are ``N+1`` account boundaries. Each execution retains its typed fields, and
every intermediate state is available without nesting another posting model.
The opening observation and earlier solved account stay unchanged.

Join a valuation view
---------------------

To ask about wealth at a market price, give a holding one valuation rule and
join its balances to the account's terminal state:

>>> class Valuation(HoldingState):
...     price = InstanceDummy("price", positive=True)
...     value = HoldingState.cash + HoldingState.quantity * price
>>> class Review(Modello):
...     account = component(AccountSequence)
...     marked = component(Valuation)
...     quantity = marked.quantity
...     quantity = account.terminal.quantity
...     cash = marked.cash
...     cash = account.terminal.cash
...     value = marked.value
>>> review = Review("review", account=bought, marked=Valuation("market", price=110))
>>> review.value
1349

Repeated assignments connect the same quantity to both components. Settlement
and valuation each keep their own mathematics. A constraint on the review's
value can reach an unknown execution price inside the account:

>>> partial = AccountSequence(
...     "unpriced_buy", fills=[Execution("buy", quantity=2, fee=1)],
...     states=ModelObservations({0: opening}),
... )
>>> fitted = Review("reconcile", account=partial,
...                 marked=Valuation("market", price=110), value=1349)
>>> fitted.account.fills[0].price, fitted.account.terminal.cash
(100, 799)

Leaving that observation out retains a SymPy formula. Its sensitivity to the
execution price follows directly from the balance law:

>>> from sympy import diff
>>> diff(partial.terminal.cash, partial.fills[0].price) == -partial.fills[0].quantity
True

For component refinement, see :doc:`nesting </nesting>`; for calibration with
an explicit impact model and bounded observations, see
:doc:`solution families <../solution_families/index>`.

Explore further
----------------

The larger `portfolio prototype
<https://github.com/Code0x58/modello/tree/master/examples/portfolio>`_ adds named
baskets, external data, heterogeneous ledger events, and reports. Its event
adapter retains individual transitions for cash flows, income, and fills.
Use the recurrence above for a homogeneous window of executions.

The `continuous execution study
<https://github.com/Code0x58/modello/blob/master/examples/portfolio/optimal_execution.py>`_
uses SymPy to derive differential equations, integrate onto a discrete grid,
and fit parameters from the same underlying model. These experimental helpers
live under ``examples/``, separately from the core library.

To run the richer prototype with its bundled synthetic data:

.. code-block:: console

   $ pip install "modello[examples]"
   $ python -m examples.portfolio --html account.html --json account.json
   $ python -m examples.portfolio.optimal_execution

An `interactive sample report <report.html>`_ is also available. The toy account
illustrates modelling and accounting, rather than evidence of investment skill.
