A portfolio in connected models
===============================

A trade changes shares and cash. That small idea is enough to explore a larger
one: build a system from typed components, connect their equations, then change
the question you ask of the whole. This is an exploration of modelling ideas;
the portfolio supplies the setting.

The examples below use only ``modello`` and its SymPy dependency. Run them in
order in a Python session.

Connect before, fill, and after
-------------------------------

Define the state of a holding and the terms of a filled trade. Positive trade
quantities buy shares; negative quantities sell them:

>>> from modello import InstanceDummy, Modello, component
>>> class HoldingState(Modello):
...     quantity = InstanceDummy("quantity", real=True)
...     cash = InstanceDummy("cash", real=True)
>>> class Execution(Modello):
...     quantity = InstanceDummy("quantity", real=True)
...     price = InstanceDummy("price", positive=True)
...     fee = InstanceDummy("fee", nonnegative=True)
...     notional = quantity * price

A posting connects two states through those execution terms:

>>> class FillPosting(HoldingState):
...     before = component(HoldingState)
...     fill = component(Execution)
...     after = component(HoldingState)
...
...     quantity = after.quantity
...     quantity = before.quantity + fill.quantity
...     cash = after.cash
...     cash = before.cash - fill.notional - fill.fee

Repeating a field assignment declares an equality. Here ``quantity`` aliases
``after.quantity`` and equates it to the opening quantity plus the trade.
``Execution`` owns its notional equation; ``FillPosting`` owns the connections
and balance equations.

Supply an opening state and a buy to solve forwards:

>>> opening = HoldingState("opening", quantity=3, cash=1000)
>>> bought = FillPosting(
...     "buy", before=opening,
...     fill=Execution("execution", quantity=2, price=100, fee=1),
... )
>>> bought.after.quantity, bought.after.cash
(5, 799)

The result retains typed ``before``, ``fill``, and ``after`` components. The
opening instance stays unchanged; the completed posting can join another model.

Make the result another input
------------------------------

A ``FillPosting`` is itself a ``HoldingState``: its top-level quantity and cash
describe the closing state. Refine its input to accept a previous posting,
retaining that posting's components and equations:

>>> class FollowingFill(FillPosting):
...     before = component(FillPosting)

.. mermaid::
   :alt: Two successive fills share their intermediate state.

   flowchart LR
       previous["State T-1"] --> first["Fill"] --> middle["State T"]
       middle --> second["Fill"] --> next["State T+1"]

Now observe the cash after selling one share, leaving the sale price unknown:

>>> sale = FollowingFill(
...     "buy_then_sell", before=bought,
...     fill=Execution("sale", quantity=-1, fee=1),
...     after=HoldingState("observed", cash=908),
... )
>>> sale.quantity, sale.cash, sale.fill.price
(4, 908, 110)
>>> sale.before.fill.price
100

The same balance equations have solved backwards. ``before`` and ``after``
describe time order, not a required calculation order. Nesting the operations
connects their states without writing a separate two-trade calculation. Use
``bought.after`` when only its completed state is needed; declare the full
posting type when its inner equations must remain part of the joint model.

Add a view by joining components
--------------------------------

To report wealth at a market price, extend a holding with one valuation rule:

>>> class Valuation(HoldingState):
...     price = InstanceDummy("price", positive=True)
...     value = cash + quantity * price

A review layer joins that valuation to the posting's result:

>>> class Review(Modello):
...     posting = component(FillPosting)
...     marked = component(Valuation)
...
...     quantity = marked.quantity
...     quantity = posting.quantity
...     cash = marked.cash
...     cash = posting.cash
...     value = marked.value
>>> review = Review("review", posting=bought, marked=Valuation("market", price=110))
>>> review.value
1349

The review supplies only the glue. Settlement and valuation each keep their own
mathematics, while a constraint on ``review.value`` can reach unknowns inside
either component. The typed paths also show where each result came from.

Refine a component, then fit it from the outside
------------------------------------------------

Suppose execution price depends on order size. Add a deliberately simple impact
law, then select this richer component through the existing layers:

>>> class ImpactExecution(Execution):
...     reference = InstanceDummy("reference", positive=True)
...     impact = InstanceDummy("impact", nonnegative=True)
...     price = reference + impact * quantity
>>> class ImpactPosting(FillPosting):
...     fill = component(ImpactExecution)
>>> class ImpactReview(Review):
...     posting = component(ImpactPosting)

Keep the opening state and known trade size, but constrain the total value at
the review layer. The system infers the impact coefficient through valuation,
settlement, and execution:

>>> fitted = ImpactReview(
...     "calibration",
...     posting=ImpactPosting(
...         "buy", before=opening,
...         fill=ImpactExecution("execution", quantity=2, reference=99, fee=1),
...     ),
...     marked=Valuation("market", price=110),
...     value=1349,
... )
>>> fitted.posting.fill.impact, fitted.posting.fill.price, fitted.posting.after.cash
(1/2, 100, 799)

No settlement or valuation formula changed. Changing the component declaration
preserved those relationships and added the impact law. Fitting here means
solving for one unknown coefficient from an observation; a statistical fit to
noisy data would need an explicit residual model and fitting criterion.

Leaving the inputs symbolic also gives SymPy the complete expression to
differentiate. Here the sensitivity of wealth to impact follows through all
three layers:

>>> from sympy import diff
>>> symbolic = ImpactReview("sensitivity")
>>> diff(symbolic.value, symbolic.posting.fill.impact) == -symbolic.posting.fill.quantity**2
True

Explore further
----------------

The pattern extends to other state transitions and reporting layers: define
each part's mathematics, connect the parts in a parent, and supply observations
where the question belongs. The :doc:`nested-model guide </nesting>` and
:doc:`API reference </api>` cover the core mechanics.

The larger `portfolio prototype
<https://github.com/Code0x58/modello/tree/master/examples/portfolio>`_ explores
named baskets, external data, account timelines, and reports. Its
`continuous execution study
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
