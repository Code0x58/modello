5. Replay an account from external transactions
===============================================

So far, quantities and cash have stayed fixed while prices varied. An account
adds an opening state and an external transaction log. We will first define the
postings and their balance equations, then supply them to a chronological
adapter using the portfolio valuation we already built.

Describe events separately from their effects
---------------------------------------------

Each posting needs an identity, account, economic time, and reporting time.
These are the shared record fields:

.. literalinclude:: ../../../examples/portfolio/sources.py
   :pyobject: Posting
   :end-before:     def __post_init__
   :language: python

A fill adds the instrument's holding name, signed shares, price, fee, and an
optional link to an intended order:

.. literalinclude:: ../../../examples/portfolio/sources.py
   :pyobject: Fill
   :end-before:     def __post_init__
   :language: python

``CashFlow`` adds an ``amount`` for an external deposit or withdrawal.
``Income`` adds an ``amount`` and optional ``member`` for investment income.
The record constructors validate identities, timestamps, and finite amounts.
They describe observations; the following models define their arithmetic.

Model a fill and a capital movement
-----------------------------------

A fill relates two instances of the same state: shares and settlement cash.
First define that state, then the execution's economic terms:

.. literalinclude:: ../../../examples/portfolio/postings.py
   :pyobject: HoldingState
   :language: python

A signed trade owns the relationship between shares, price, and notional.
An execution extends it with a settlement fee:

.. literalinclude:: ../../../examples/portfolio/models.py
   :pyobject: Trade
   :language: python

.. literalinclude:: ../../../examples/portfolio/postings.py
   :pyobject: Execution
   :language: python

The ledger's ``Fill`` record supplies identity and timestamps; ``Execution``
supplies the scalar terms of the balance equations. Define those equations
between typed ``before`` and ``after`` components:

.. literalinclude:: ../../../examples/portfolio/postings.py
   :pyobject: FillPosting
   :language: python

>>> from examples.portfolio.postings import HoldingState, Execution, FillPosting
>>> before = HoldingState("opening", quantity=3, cash=1000)
>>> posted = FillPosting(
...     "fill", before=before, fill=Execution("execution", quantity=2, price=100, fee=1)
... )
>>> posted.after.quantity, posted.after.cash
(5, 799)

The repeated ``quantity`` assignments equate ``after.quantity`` to the opening
quantity plus the signed fill. The cash assignments connect the other pair.
These are symbolic equalities, so supplying an observed closing balance can
infer an execution price through the same definition:

>>> inferred = FillPosting(
...     "observed",
...     before=before,
...     fill=Execution("execution", quantity=2, fee=1),
...     after=HoldingState("observed", cash=799),
... )
>>> inferred.fill.price
100

Neither component name fixes the solving direction. ``before`` and ``after``
describe economic order; known values determine the numerical view. The
opening instance remains unchanged:

>>> before.quantity, before.cash
(3, 1000)

A capital deposit changes cash, but it should not count as investment return.
Keep an account's number of units and issue new units at its pre-flow unit
price. Define a valued account state and a signed cash movement:

.. literalinclude:: ../../../examples/portfolio/models.py
   :pyobject: AccountValue
   :language: python

.. literalinclude:: ../../../examples/portfolio/postings.py
   :pyobject: AccountState
   :language: python

.. literalinclude:: ../../../examples/portfolio/postings.py
   :pyobject: CashMovement
   :language: python

The transition relates cash, NAV, and units on both sides of the flow. A
withdrawal redeems units through the same equations:

.. literalinclude:: ../../../examples/portfolio/postings.py
   :pyobject: CapitalFlow
   :language: python

>>> from examples.portfolio.postings import AccountState, CashMovement, CapitalFlow
>>> account_before = AccountState("opening", cash=1000, nav=2500, units=25)
>>> deposit = CapitalFlow(
...     "deposit", before=account_before, flow=CashMovement("external", amount=500)
... )
>>> deposit.before.unit_price, deposit.issued_units, deposit.after.cash
(100, 5, 1500)
>>> deposit.after.units, deposit.after.nav, deposit.after.unit_price
(30, 3000, 100)

SymPy can establish the preserved unit price before we supply any numbers:

>>> from sympy import simplify
>>> flow = CapitalFlow("identity")
>>> simplify(flow.after.unit_price - flow.before.unit_price)
0

Future unit-price changes measure investment return while account NAV also
reflects the supplied capital.

Connect transitions into a progression
--------------------------------------

Each event advances its relevant state from :math:`T-1` to :math:`T`. Two fills
of the same holding and cash account can share the intermediate state:

.. mermaid::
   :caption: A shared state connects successive executions
   :alt: State at T minus one passes through a fill to state T, then through another fill to state T plus one.

   flowchart LR
       previous["State T-1"] --> first["First fill"]
       first --> middle["State T"]
       middle --> second["Second fill"]
       second --> next["State T+1"]

Declare the two transitions and equate their intermediate quantities and cash.
As with basket composition, the outer model supplies only the connecting
equalities; each child owns its economic equations:

>>> from examples.symbolic import SymbolicModel
>>> from modello import component
>>> class TwoFills(SymbolicModel):
...     first = component(FillPosting)
...     second = component(FillPosting)
...     quantity = second.before.quantity
...     quantity = first.after.quantity
...     cash = second.before.cash
...     cash = first.after.cash

Reuse our known buy as the first fill. For the second, observe the cash after
selling one share and leave its price unknown. The joint model connects the
first fill's closing cash to the second fill's opening cash and infers the
sale price:

>>> pair = TwoFills(
...     "buy_then_sell",
...     first=posted,
...     second=FillPosting(
...         "sale",
...         fill=Execution("sale", quantity=-1, fee=1),
...         after=HoldingState("observed", cash=908),
...     ),
... )
>>> pair.second.before.quantity, pair.second.before.cash
(5, 799)
>>> pair.second.fill.price, pair.second.after.quantity
(110, 4)

The links identify state at event boundaries. The account adapter below supplies
event order, economic timestamps, and knowledge horizons. An observation axis
can evaluate many such cases; it does not implicitly connect one row's state
to the next. Signals needing a longer history will use explicit lagged series.

Compile and reuse the typed transition
--------------------------------------

The same ``SymbolicModel.plan()`` helper introduced in chapter 3 caches an
unbound model view. Supply known values by field reference and read the result
through its state components:

>>> from examples.model_plan import Known
>>> view = FillPosting.plan()
>>> model = view.source
>>> result = view.evaluate(
...     Known(model.before.quantity, 3),
...     Known(model.before.cash, 1000),
...     Known(model.fill.quantity, 2),
...     Known(model.fill.price, 100),
...     Known(model.fill.fee, 1),
... )
>>> type(result.after) is HoldingState, float(result.after.cash)
(True, 799.0)

Compilation keeps the before/event/after structure for traversal and display.
Repeated numerical evaluation reuses the compiled expressions without solving
again. An inverse case can use its own ``compile()`` view with observed closing
state as input, just as in chapter 3.

Separate profit from external capital
-------------------------------------

We also need a reconciliation identity. First define gross and net profit:

.. literalinclude:: ../../../examples/portfolio/pnl.py
   :pyobject: Profit
   :end-before:     @staticmethod
   :language: python

``PROFIT_FIELDS`` names ``holding``, ``trading``, ``income``, ``fees``, ``gross``,
and ``net``. A reporting view contains this profit model and exposes its fields
as convenient aliases. Both instrument and account views inherit that binding:

.. literalinclude:: ../../../examples/portfolio/pnl.py
   :pyobject: ProfitView
   :language: python

For an interval we need two marked positions, the signed activity between them,
and a price change. ``Position`` and ``Quote`` were defined in chapter 1; now
define their interval counterparts:

.. literalinclude:: ../../../examples/portfolio/pnl.py
   :pyobject: TradingActivity
   :language: python

.. literalinclude:: ../../../examples/portfolio/models.py
   :pyobject: PriceChange
   :language: python

Connect those parts to each other and to the profit components:

.. literalinclude:: ../../../examples/portfolio/pnl.py
   :pyobject: InstrumentPnL
   :language: python

Holding profit marks opening shares from opening to closing price. Trading
profit marks signed fills from their execution prices to that closing price.
The interval adapter supplies ``sum(quantity * price)`` as signed fill notional.
Income and fees remain separate columns. Closing shares are opening shares
plus signed trading activity. All these relationships belong to one joint
model; the adapter only supplies observations from the external log.

At account level, the identity is closing NAV minus opening NAV, less net profit
and external capital. A zero residual reconciles the layers:

.. literalinclude:: ../../../examples/portfolio/pnl.py
   :pyobject: AccountPnL
   :language: python

These definitions will be reused for signal attribution later. Reporting will
supply different interval inputs to the same equations.

Supply state to the portfolio's valuation
-----------------------------------------

``Account`` is a name, opening time, and typed opening portfolio. ``AccountSnapshot``
retains the evaluated portfolio and participation units at a mark, together with
its knowledge horizon. Its ``portfolio`` preserves the declared basket and
member types. Quantities, prices, values, cash, NAV, and weights are projections
of that same model. ``AccountPath`` holds these snapshots and the in-window
ledger used to produce them.

.. literalinclude:: ../../../examples/portfolio/account.py
   :pyobject: Account
   :end-before:     def __post_init__
   :language: python

.. literalinclude:: ../../../examples/portfolio/account.py
   :pyobject: AccountSnapshot
   :end-before:     @cached_property
   :language: python

The account needs to vary prices, quantities, and cash. This compiled view of
``NamedPortfolio`` selects exactly those inputs from an unbound instance:

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: SymbolicModel.plan
   :dedent: 4
   :language: python

The cached factory compiles the case's remaining independent fields and keeps
its full component schema:

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: _model_plan
   :language: python

At a mark, the replay adapter obtains causal prices and supplies the current
state to that plan. Its constructor validates the opening portfolio, records
its known quantities and cash, stores the market/age policy, and compiles this
one unbound view:

.. literalinclude:: ../../../examples/portfolio/account.py
   :pyobject: AccountEngine._value
   :dedent: 4
   :language: python

Replay orders postings through each mark. It uses ``FillPosting`` and
``CapitalFlow`` for state changes, adds income to cash, and records a valued
snapshot. Its complete loop is shown here now that the inputs, models, and
valuation step have been introduced:

.. literalinclude:: ../../../examples/portfolio/account.py
   :pyobject: AccountEngine.evaluate
   :dedent: 4
   :language: python

The adapter starts account units at an arbitrary index level of 100. It requires
funded cash, nonnegative actual positions, positive NAV, and positive remaining
units. Events are applied in ``(opened_at, mark]``; the opening balances already
include earlier history. The final reconciliation checks the profit equations
above against the replayed balances.

Assemble the worked account
---------------------------

Reuse the two-equity classes developed in the previous chapter. For this
account, start with ten shares of each equity and USD 1,000 cash:

>>> from datetime import timedelta
>>> from pathlib import Path
>>> import examples.portfolio as portfolio_examples
>>> from examples.portfolio.explicit import Portfolio, Basket, EquityMembers
>>> from examples.portfolio.basket import equity
>>> from examples.portfolio.models import USDCash
>>> from examples.portfolio.account import Account
>>> from examples.portfolio.sources import CSVTransactionSource, timestamp
>>> times = tuple(timestamp(f"2026-01-{day:02d}T16:01:00Z") for day in range(5, 9))
>>> opening = Portfolio(
...     "opening",
...     basket=Basket(
...         "basket",
...         members=EquityMembers(
...             "equities",
...             aapl=equity("apple", quantity=10),
...             msft=equity("microsoft", quantity=10),
...         ),
...     ),
...     cash=USDCash("cash", amount=1000),
... )
>>> account = Account("research", times[0], opening)
>>> age = timedelta(days=4)
>>> market = opening.load_market()
>>> data_dir = Path(portfolio_examples.__file__).with_name("data")
>>> ledger = CSVTransactionSource(data_dir / "transactions.csv").load()
>>> path = account.replay(ledger, times, market=market, max_price_age=age)
>>> tuple(round(value, 2) for value in path.nav.values)
(2500.0, 2510.0, 3041.4, 3066.9)
>>> path.positions["aapl"].quantity.values, path.positions["msft"].quantity.values
((10.0, 10.0, 17.0, 17.0), (10.0, 10.0, 0.0, 0.0))

The transaction source implements ``Provider[TransactionLog]``. Its fixture has
three fills on January 7, a USD 500 deposit that day, and USD 8.50 income on
January 8. It remains independent of the model's future order intentions.
The public ``replay`` method simply constructs the adapter we just inspected:

.. literalinclude:: ../../../examples/portfolio/account.py
   :pyobject: Account.replay
   :dedent: 4
   :language: python

Check the accounting relationships
----------------------------------

``path.positions`` projects each holding's quantity, price, value, and weight
across snapshots. For profit, the path groups fills and other postings into
``(previous mark, current mark]`` intervals, then supplies their columns to
``InstrumentPnL`` and ``AccountPnL``:

.. raw:: html

   <details><summary>Inspect how postings become interval-model inputs</summary>

.. literalinclude:: ../../../examples/portfolio/account.py
   :pyobject: AccountPath.pnl
   :dedent: 4
   :language: python

.. raw:: html

   </details>

>>> tuple(round(value, 2) for value in path.pnl.total.values)
(10.0, 31.4, 25.5)
>>> path.pnl.external_flows.values
(0.0, 500.0, 0.0)
>>> tuple(round(value, 10) for value in path.reconciliation.values)
(0.0, 0.0, 0.0)
>>> round(path.unit_nav.values[-1] / path.unit_nav.values[0] - 1, 6)
0.024518

Total investment profit is USD 66.90. NAV also includes USD 500 of supplied
capital; its raw percentage change answers a different question from the
unitized investment return.

Keep decision knowledge separate from later reports
---------------------------------------------------

A transaction's execution time and reporting time can differ. Replay uses
postings known by the final reporting horizon, so a later report can restate an
earlier balance. Snapshots carry ``known_at`` to make that choice visible.

A decision must instead use a path ending at its own information horizon.
The :doc:`next chapter <signals>` constructs such a snapshot, builds a trailing
signal, and derives order intentions without replacing the external fill log.
