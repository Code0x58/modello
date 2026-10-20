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

A signed fill changes held shares and pays its notional plus fees:

.. literalinclude:: ../../../examples/portfolio/postings.py
   :pyobject: FillPosting
   :language: python

>>> from examples.portfolio.postings import FillPosting, CapitalFlow
>>> posted = FillPosting(
...     "fill", opening_quantity=3, opening_cash=1000, quantity=2, price=100, fee=1
... )
>>> posted.closing_quantity, posted.closing_cash
(5, 799)

The same definition can infer an execution price from an observed cash change:

>>> inferred = FillPosting(
...     "observed",
...     opening_quantity=3,
...     opening_cash=1000,
...     quantity=2,
...     closing_cash=799,
...     fee=1,
... )
>>> inferred.price
100

A capital deposit changes cash, but it should not count as investment return.
Keep an account's number of units and issue new units at its pre-flow unit
price. A withdrawal redeems units using the same relationship:

.. literalinclude:: ../../../examples/portfolio/postings.py
   :pyobject: CapitalFlow
   :language: python

>>> deposit = CapitalFlow(
...     "deposit", opening_cash=1000, opening_nav=2500, opening_units=25, amount=500
... )
>>> deposit.unit_price, deposit.issued_units, deposit.closing_cash
(100, 5, 1500)
>>> deposit.closing_units
30

The unit price remains 100 at that deposit. Future unit-price changes measure
investment return while account NAV also reflects the supplied capital.

Separate profit from external capital
-------------------------------------

We also need a reconciliation identity. First define gross and net profit:

.. literalinclude:: ../../../examples/portfolio/pnl.py
   :pyobject: Profit
   :end-before:     @classmethod
   :language: python

``PROFIT_FIELDS`` names ``holding``, ``trading``, ``income``, ``fees``, ``gross``,
and ``net``. An instrument supplies the holding and trading terms from interval
inputs:

.. literalinclude:: ../../../examples/portfolio/pnl.py
   :pyobject: InstrumentPnL
   :language: python

Holding profit marks opening shares from opening to closing price. Trading
profit marks signed fills from their execution prices to that closing price.
The interval adapter supplies ``sum(quantity * price)`` as signed fill notional.
Income and fees remain separate columns.

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
records a mark's quantities, prices, values, cash, NAV, and units, together with
its knowledge horizon. ``AccountPath`` holds the snapshots and the in-window
ledger used to produce them.

.. literalinclude:: ../../../examples/portfolio/account.py
   :pyobject: Account
   :end-before:     def __post_init__
   :language: python

.. literalinclude:: ../../../examples/portfolio/account.py
   :pyobject: AccountSnapshot
   :end-before:     @property
   :language: python

The account needs to vary prices, quantities, and cash. This compiled view of
``NamedPortfolio`` selects exactly those inputs from an unbound instance:

.. literalinclude:: ../../../examples/portfolio/valuation.py
   :pyobject: NamedPortfolio.plan
   :dedent: 4
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
