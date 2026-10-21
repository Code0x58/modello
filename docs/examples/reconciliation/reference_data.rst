Compare source components
==========================

Bundle related observations from one source into a component. To investigate
a break, swap that component and ask whether its assumptions fit the trusted
output, keeping other observations fixed. This works for a market-data
snapshot, a counterparty trade record, or a shadow account's inputs.

CDS identifiers and provenance
------------------------------

For single-name CDS, RED distinguishes the six-character entity code from the
nine-character entity–obligation pair code; see `Markit's identifier description
<https://www.esma.europa.eu/sites/default/files/1-Markit_1.pdf>`_. A different
obligation can mean a different pair while the entity stays the same. The
`RED service <https://www.spglobal.com/market-intelligence/en/solutions/products/red-cds>`_
also tracks reference-data events. A changed identifier alone does not reprice
a trade, and an unchanged identifier does not guarantee the right parameters.

Sovereign CDS can have different 2003 and 2014 ISDA contract definitions; see
the `ESRB report, section 3.3
<https://www.esrb.europa.eu/pub/pdf/reports/esrb.report250411_creditdefaultswaps.en.pdf>`_.
Check the actual contract independently: a definitions change need not imply
a RED change, and legacy contracts are not automatically upgraded here.

One component per source snapshot
---------------------------------

The two source models below each own their numerical assumptions and
provenance. All ``DEMO:`` identifiers and prices are synthetic; this is not a
reconstruction of a sovereign event. The alternative snapshot changes both
the pair and date, along with its survival assumption:

.. literalinclude:: ../../../examples/reconciliation.py
   :start-at: class PreviousREDMarket
   :end-before: def reference_problem
   :language: python

Strings and dates are ordinary class metadata. Numerical assumptions are model
equations, inherited alongside the one-period pricing relationships. Definitions
and other contract labels do not implement legal payoff differences in this
simplified model.

Swap the source in the evaluation
---------------------------------

``reference_problem(source_model)`` refines the closing market component through
the existing composition layers. This retains the source's equations, type,
and class metadata in the result. Trade terms, opening market, cashflow, and
trusted PnL stay fixed:

.. literalinclude:: ../../../examples/reconciliation.py
   :start-at: def reference_problem
   :end-before: def main
   :language: python

>>> from examples.reconciliation import PreviousREDMarket, CurrentREDMarket, reference_problem
>>> for source in (PreviousREDMarket, CurrentREDMarket):
...     trial = reference_problem(source)
...     calculated = trial.without("pnl").solutions().one()
...     print(source.pair_code, calculated.pnl, trial.analyse().status)
DEMO:PAIR:OLD 20500 inconsistent
DEMO:PAIR:CURRENT 51000 consistent
>>> refreshed = reference_problem(CurrentREDMarket).solutions().one()
>>> refreshed.signed_notional, refreshed.closing.value, refreshed.pnl
(10000000, 83000, 51000)
>>> type(refreshed.closing.market) is CurrentREDMarket
True

The current component fits; the old component does not. Only the numerical
assumptions affect this price: relabelling old inputs leaves the break intact.
Several different parameter sets could also fit the same PnL. Compare each
candidate with independent identity, terms, date, and source records before
accepting it.

Holding the stale component fixed and releasing notional instead could hide
the source issue:

>>> from examples.reconcile import Reconciliation, Suspect
>>> problem = reference_problem(PreviousREDMarket)
>>> notional = problem.field("signed_notional")
>>> hypothetical = Reconciliation(problem, [Suspect(notional)]).infer(notional).one()
>>> hypothetical.signed_notional, hypothetical.pnl
(20000000, 51000)

That explanation conflicts with the independently confirmed 10m notional.
The :doc:`mutation workflow <framework>` remains useful for observations
considered uncertain after comparing source components.
