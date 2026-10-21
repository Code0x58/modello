Check CDS reference data first
==============================

For single-name CDS, RED identifies entities and their relationships to obligations:
the six-character entity code and nine-character entity–obligation pair code
are distinct (see `Markit's identifier description
<https://www.esma.europa.eu/sites/default/files/1-Markit_1.pdf>`_). The
`RED service <https://www.spglobal.com/market-intelligence/en/solutions/products/red-cds>`_
also tracks reference-data events. A different obligation can mean a different
pair while the entity remains the same. This does not by itself change a price.

An unchanged code is not sufficient either. Sovereign CDS can have different
2003 and 2014 ISDA contract definitions; see the `ESRB report, section 3.3
<https://www.esrb.europa.eu/pub/pdf/reports/esrb.report250411_creditdefaultswaps.en.pdf>`_.
The example therefore keeps definitions, currency, seniority, restructuring,
maturity, and the market-data date alongside the RED identifiers. Confirm these
coordinates independently; there is no automatic rule here that a definitions
change requires a RED change or that every legacy contract should be upgraded.
Definitions are lookup coordinates here; the one-period model does not implement
their legal payoff differences.

A stale parameter bundle
------------------------

Run ``python -m examples.reconciliation`` for this scenario as well as the
entry-error example. All identifiers prefixed ``DEMO:`` and all prices are
synthetic. This is not a reconstruction of a particular sovereign event.

The confirmed mapping now specifies another pair for the same entity. The
loaded inputs still come from the old pair and an earlier market-data date:

>>> from examples.reconciliation import stale_reference_case
>>> case = stale_reference_case()
>>> case.loaded.key.entity_code == case.confirmed.entity_code
True
>>> case.loaded.key.pair_code, case.confirmed.pair_code
('DEMO:PAIR:OLD', 'DEMO:PAIR:CURRENT')
>>> case.mismatches
('pair_code', 'as_of')
>>> problem = case.problem(case.loaded)
>>> recorded = problem.without("pnl").solutions().one()
>>> recorded.signed_notional, recorded.closing.value, recorded.pnl
(10000000, 52500, 20500)

Confirmed trade terms and trusted PnL stay fixed. Select a source bundle by the
full confirmed key and requested date, then rebuild the closing market inputs:

>>> selected = case.select()
>>> case.loaded.survival, selected.survival
(39/40, 97/100)
>>> refreshed = case.problem(selected).solutions().one()
>>> refreshed.signed_notional, refreshed.closing.value, refreshed.pnl
(10000000, 83000, 51000)

The lookup does not choose whichever source makes PnL fit. Missing or duplicate
matches raise ``ValueError``. Historical lookups use their requested key and
date, rather than silently taking today's latest mapping.

The numerical inputs caused this break. Merely changing the old row's labels
does not reprice it; the matching source bundle must actually be loaded.
Identifiers are metadata outside the symbolic algebra.

Why this comes before adjusting trade fields
--------------------------------------------

Holding the stale inputs fixed and releasing notional could conceal the issue:

>>> from examples.reconcile import Reconciliation, Suspect
>>> notional = problem.field("signed_notional")
>>> hypothetical = Reconciliation(problem, [Suspect(notional)]).infer(notional).one()
>>> hypothetical.signed_notional, hypothetical.pnl
(20000000, 51000)

That numerical explanation conflicts with the independently confirmed 10m
notional. Check identity, contract terms, source date, and data lineage before
proposing scalar corrections. The existing :doc:`mutation workflow <framework>`
then applies to fields genuinely considered uncertain. Neither matching PnL
nor matching metadata alone establishes that a parameter source is correct.
