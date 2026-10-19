Portfolio research
==================

This exploratory application follows declared equity holdings through price
observations, portfolio valuation, account replay, signal decisions, execution
analysis, and symbolic reporting. Its synthetic data demonstrates the
calculation layers and their accounting relationships.

Code and fixtures live together in ``examples/portfolio/``. These walkthroughs
and the generated `interactive report <report.html>`_ form a separate example
section; portfolio policies and time-series operations are outside the core
Modello API.

Start with fixed holdings, then follow the named-member and account extensions:

.. toctree::
   :maxdepth: 1

   holdings
   named_members
   providers
   accounts
   reporting
   layers

The :doc:`named-solving design note </design/named_solving>` isolates the
general modeling question from the financial application.
