modello documentation
=====================

``modello`` lets you express symbolic models as Python classes. SymPy handles
the algebra; each model instance binds its own symbols and solves the equations
implied by its expressions and supplied values.

Install with Python 3.10 or newer:

.. code-block:: console

   $ pip install modello

Declare independent inputs with :class:`modello.InstanceDummy` and derive a model
from :class:`modello.Modello`:

.. code-block:: python

   from sympy import sqrt
   from modello import InstanceDummy, Modello

   class RightAngleTriangle(Modello):
       a = InstanceDummy("a", real=True, positive=True)
       b = InstanceDummy("b", real=True, positive=True)
       c = sqrt(a**2 + b**2)

   triangle = RightAngleTriangle("T", a=3, b=4)
   assert triangle.c == 5

   triangle = RightAngleTriangle("T", b=4, c=5)
   assert triangle.a == 3

Positive assumptions select the positive side length when solving the inverse
problem. Distinct instances receive distinct bound symbols, even if they share
a name. Values are solved at construction time; changing an attribute afterward
does not recompute the model. When constraints are present, a solver result
without exactly one solution raises :class:`ValueError`.

The `examples directory <https://github.com/Code0x58/modello/tree/master/examples>`_
contains geometry, flow-scaling, and historical tax-model examples. Tax examples
illustrate symbolic modeling with fixed historical rates.

The :doc:`portfolio analytics experiment <portfolio>` combines named equity
positions with price vectors, return attribution, and sample risk estimates.
Its :doc:`named-member extension <named_members>` explores projection and
reduction over maps of typed positions and their weights.
The :doc:`account and execution walkthrough <accounts>` adds external market and
transaction sources, time-aware signals, ledger replay, and execution costs.
Its :doc:`symbolic reporting layer <reporting>` adds signal P&L attribution,
cash-utility derivatives, and account/instrument tables and timeline charts.

.. toctree::
   :maxdepth: 2

   api
   nesting
   portfolio
   named_members
   accounts
   reporting
   research_layers
   contributing

* :ref:`genindex`
* :ref:`search`
