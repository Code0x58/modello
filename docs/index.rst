modello documentation
=====================

``modello`` expresses symbolic models as Python classes. Compose typed
components, supply the values you know, and let SymPy solve the combined
equations. The same model can calculate outputs or infer missing inputs from
observed results.

Install with Python 3.10 or newer:

.. code-block:: console

   $ pip install modello

Build a system from its parts. Each component owns its quantities; the parent
connects them with equations. Here a fill relates a holding's opening and closing
states:

.. literalinclude:: ../README.md
   :start-after: ```python
   :end-before: ```
   :language: python

The first ``quantity`` assignment binds that name to ``after.quantity``; the
second defines its relationship to the opening state and execution. Cash follows
the same pattern. Supply typed instances with the values you know:

.. code-block:: python

   opening = HoldingState("opening", quantity=3, cash=1000)
   posted = FillPosting(
       "buy", before=opening,
       fill=Execution("execution", quantity=2, price=100, fee=1),
   )
   assert posted.after.quantity == 5
   assert posted.after.cash == 799

The components are solved jointly. Supplying closing cash instead of execution
price lets the same equations infer that price. The :doc:`nested-model guide
<nesting>` develops composition, connected components, and symbolic inputs.
The :doc:`practical applications <examples/applications/index>` use it for
invoice totals, production yield, and circuit measurements. The
:doc:`CDS reconciliation walkthrough <examples/reconciliation/index>` compares
source components, checks common entry errors, and exposes the pricing formula.
The :doc:`portfolio walkthrough <examples/portfolio/index>` adds nested
transitions, reporting, and fitting a refined component from an outer constraint.

Distinct instances receive distinct bound symbols, even if they share a name.
Values are solved at construction time; changing an attribute afterward does
not recompute the model. Construction requires one retained solver candidate;
zero or several candidates raise :class:`~modello.errors.SolutionCountError`.
A candidate may still represent a symbolic family of answers. The
:doc:`debugging guide <debugging>` explains these outcomes and how to inspect them.
Use :doc:`solution families <solutions>` to explore alternatives before selecting a state.

The `examples directory <https://github.com/Code0x58/modello/tree/master/examples>`_
contains geometry, flow-scaling, and historical tax-model examples. Tax examples
illustrate symbolic modeling with fixed historical rates.

.. toctree::
   :caption: Library
   :maxdepth: 2

   api
   nesting
   vectors
   series
   debugging
   solutions
   diagnostics
   gui

.. toctree::
   :caption: Examples
   :maxdepth: 1

   examples/applications/index
   examples/reconciliation/index
   examples/portfolio/index
   examples/timeseries/index
   examples/solution_families/index

.. toctree::
   :caption: Development
   :maxdepth: 1

   contributing

* :ref:`genindex`
* :ref:`search`
