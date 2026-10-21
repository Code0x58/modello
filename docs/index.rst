modello documentation
=====================

``modello`` lets you express symbolic models as Python classes. SymPy handles
the algebra; each model instance binds its own symbols and solves the equations
implied by its expressions and supplied values.

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
The :doc:`portfolio walkthrough <examples/portfolio/index>` takes this pattern
through nested transitions, reporting, and fitting a refined component from an
outer constraint.

Distinct instances receive distinct bound symbols, even if they share a name.
Values are solved at construction time; changing an attribute afterward does
not recompute the model. When constraints are present, a solver result without
exactly one solution raises :class:`ValueError`.

The `examples directory <https://github.com/Code0x58/modello/tree/master/examples>`_
contains geometry, flow-scaling, and historical tax-model examples. Tax examples
illustrate symbolic modeling with fixed historical rates.

.. toctree::
   :caption: Library
   :maxdepth: 2

   api
   nesting
   debugging
   solutions
   diagnostics

.. toctree::
   :caption: Examples
   :maxdepth: 1

   examples/portfolio/index
   examples/solution_families/index

.. toctree::
   :caption: Development
   :maxdepth: 1

   contributing

* :ref:`genindex`
* :ref:`search`
