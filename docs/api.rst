API reference
=============

.. currentmodule:: modello

.. autoclass:: InstanceDummy
   :members: bound

.. autoclass:: BoundInstanceDummy

.. autoclass:: Modello
   :special-members: __init__

   Model attributes can be SymPy expressions involving other model attributes.
   Pass known values as keyword arguments to the constructor. Omitted inputs
   remain symbolic unless the supplied constraints determine them. Each instance
   exposes solved values as normal Python attributes. Declared formulas are
   expanded before solving supplied constraints; unconstrained independent inputs
   are not assigned arbitrary particular values. A single symbolic solution may
   still contain free parameters. Zero or multiple discrete solutions raise
   :class:`ValueError`.

   Declare named components by assigning another model class in the class body.
   Child keyword arguments accept mappings of values or compatible model
   instances. The entire tree is solved together. See :doc:`nesting` for symbol
   scopes, snapshots, inheritance, validation, and construction semantics.

   The metaclass and private ``_modello_*`` attributes are implementation details.
