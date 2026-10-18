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
   remain symbolic or are selected by SymPy's particular-solution behavior for
   underdetermined equations. Each instance exposes the solved values as normal
   Python attributes.

   The metaclass and private ``_modello_*`` attributes are implementation details.
