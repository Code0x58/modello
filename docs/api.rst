API reference
=============

.. currentmodule:: modello

.. autoclass:: InstanceDummy
   :members: bound

.. autoclass:: BoundInstanceDummy

.. autofunction:: component

   Use ``child = component(Child)`` in a model class body. Type checkers infer
   ``instance.child`` as a ``Child`` instance, including its methods and fields.
   The helper declares a component; it does not construct an input instance.

.. autoclass:: SymbolicScalar

   Structural typing protocol for SymPy scalar inputs. Runtime scalar validation
   still requires conversion to a SymPy expression. ``ScalarValue`` combines this
   protocol with Python numeric, decimal, fraction, and string input types.

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

   Declare named components with ``component(Model)`` in the class body.
   Child keyword arguments require compatible model instances; dictionaries and
   other mappings are rejected. The entire tree is solved together. See
   :doc:`nesting` for symbol scopes, snapshots, inheritance, validation, and
   construction semantics.

   The metaclass and private ``_modello_*`` attributes are implementation details.
