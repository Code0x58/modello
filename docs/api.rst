API reference
=============

.. currentmodule:: modello

.. autoclass:: Modello
   :special-members: __init__

   Define quantities and their relationships as class attributes. Pass known
   scalar values and component instances as constructor keyword arguments.
   The model and its components are solved together; omitted inputs remain
   symbolic unless the supplied equations determine them.

   ``name`` labels an instance's symbols. Separate constructions and separate
   components have independent unknowns, even when their names are the same.
   Solved quantities are exposed as normal Python attributes. Create a new
   instance to solve with different inputs.

   **Inputs and errors**

   Scalar inputs can be Python numbers, decimals, fractions, SymPy expressions,
   or expression strings. The ``ScalarValue`` annotation describes these inputs;
   :class:`SymbolicScalar` provides the expression protocol for static typing.
   Components accept instances of their declared model class, including
   subclasses with the same field paths.

   Scalar expressions can use the model class's quantities or a parent's
   component paths. Other SymPy symbols remain external parameters.

   Unknown keyword names, invalid scalar values, and incompatible component
   types raise :class:`TypeError`. Inconsistent equations and multiple discrete
   solutions raise :class:`ValueError`. Symbol assumptions help select valid
   solutions; assumptions about external symbolic parameters may remain
   undecided. Each explicitly constructed input model must have a valid solution
   before it can be passed to a parent.

   **Component copies and methods**

   A parent copies the declared quantities of each supplied component and its
   descendants. It can complete symbolic inputs without changing the source.
   Copies have the declared component type and its methods. Ordinary instance
   attributes and extra fields of a supplied subclass belong to the source.

   Express relationships as class attributes and use methods for additional
   behavior. Component copies are filled from the parent's solution; custom
   ``__init__`` and ``__new__`` methods run when an input instance is explicitly
   constructed, and are not called again for those copies.

   **Inheritance**

   A subclass inherits its parent's quantities and relationships. Replacing a
   component updates inherited expressions by matching field paths. When a
   replacement changes the available fields or their aliases, redefine the
   expressions that use the changed fields too.

   When multiple model bases define the same quantity, the later base supplies
   that quantity. Methods follow Python's normal method resolution order.

.. autofunction:: component

   Use ``child = component(Child)`` in a model class body, then supply a
   ``Child`` instance when constructing the parent. Child attributes retain their
   model instance types for editors and type checkers. Constructor keyword names
   and compatible component classes are checked at runtime.

   See :doc:`nesting` for a walkthrough.

.. autoclass:: InstanceDummy
   :members: bound

   Declare an independent quantity with optional SymPy assumptions, such as
   ``positive=True`` or ``real=True``. Each model instance receives its own symbol
   for an unresolved quantity.

.. autoclass:: BoundInstanceDummy

.. autoclass:: SymbolicScalar
