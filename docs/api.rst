API reference
=============

.. currentmodule:: modello

.. autoclass:: Modello
   :special-members: __init__

   Define quantities and their relationships as class attributes. Pass known
   scalar values, finite vectors, and component instances as constructor keyword arguments.
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

   ``VectorValue`` describes vector observations: a ``Vector`` or a sequence of
   scalar values with optional ``None`` entries for unobserved coordinates.
   Shape and field kind are checked at runtime. See :doc:`vectors` for indexed
   formulas, coordinate assumptions, and joint vector solving.

   Scalar expressions can use the model class's quantities or a parent's
   component paths. Other SymPy symbols remain external parameters.

   Unknown keyword names, invalid scalar values, and incompatible component
   types raise :class:`modello.errors.ModelInputError`, also a :class:`TypeError`.
   No accepted solver candidates or multiple candidates raise
   :class:`modello.errors.SolutionCountError`; a backend exception raises
   :class:`modello.errors.SolverBackendError`. Both are
   :class:`modello.errors.ModelSolveError` and :class:`ValueError` subclasses.
   Symbol assumptions help select candidates; assumptions about external symbolic
   parameters may remain undecided. A supplied component is constructed first in
   its own context. Parent observations can impose additional constraints, so
   successful child construction does not establish consistency of the parent.

   Solve failures retain a frozen ``problem`` snapshot with actual symbol
   identities, aliases, origins, equations, inputs, and solver candidates.
   Backend solver exceptions are chained as the cause. Diagnostic analysis runs
   only when explicitly invoked. A zero candidate count is not itself a proof
   of inconsistency. See
   :doc:`debugging` for symbolic results, redundant observations, conflicting
   constraints, and ambiguous solutions.

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

.. autofunction:: inspect_model

   Supply the same typed inputs as construction. Preparing a snapshot does not
   invoke the solver. Component instances supplied as inputs have already been
   constructed separately.
   See :doc:`diagnostics` for the structured snapshot, report types, and guarantees.
   ``problem.symbols`` exposes a read-only symbol tree with the model's component
   types. Use its attributes to write conditions and select outputs when
   ``problem.solutions()`` explores a joint solution family; see :doc:`solutions`.

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

.. autoclass:: SymbolicCondition

Coordinates and indexed formulas
--------------------------------

.. autoclass:: Coordinate

.. autoclass:: Index

.. autoclass:: Vector
   :members: shape, elements, applyfunc, sum, product, dot, doit

.. autofunction:: vector

See :doc:`series` for ``Series`` and ``Recurrence``: scalar families and causal
laws queried without a chosen horizon. See :doc:`model-vectors` for typed
ordered components, ``model_vector``, and
``model_recurrence``. ``ModelVectorValue[Model]`` describes observations as a
typed model vector or a sequence of compatible model instances and ``None``.
It also accepts ``ModelObservations[Model]`` for sparse typed observations of
positions within an independently determined window. Indexed field laws use
``model_recurrence(states[n], ...)`` and expose their source ``ModelRecurrence``
template alongside the finite model equations.

.. _extension-annotations:

Extension annotations
---------------------

Extensions can attach metadata to declarations without changing their values,
symbols, assumptions, aliases, or equations. Subclass ``Annotation`` and use
``annotate`` inside a model class body:

.. code-block:: python

   from dataclasses import dataclass
   from modello import Annotation, InstanceDummy, Modello, annotate, component

   @dataclass(frozen=True)
   class Unit(Annotation):
       symbol: str

   class Panel(Modello):
       width = annotate(InstanceDummy("width", positive=True), Unit("m"))
       area = annotate(width**2, Unit("m²"))

   class Project(Modello):
       panel = component(Panel)

   assert Unit.for_model(Project)["panel.width"].symbol == "m"
   assert Panel.__modello_annotations__[Unit]["area"].symbol == "m²"
   assert Panel("case", width=3).area == 9

The class collector unwraps each declaration before subsequent expressions use
it. Helpers preserve the declared value's static type, including typed
components. Use ``annotate`` around an entire assigned value; it is a class-body
declaration helper rather than a general symbolic expression wrapper. Ordinary
Python attributes keep their ordinary behaviour when annotated.

``Model.__modello_annotations__`` groups metadata by its **exact annotation
class**, then relative dotted attribute path. Both mapping levels are read-only.
The empty path ``""`` stores model-level metadata. ``annotate_model`` supports
extension decorators and returns the original class. Decorate a component
before declaring models that contain it, so its metadata is included in their
compiled registries.

Several annotation types can decorate one field, with one declaration per exact
type in each class scope. Duplicate types in a call, stacked helpers, repeated
field annotations, and repeated model decorators raise
``ModelDefinitionError``, even when the metadata is identical. A rejected model
annotation leaves its registry intact.

Inheritance preserves metadata by field name. A subclass has a new declaration
scope and may explicitly replace an inherited annotation as a whole. Later
model bases take precedence for each annotation type. A plain field reassignment
retains its annotations, including the common pattern of declaring an assumed
symbol and then assigning its defining expression. Aliases have separate named
metadata while retaining their shared symbol.

Nested components contribute scoped paths automatically. Replacing a component
rebuilds its descendant metadata from the replacement class, dropping obsolete
paths. A label attached directly to the containing component declaration stays
with that named field. Each model has its own registry mappings.

This mechanism has no dependency on a particular extension. For example,
``modello.gui`` implements ``FieldInfo(Annotation)`` and the ``gui`` helper;
importing and using core annotations does not import the GUI or Flask.

.. autoclass:: Annotation
   :members: for_model

.. autofunction:: annotate

.. autofunction:: annotate_model

.. _errors:

Errors
------

Catch ``ModelloError`` for Modello's validation and construction failures, or
``ModelSolveError`` for failed solves. These classes are available from
``modello`` and ``modello.errors``. Existing ``TypeError`` and ``ValueError``
handlers still apply. Ordinary key and attribute lookups retain their Python
``KeyError`` and ``AttributeError`` behaviour.

.. code-block:: text

   ModelloError
   ├── ModelInputError (also TypeError)
   ├── ModelValidationError (also ValueError)
   │   ├── ModelDefinitionError
   │   ├── SolutionSelectionError
   │   └── UnresolvedSolutionError
   └── ModelSolveError (also ValueError)
       ├── SolutionCountError
       └── SolverBackendError

``SolutionCountError.solution_count`` is the number of retained solver
candidates. ``returned_count`` counts the captured candidates before domain
filtering; these are available through ``problem.candidates``, including rejected
proposals. Each candidate can contain symbolic expressions representing a family
of numerical answers. Neither count establishes how many mathematical solutions
the model has, and a zero retained count alone does not prove inconsistency.

``SolverBackendError.backend_error`` retains the original exception, also
available as ``__cause__``; ``solution_count`` is ``None``. The recorded problem
is available for both failure types. Handle failures through these attributes
rather than parsing exception messages. Constructing or printing an error does
not launch diagnostic analysis.

For example, ``case SolutionCountError(solution_count=0)`` selects a zero-count
failure. Positional ``case SolutionCountError(0)`` is supported too.
``case SolverBackendError(backend_error=NotImplementedError())`` selects a
particular backend exception type. See :doc:`debugging` for a complete workflow.
The :doc:`solution-family guide <solutions>` describes the selection errors and
their retained families, counts, and candidate checks.

.. currentmodule:: modello.errors

.. autoclass:: ModelloError

.. autoclass:: ModelInputError

.. autoclass:: ModelValidationError

.. autoclass:: ModelDefinitionError

.. autoclass:: ModelSolveError

.. autoclass:: SolutionCountError
   :members: returned_count

.. autoclass:: SolverBackendError
