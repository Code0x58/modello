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
   types raise :class:`modello.errors.ModelInputError`, also a :class:`TypeError`.
   No accepted solver candidates or multiple candidates raise
   :class:`modello.errors.SolutionCountError`; a backend exception raises
   :class:`modello.errors.SolverBackendError`. Both are
   :class:`modello.errors.ModelSolveError` and :class:`ValueError` subclasses.
   Symbol assumptions help select valid
   solutions; assumptions about external symbolic parameters may remain
   undecided. Each explicitly constructed input model must have a valid solution
   before it can be passed to a parent.

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

   Supply the same typed inputs as construction. No solver runs during
   preparation; explicitly constructed component inputs must already be valid.
   See :doc:`diagnostics` for the structured snapshot, report types, and guarantees.
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
filtering; these are available through ``problem.candidates``. Either count can
refer to symbolic families, rather than individual numerical points. A zero
retained count alone is not a proof of inconsistency.

``SolverBackendError.backend_error`` retains the original exception, also
available as ``__cause__``; ``solution_count`` is ``None``. The recorded problem
is available for both failure types. No counts or categories are extracted by
parsing exception messages, and constructing or printing an error does not
launch diagnostic analysis.

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
