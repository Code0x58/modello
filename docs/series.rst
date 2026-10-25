Scalar recurrence queries
=========================

``Series`` names an indexed scalar family. ``Recurrence`` relates discrete
positions. Their queries are scalar formulas in an ordinary Modello model, so
the same solver can calculate results or infer parameters from observations.

Use this representation for a quantity such as ``T[n]`` when you want to query
positions without choosing a finite horizon. A :doc:`finite model sequence
<model-vectors>` instead solves joint equations over a declared window and can
also express implicit updates and boundary constraints. See
:ref:`sequence-representations` for their overlap and different semantics.

One relative law, many queries
------------------------------

.. code-block:: pycon

   >>> from modello import Index, InstanceDummy, Modello, Recurrence, Series
   >>> class Growth(Modello):
   ...     n = Index("n", nonnegative=True)
   ...     T = Series("T", n, positive=True)
   ...     opening = InstanceDummy("opening", positive=True)
   ...     ratio = InstanceDummy("ratio", positive=True)
   ...     periods = InstanceDummy("periods", integer=True, nonnegative=True)
   ...     law = Recurrence(T[n], ratio * T[n - 1], initial={0: opening})
   ...     closing = T[periods]
   ...     samples = law.window(0, 4)
   >>> case = Growth("forward", opening=1, ratio=2, periods=10)
   >>> case.closing, case.samples
   (1024, [1, 2, 4, 8, 16])
   >>> Growth("inverse", closing=1024, ratio=2, periods=10).opening
   1
   >>> case.law.at(20).doit()
   1048576

The law has no fixed horizon. The query at position ten and the window from
zero through four are separate views. In model definitions and family queries,
``T[k]`` uses its declared recurrence; ``law.at(k)`` explicitly selects one law.
Use the populated instance's law for direct evaluation outside a model.
Explicit queries such as ``other_law.at(n)`` compose directly, including nested
queries evaluated by one ``.doit()``. Unqualified terms from several series need
the model or solution family's law registry;
``family.expression(problem.symbols.T[k])`` retains that context.

Series indices are mathematical positions: ``T[n-1]`` is the predecessor and
``T[-1]`` is position minus one. A series has no length or implicit last entry.
``law.window(first, last)`` creates an inclusive finite :doc:`vector <vectors>`
whose Python position zero represents ``T[first]``.

History and domains
-------------------

A recurrence needs contiguous history ending immediately before ``start``.
Its fixed positive lags must be covered by that history:

.. code-block:: pycon

   >>> n = Index("n")
   >>> F = Series("F", n, integer=True)
   >>> law = Recurrence(F[n], F[n - 1] + F[n - 2], initial={0: 0, 1: 1})
   >>> law.start, law.at(10).doit()
   (2, 55)

History may start at a negative or shifted position. ``law.origin`` is its first
mathematical position; ``law.start`` is the first generated position. Neither is
a Python vector offset. Fixed references to initial positions, such as ``T[0]``,
can also appear in the transition. Queries before the history are undefined.

Series domains apply to all declared initial history and every intermediate
state needed by the query. For example, a positive terminal value cannot excuse
a negative predecessor in a positive series. A later invalid state does not
invalidate an earlier query. A finite model sequence checks its entire declared
window, so these are different problems when that window includes the invalid
state.

Finite causal queries support nonlinear laws. Verified closed forms support
constant first-order affine laws and suitable higher-order linear laws through
SymPy's recurrence solver. Zero and unit coefficients retain their special
cases. Unsupported symbolic queries remain unevaluated; exact solution families
require them to resolve.

Let observations choose the window
----------------------------------

The guide's autoregression declares ``N`` once. State observations have
``N+1`` entries and innovations have ``N``:

.. literalinclude:: ../examples/series.py
   :start-at: class ObservedAutoregression
   :language: python

.. code-block:: pycon

   >>> from examples.series import ObservedAutoregression
   >>> case = ObservedAutoregression("states", retention=1, levels=[1, 3, 6, 10])
   >>> case.N, case.opening, case.innovations
   (3, 1, [2, 3, 4])

Either vector's length, or an explicit ``N``, determines the finite window.
``None`` entries still count toward its length. Observing states and innovations
constrains the same equation; contradictory observations are rejected. To change
the driving data, construct a new case with innovations instead of observed states.
Although the law has no intrinsic horizon, finite inputs constrain which queries
are defined. Relative input positions use the same checked indexing as
:ref:`finite model sequences <sequence-representations>`; they never wrap to the
end of an input vector.

History and observation length are separate. A recurrence of order ``p`` needs
``p`` history positions, but those quantities may be observed separately or
inferred. A vector of length at least ``p`` does not by itself identify unknown
coefficients or innovations. The innovation vector carries no distribution
unless the model declares one.

Fixed finite sums can express a law's relative lags, for example
``Sum(coefficients[j]*T[n-1-j], (j, 0, p-1))`` with ``p`` initial values.
The nested sum uses its own index. Multiple declared series can drive each
other; finite query cycles must decrease towards their initial boundaries.

For validation of existing histories, the :doc:`window validator
<contrib/validation>` reuses typed ``model_recurrence`` laws and checks observed
predecessors instead of generating a series path. An optional
:doc:`pandas adapter <contrib/pandas>` binds DataFrame columns to that same function.


API
---

.. currentmodule:: modello

.. autoclass:: Series(name, index=None, **assumptions)
   :members: index, discrete

.. autoclass:: Recurrence(target, expression, initial, start=None)
   :members: at, window, index, initial, origin, start, equations
