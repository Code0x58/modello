Start modelling
===============

Declare quantities and their relationships, then supply the observations you
have. The same model can calculate a result, infer a missing input, or retain a
formula while a quantity is unknown. There is no separate inverse model.

This walkthrough builds up four ways to express a problem: scalar quantities,
components, vectors, and recurrent states. It uses basic Python and SymPy's
mathematical notation.
Install with Python 3.10 or newer:

.. code-block:: console

   $ pip install modello

1. Relate scalar quantities
---------------------------

What length of diagonal brace spans a rectangular opening? Declare its two
sides and the relationship that determines the diagonal:

>>> from sympy import sqrt
>>> from modello import InstanceDummy, Modello
>>> class Triangle(Modello):
...     a = InstanceDummy("a", positive=True)
...     b = InstanceDummy("b", positive=True)
...     c = sqrt(a**2 + b**2)
>>> Triangle("brace", a=3, b=4).c
5

``InstanceDummy`` declares a quantity to observe or infer. Assumptions such as
``positive=True`` state its domain. ``"brace"`` labels this case.

To ask the inverse question, supply the diagonal instead of a side:

>>> Triangle("run", b=8, c=10).a
6

To try different measurements, construct another case:

>>> Triangle("larger", a=6, b=8).c
10

Omitting an input keeps the result symbolic:

>>> undecided = Triangle("open", b=4)
>>> undecided.c == sqrt(undecided.a**2 + 16)
True

Results are ordinary SymPy expressions. A new construction answers a new
question; assigning an attribute afterward does not recalculate the model.

2. Compose models
-----------------

A parcel's full diagonal depends on its base diagonal and height. Reuse the
triangle as a typed component rather than repeating its equation:

>>> from modello import component
>>> class Box(Modello):
...     base = component(Triangle)
...     height = InstanceDummy("height", positive=True)
...     diagonal = sqrt(base.c**2 + height**2)
>>> base = Triangle("base", a=3, b=4)
>>> Box("parcel", base=base, height=12).diagonal
13

A parent observation can determine an unknown inside a child. Specify exactly
13 units of diagonal clearance and infer the base width:

>>> parcel = Box("width", base=Triangle("base", b=4), height=12, diagonal=13)
>>> parcel.base.a
3

The parent and children are one system of relationships. ``parcel.base`` is
still a ``Triangle``. See :doc:`nesting` for connecting several components and
reusing partial observations.

3. Model collections and fit parameters
---------------------------------------

For a line through sensor readings, express predictions for each observation,
then combine their residuals into an error objective:

>>> from sympy import Derivative
>>> from modello import Index, Vector, vector
>>> class Calibration(Modello):
...     N = InstanceDummy("N", integer=True, positive=True)
...     times = Vector("t", size=N, real=True)
...     observations = Vector("y", size=N, real=True)
...     slope = InstanceDummy("slope", real=True)
...     intercept = InstanceDummy("intercept", real=True)
...     i = Index("i")
...     fitted = vector(intercept + slope * times[i], (i, 0, N - 1))
...     residuals = observations - fitted
...     squared_error = residuals.dot(residuals)
...     slope_gradient = Derivative(squared_error, slope)
...     intercept_gradient = Derivative(squared_error, intercept)
>>> readings = dict(times=[0, 1, 2, 3], observations=[1, 2, 2, 4])
>>> proposed = Calibration("proposed", **readings, slope=1, intercept=1)
>>> proposed.fitted, proposed.squared_error
([1, 2, 3, 4], 1)

Observation lengths determine ``N``. ``Index`` is a local integer coordinate;
``vector`` includes both endpoints. Another dataset uses the same declaration.

Fitting is another question for this model. Leave the coefficients unknown and
require both error gradients to be zero. With distinct times, these conditions
minimise the sum of squared errors:

>>> fitted = Calibration("fit", **readings, slope_gradient=0, intercept_gradient=0)
>>> fitted.slope, fitted.intercept, fitted.squared_error
(9/10, 9/10, 7/10)

The readings need not lie exactly on a line. Approximation comes from the
objective you declared; Modello does not silently relax equations. See
:doc:`vectors` for missing entries, reductions, and other symbolic formulas.

4. Relate states across a sequence
----------------------------------

When each position contains several related quantities, use a vector of models.
An account has inventory and cash; each fill has quantity, price, and fee:

>>> from modello import ModelObservations, model_recurrence, model_vector
>>> class Balance(Modello):
...     quantity = InstanceDummy("quantity", real=True)
...     cash = InstanceDummy("cash", real=True)
>>> class Fill(Modello):
...     quantity = InstanceDummy("quantity", real=True)
...     price = InstanceDummy("price", positive=True)
...     fee = InstanceDummy("fee", nonnegative=True)
...     notional = quantity * price
>>> class Account(Modello):
...     N = InstanceDummy("N", integer=True, nonnegative=True)
...     fills = model_vector(Fill, size=N)
...     states = model_vector(Balance, size=N + 1)
...     n = Index("n")
...     law = model_recurrence(
...         states[n],
...         quantity=states[n - 1].quantity + fills[n - 1].quantity,
...         cash=states[n - 1].cash - fills[n - 1].notional - fills[n - 1].fee,
...     )
...     terminal = states[-1]
>>> opening = Balance("opening", quantity=3, cash=1000)
>>> buy = Fill("buy", quantity=2, price=100, fee=1)
>>> sale = Fill("sale", quantity=-1, price=110, fee=1)
>>> posted = Account("posted", fills=[buy, sale], states=ModelObservations({0: opening}))
>>> posted.terminal.quantity, posted.terminal.cash
(4, 908)

There are ``N`` fills and ``N+1`` account boundaries. ``n-1`` means the previous
position. ``ModelObservations`` supplies the opening boundary without fixing the
window length; adding a fill creates another state.

The same recurrence can infer an unreported sale price from closing cash:

>>> reconciled = Account(
...     "reconciled", fills=[buy, Fill("sale", quantity=-1, fee=1)],
...     states=ModelObservations({0: opening}),
...     terminal=Balance("statement", cash=908),
... )
>>> reconciled.fills[1].price
110

Without that closing observation, cash remains a formula in the missing price.
See :doc:`model-vectors` for partial states and boundary equations. A scalar
recurrence can instead answer position queries without choosing a finite state
window; :ref:`sequence-representations` explains when to use each representation.

Keep alternatives and check observations
----------------------------------------

When observations leave several possibilities, keep a solution family and
narrow it with more information:

>>> from modello import inspect_model
>>> family = inspect_model(Triangle, "possible", a=3).solutions()
>>> family.observe(c=5).one().b
4

``observe`` adds constraints; it does not replace earlier observations. Start a
new case for a different dataset. Solution families require exact observations;
use SymPy ``Rational(9, 10)`` or ``Rational("0.9")`` for fractional or decimal
data. See :doc:`solutions` for bounds, alternatives, and expressions in the
remaining unknowns.

Extra observations can check a model against measurements:

>>> report = inspect_model(Triangle, "check", a=3, b=4, c=4).analyse()
>>> report.status
'inconsistent'

For larger problems, inspect the equations and reported conflict before deciding
which observation to correct. A solver that cannot resolve a question has not
necessarily proved a contradiction. The :doc:`debugging guide <debugging>` and
:doc:`solver boundaries <generality>` explain these outcomes when needed.

Try the same questions interactively
------------------------------------

The :doc:`gallery <examples/gallery/index>` follows these four concepts. Each
example offers the same questions: calculate, infer, change inputs, leave
unknowns, and check observations. Its :doc:`optional explorer <gui>` displays
the formulas and lets you change the supplied observations.

For domain-specific formulas, see the opt-in :doc:`stochastic extension
<contrib/stochastic>`. The :doc:`API reference <api>` covers individual methods,
input types, and model refinement.
