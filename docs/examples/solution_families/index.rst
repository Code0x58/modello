Bound a prediction from an uncertain observation
=================================================

A recorded execution cost may be known only within a range. Instead of choosing
a fitted parameter, keep every parameter consistent with that observation and
carry the resulting uncertainty into a planned execution. This small example
explores solution families; its linear impact law is illustrative.

Define an execution's mathematics once:

.. literalinclude:: ../../../examples/solution_families.py
   :pyobject: Execution

Join two executions through the same impact coefficient:

.. literalinclude:: ../../../examples/solution_families.py
   :start-at: class ImpactStudy
   :end-before:     @classmethod

Fix the quantities, reference prices, and fees. Impact remains unknown:

.. literalinclude:: ../../../examples/solution_families.py
   :pyobject: ImpactStudy.prepare

>>> from examples.solution_families import ImpactStudy
>>> problem = ImpactStudy.prepare()
>>> study = problem.symbols
>>> family = problem.solutions(study.impact)

Keep impact as the exploration coordinate. Constrain the recorded cost to its
observed interval, then ask what that means for the planned trade:

>>> calibrated = family.where(study.recorded.cost >= 200, study.recorded.cost <= 202)
>>> calibrated.values(study.impact)
Interval(1/4, 3/4)

The joining model carries the calibration into the prediction through one
shared quantity. ``ImpactStudy.calibrated()`` packages these preparation and
observation steps into a prediction that is ready to query:

>>> prediction = ImpactStudy.calibrated()
>>> prediction.values(prediction.symbols.planned.cost)
Interval(1026, 1076)
>>> prediction.expression(prediction.symbols.planned.cost) == 100 * prediction.symbols.impact + 1001
True

There is no statistical confidence level attached to this interval; it follows
from the specified bounds and the model's assumptions.

Select a particular impact to obtain a normal typed model:

>>> from sympy import Rational
>>> case = prediction.observe(impact=Rational(1, 2)).one()
>>> case.recorded.cost, case.planned.cost
(201, 1051)
>>> case.recorded.impact == case.planned.impact
True

Plot the same expression over its admissible interval with
``ImpactStudy.plot(prediction, Path("cost.png"))``. From a checkout, the
notebook dependency group supplies the optional Matplotlib backend:

.. code-block:: console

   $ uv run --group notebooks python -m examples.solution_families --plot cost.png

Running ``python -m examples.solution_families`` prints the intervals with only
the library's existing SymPy dependency. The plot is a view of the bounded
family; numerical plotting does not establish additional guarantees.

.. image:: impact-cost.png
   :alt: Planned execution cost increases from 1026 to 1076 as impact varies from one quarter to three quarters.

Explore combinations of controls
--------------------------------

Reuse ``Execution`` with a fixed reference price and a reference execution:

>>> from modello import inspect_model
>>> from examples.solution_families import Execution
>>> from examples.solution_views import ModelViews
>>> problem = inspect_model(Execution, "views", reference=100)
>>> execution = problem.symbols
>>> family = problem.solutions(execution.quantity, execution.impact, execution.fee)
>>> reference = Execution("reference", quantity=5, reference=100, impact=Rational(1, 2), fee=1)

First vary quantity alone. Impact and fee stay at the reference values; cost
changes through the same equations:

>>> quantity = family.section(execution.quantity, through=reference)
>>> quantity.expression(execution.cost).expand() == 100 * execution.quantity + execution.quantity**2 / 2 + 1
True

``ModelViews.sections()`` lazily enumerates one-, two-, and three-control
combinations: seven sections here.

>>> views = ModelViews(family, through=reference)
>>> together = tuple(views.sections())[-1]
>>> together.control_dimension, together.image_dimension(execution.price, execution.cost)
(3, 2)

Three controls produce only two independent price/cost dimensions. Fee shifts
cost uniformly, while impact sensitivity grows with quantity squared. These
named coordinate combinations give a finite catalogue of views.

Control dimension is certified for explicit real interval boxes and finite real
point sets. Polynomial outputs with rational coefficients have an exact image
dimension on boxes, using SymPy's
`DomainMatrix <https://docs.sympy.org/latest/modules/polys/domainmatrix.html>`_
Jacobian rank over a rational-function field
(`Geiger et al., Lemma 9 and Theorem 10 <https://doi.org/10.1214/aos/1009210550>`_).
This concerns the whole image: ``x**2`` on ``[-1, 1]`` has dimension one despite
its zero derivative at zero. Unsupported or coupled domains remain ``None``;
general real algebraic dimension needs stronger methods
(`Lairez and Safey El Din, 2021 <https://doi.org/10.1145/3452143.3465551>`_).

Export the execution sections with the optional notebook dependencies:

.. code-block:: console

   $ uv run --group notebooks python -m examples.solution_views --plot-dir views

Open ``views/index.html`` for curves, two-control response surfaces, and
three-control samples coloured by a fourth response quantity.
``ModelView.plot(...)`` takes typed ``lower`` and ``upper`` executions for its
sampling windows; plots do not certify dimensions or feasibility.
