"""Model views measure control and image dimensions without pointwise rank."""

from itertools import combinations

from sympy import Rational, diff

from examples.solution_families import Execution
from examples.solution_views import ModelViews
from modello import InstanceDummy, Modello, inspect_model


def test_model_views_preserve_unknown_dimensions_for_nonlinear_relations():
    class Circle(Modello):
        x = InstanceDummy("x", real=True)
        y = InstanceDummy("y", real=True)
        squared_radius = x**2 + y**2

    p = inspect_model(Circle, "circle", squared_radius=1)
    family = p.solutions()
    x, y = p.symbols.x, p.symbols.y
    reference = Circle("reference", x=1, y=0)
    views = tuple(ModelViews(family, through=reference).sections())
    whole = next(view for view in views if len(view.controls) == 2)
    assert whole.control_dimension is None
    assert whole.image_dimension(x, y) is None
    discrete = next(view for view in views if view.controls == (p.field("x"),))
    assert discrete.control_dimension == discrete.image_dimension(x) == 0


def test_model_views_follow_dependencies_and_measure_images_without_pointwise_rank():
    class Reading(Modello):
        x = InstanceDummy("x", real=True)
        squared = x**2

    problem = inspect_model(Execution, "views", reference=100)
    execution = problem.symbols
    quantity, impact, fee = execution.quantity, execution.impact, execution.fee
    controls = (quantity, impact, fee)
    family = problem.solutions(*controls)
    reference = Execution("reference", quantity=5, reference=100, impact=Rational(1, 2), fee=1)
    views = tuple(ModelViews(family, through=reference).sections())
    assert {tuple(field.symbol for field in view.controls) for view in views} == {
        selected for size in range(1, 4) for selected in combinations(controls, size)
    }
    whole = next(view for view in views if len(view.controls) == 3)
    cost = whole.family.expression(execution.cost)
    assert diff(cost, fee) == 1
    assert diff(cost, impact) == quantity**2
    assert whole.control_dimension == 3
    assert whole.image_dimension(execution.price, execution.cost) == 2
    assert whole.image_dimension(execution.price, execution.price, execution.reference) == 1
    assert whole.image_dimension(execution.reference) == 0
    reading = inspect_model(Reading, "stationary reference")
    x = reading.symbols.x
    bounded = reading.solutions().where(x >= -1, x <= 1)
    view = next(ModelViews(bounded, through=Reading("zero", x=0)).sections())
    assert view.control_dimension == view.image_dimension(reading.symbols.squared) == 1
