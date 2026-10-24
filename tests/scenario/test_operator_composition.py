"""Native operators compose across data, moving bounds, and symbolic definitions."""

from math import prod

import pytest
from sympy import Derivative, Eq, Integral, Piecewise, Product, Rational, S, Sum, exp, pi, simplify, sqrt

from modello import Coordinate, Index, InstanceDummy, Modello, Vector, inspect_model, vector
from modello.contrib.stochastic import ItoProcess
from modello.errors import ModelInputError


@pytest.mark.parametrize("product", [False, True], ids=["sum", "product"])
@pytest.mark.parametrize("count", [0, 1, 4])
def test_partitioning_an_ordered_reduction_does_not_change_its_value(solve_model, product, count):
    class Reduction(Modello):
        values = Vector("x", size=count, real=True)
        i = Index("i")
        whole = Product(values[i], (i, 0, count - 1)) if product else Sum(values[i], (i, 0, count - 1))
        split = (
            Product(values[i], (i, 0, count // 2 - 1)) * Product(values[i], (i, count // 2, count - 1))
            if product
            else Sum(values[i], (i, 0, count // 2 - 1)) + Sum(values[i], (i, count // 2, count - 1))
        )

    entries = [(-1) ** i * (i + 1) for i in range(count)]
    expected = prod(entries) if product else sum(entries)
    case = solve_model(Reduction, values=entries)
    assert case.whole == case.split == expected


@pytest.mark.parametrize("product", [False, True], ids=["sum", "product"])
def test_two_bound_indices_can_address_independent_observation_vectors(solve_model, product):
    class Rectangle(Modello):
        left = Vector("x", size=2, real=True)
        right = Vector("y", size=3, real=True)
        i, j = Index("i"), Index("j")
        result = (
            Product(left[i] + right[j], (i, 0, 1), (j, 0, 2))
            if product
            else Sum(left[i] * right[j], (i, 0, 1), (j, 0, 2))
        )

    case = solve_model(Rectangle, left=[1, 2], right=[3, 4, 5])
    expected = prod(x + y for x in (1, 2) for y in (3, 4, 5)) if product else 36
    assert case.result == expected


@pytest.mark.parametrize("product", [False, True], ids=["sum", "product"])
def test_reversed_sum_and_product_bounds_follow_sympys_oriented_interval_convention(solve_model, product):
    class Reversed(Modello):
        values = Vector("x", size=3, real=True)
        i = Index("i")
        reversed = Product(values[i], (i, 3, -1)) if product else Sum(values[i], (i, 3, -1))

    assert solve_model(Reversed, values=[1, 2, 4]).reversed == (Rational(1, 8) if product else -7)


def test_a_piecewise_boundary_does_not_apply_the_relative_rule_at_the_initial_position(solve_model):
    class Prefix(Modello):
        inputs = Vector("u", size=4, real=True)
        values = Vector("x", size=4, real=True)
        n = Index("n")
        values = vector(Piecewise((inputs[0], Eq(n, 0)), (values[n - 1] + inputs[n], True)), (n, 0, 3))

    assert solve_model(Prefix, inputs=[1, -2, 4, 8]).values == Vector(1, -1, 3, 11)


def test_differentiating_a_named_integral_retains_the_moving_boundary_term(solve_model):
    class Area(Modello):
        a = InstanceDummy("a", real=True)
        stop = InstanceDummy("stop", positive=True)
        x, y = Coordinate("x", real=True), Coordinate("y", real=True)
        area = Integral(a * x + y, (y, 0, x), (x, 0, stop))
        marginal = Derivative(area, stop)
        sensitivity = Derivative(marginal, a)

    case = solve_model(Area, a=2, stop=3)
    assert (case.area, case.marginal, case.sensitivity) == (Rational(45, 2), Rational(45, 2), 9)


def test_an_improper_integral_can_calibrate_a_positive_model_parameter(solve_model):
    class GaussianMass(Modello):
        rate = InstanceDummy("rate", positive=True)
        x = Coordinate("x", real=True)
        mass = Integral(exp(-rate * x**2), (x, S.NegativeInfinity, S.Infinity))

    assert solve_model(GaussianMass, mass=sqrt(pi) / 2).rate == 4


def test_scalar_product_rule_and_vector_reductions_share_named_model_dependencies(solve_model):
    class Energy(Modello):
        position = InstanceDummy("position", real=True)
        coefficients = Vector("a", size=3, real=True)
        i = Index("i")
        polynomial = Sum(coefficients[i] * position ** (i + 1), (i, 0, 2))
        square = polynomial**2
        first = Derivative(square, position)
        second = Derivative(square, position, 2)

    case = solve_model(Energy, position=0, coefficients=[2, 3, 5])
    assert (case.polynomial, case.square, case.first, case.second) == (0, 0, 0, 8)
    assert simplify(case.second - 8) == 0


@pytest.fixture(scope="module")
def conditional_model():
    class Conditional(Modello):
        mode = InstanceDummy("mode", real=True)
        x, t = Coordinate("x", real=True), Coordinate("t", real=True)
        process = ItoProcess(x, -(x**3), 1, t)
        value = Piecewise((42, mode > 0), (process.mean(1, 1), True))

    return Conditional


def test_observed_piecewise_conditions_skip_an_unsupported_inactive_branch(solve_model, conditional_model):
    assert solve_model(conditional_model, mode=1).value == 42


@pytest.mark.parametrize("mode", [0, -1])
def test_an_unsupported_piecewise_branch_is_reported_when_it_is_selected(conditional_model, mode):
    with pytest.raises(ModelInputError, match="coefficients"):
        inspect_model(conditional_model, "selected", mode=mode).solutions()
