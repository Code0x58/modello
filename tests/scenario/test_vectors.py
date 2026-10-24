"""Ordered observations, contractions, unknown sizes, and interchangeable formulas."""

from math import prod

import pytest
from sympy import I, Rational, Symbol, conjugate, simplify

from modello import Index, InstanceDummy, Modello, Vector, inspect_model, vector
from modello.errors import ModelInputError


@pytest.mark.parametrize("entries", [(), (0,), (-2, 0, 3), (Rational(1, 2), -3, 4), (1 + I, 2 - I)])
def test_vector_algebra_agrees_with_scalar_arithmetic_for_empty_real_and_complex_values(entries):
    """No domain, length, or zero-factor choice changes the contraction rules."""
    values = Vector(entries)
    assert values.sum().doit() == sum(entries)
    assert values.product().doit() == prod(entries)
    assert simplify(values.dot(values).doit() - sum(entry**2 for entry in entries)) == 0
    assert (values + values).doit() == Vector([2 * entry for entry in entries])
    assert (3 * values - values / 2).doit() == Vector([Rational(5, 2) * entry for entry in entries])
    assert (-values).doit() == Vector([-entry for entry in entries])


def test_complex_dot_is_bilinear_and_hermitian_contraction_is_explicit():
    values = Vector(1 + I, 2 - I)
    i = Index("i")
    hermitian = vector(conjugate(values[i]), (i, 0, 1)).dot(values)
    assert simplify(values.dot(values).doit() - (3 - 2 * I)) == 0
    assert simplify(hermitian.doit() - 7) == 0


@pytest.mark.parametrize("count", [0, 1, 2, 5, 9])
def test_observation_length_and_one_total_determine_partial_vectors(solve_model, count):
    class Window(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        values = Vector("x", size=N, real=True)
        total = values.sum()
        doubled = 2 * values

    expected = [i - 3 for i in range(count)]
    observed = [*expected]
    if count:
        observed[count // 2] = None
    case = solve_model(Window, values=observed, total=sum(expected))
    assert case.N == count
    assert case.values == Vector(expected)
    assert case.doubled == Vector([2 * value for value in expected])


def test_two_independent_size_fields_are_determined_by_joint_length_equations(solve_model):
    class Streams(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        M = InstanceDummy("M", integer=True, nonnegative=True)
        wide = Vector("w", size=N + M, real=True)
        narrow = Vector("v", size=N - M, real=True)
        total = wide.sum() + narrow.sum()

    case = solve_model(Streams, wide=[1, 2, 3, 4, 5], narrow=[7])
    assert (case.N, case.M, case.total) == (3, 2, 22)


def test_mathematically_equal_symbolic_shapes_allow_elementwise_arithmetic():
    """Shape checking must not depend on expanded versus factored syntax."""
    N = Symbol("N", integer=True, nonnegative=True)
    left = Vector("x", size=2 * (N + 1))
    right = Vector("y", size=2 * N + 2)
    combined = left + right
    assert combined.subs(N, 1).doit() == (left.subs(N, 1) + right.subs(N, 1)).doit()


@pytest.mark.parametrize("left, right", [(Vector(1), Vector(1, 2)), (Vector(1), 1)])
def test_shape_mismatch_and_scalar_addition_are_rejected_instead_of_broadcasting(left, right):
    with pytest.raises(ModelInputError):
        left + right


@pytest.mark.parametrize("position", [0, 1, -1])
def test_an_observed_index_selects_the_same_entry_before_and_after_shape_preparation(solve_model, position):
    class Reading(Modello):
        N = InstanceDummy("N", integer=True, positive=True)
        samples = Vector("x", size=N, real=True)
        position = InstanceDummy("position", integer=True)
        selected = samples[position]

    case = solve_model(Reading, samples=[3, 7, 11], position=position)
    assert case.selected == [3, 7, 11][position]


def test_a_zero_product_does_not_identify_an_unobserved_factor():
    class ProductModel(Modello):
        factors = Vector("f", size=2, real=True)
        product = factors.product()

    family = inspect_model(ProductModel, "zero", factors=[0, None], product=0).solutions()
    assert len(family.parameters) == 1
    assert family.expression(family.symbols.product) == 0
    assert family.observe(factors=[None, 7]).one().factors == Vector(0, 7)


def test_complex_observations_support_inverse_coordinates_and_explicit_hermitian_norms(solve_model):
    class ComplexPair(Modello):
        values = Vector("z", size=2, complex=True)
        total = values.sum()
        i = Index("i")
        squared_norm = vector(conjugate(values[i]), (i, 0, 1)).dot(values)
        bilinear = values.dot(values)

    case = solve_model(ComplexPair, values=[1 + I, None], total=3 - 2 * I)
    assert case.values == Vector(1 + I, 2 - 3 * I)
    assert simplify(case.squared_norm - 15) == 0
    assert simplify(case.bilinear - (-5 - 10 * I)) == 0


def test_noncommutative_vector_products_preserve_the_declared_entry_order():
    A, B = Symbol("A", commutative=False), Symbol("B", commutative=False)
    values = Vector(A, B)
    assert values.product().doit() == A * B
    assert Vector(*reversed(values)).product().doit() == B * A
    assert (values * A).doit() == Vector(A**2, B * A)
    assert (A * values).doit() == Vector(A**2, A * B)
