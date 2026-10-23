"""Observation lengths select a finite system from a symbolic declaration."""

import pickle
from copy import deepcopy

import pytest
from sympy import Rational

from examples.relative_series import ObservedAutoregression
from modello import Index, InstanceDummy, Modello, Vector, component, inspect_model, vector
from modello.errors import ModelInputError, SolutionCountError


def test_symbolic_vector_substitution_preserves_coordinate_identities():
    N = InstanceDummy("N", integer=True, nonnegative=True)
    x = Vector("x", size=N, real=True)
    assert x.shape == (N,)
    with pytest.raises(TypeError, match="symbolic length"):
        len(x)
    concrete = x.subs(N, 3)
    assert concrete == x.subs(N, 3) == deepcopy(x).subs(N, 3) == pickle.loads(pickle.dumps(x)).subs(N, 3)
    assert concrete[0] == x.subs(N, 5)[0]
    assert x[1].subs(N, 3) == concrete[1]
    assert x[-1].subs(N, 3) == concrete[-1]


@pytest.mark.parametrize("states", [[1], [1, 3, 6], [1, 3, 6, 10, 15, 21, 28]])
def test_state_lengths_determine_the_horizon_and_infer_innovations(states):
    case = ObservedAutoregression("path", retention=1, levels=states)
    assert isinstance(case, ObservedAutoregression)
    assert case.N == len(states) - 1
    assert case.opening == states[0]
    assert case.closing == states[-1]
    assert case.innovations == Vector([right - left for left, right in zip(states, states[1:])])
    assert case.levels == Vector(states)
    assert ObservedAutoregression.levels.shape == (ObservedAutoregression.N + 1,)


def test_innovation_lengths_and_explicit_sizes_use_the_same_declaration():
    case = ObservedAutoregression("driven", opening=2, retention=Rational(1, 2), innovations=[1, 0, 2, 0])
    assert case.N == 4
    assert case.levels == Vector(2, 2, 1, Rational(5, 2), Rational(5, 4))
    family = inspect_model(ObservedAutoregression, "open", N=4, opening=2, retention=Rational(1, 2)).solutions()
    assert family.expression(family.symbols.N) == 4
    fitted = family.observe(levels=case.levels).one()
    assert fitted.innovations == case.innovations


def test_conflicting_lengths_and_unresolved_sizes_fail_before_scalar_solving():
    with pytest.raises(ModelInputError, match="conflict"):
        ObservedAutoregression("lengths", levels=[1, 2, 3], innovations=[1, 2, 3])
    with pytest.raises(ModelInputError, match="conflict"):
        ObservedAutoregression("size", N=4, levels=[1, 2, 3])
    with pytest.raises(ModelInputError, match="determine N"):
        inspect_model(ObservedAutoregression, "unshaped")
    with pytest.raises(ModelInputError):
        ObservedAutoregression("empty", levels=[])
    with pytest.raises(SolutionCountError):
        ObservedAutoregression("values", retention=1, levels=[1, 2, 3], innovations=[10, 10])


def test_symbolic_shapes_preserve_comprehensions_reductions_and_aliases():
    class Window(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        i = Index("i")
        samples = Vector("x", size=N, real=True)
        alias = samples
        doubled = vector(2 * samples[i], (i, 0, N - 1))
        total = samples.sum()

    result = Window("window", samples=[1, None, 3, 4], total=10)
    assert result.N == 4
    assert result.alias == result.samples == Vector(1, 2, 3, 4)
    assert result.doubled == Vector(2, 4, 6, 8)
    assert Window("empty", samples=[]).total == 0
    with pytest.raises(ModelInputError, match="conflict"):
        Window("aliases", samples=[1, 2], alias=[1, 2, 3])


def test_component_instances_supply_independent_symbolic_shapes():
    class Pair(Modello):
        left = component(ObservedAutoregression)
        right = component(ObservedAutoregression)
        total = left.closing + right.closing

    left = ObservedAutoregression("left", retention=1, levels=[1, 2, 3])
    right = ObservedAutoregression("right", retention=1, levels=[2, 4, 6, 8, 10])
    problem = inspect_model(Pair, "pair", left=left, right=right)
    result = problem.solutions().one()
    assert (result.left.N, result.right.N, result.total) == (2, 4, 13)
    assert problem.symbols.left.innovations[0] != problem.symbols.right.innovations[0]
