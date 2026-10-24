"""The scalar-series guide also composes with numerical model views."""

import pytest
from sympy import Rational

from examples.series import ObservedAutoregression
from examples.symbolic import Known, SymbolicModel
from modello import Vector
from modello.errors import ModelInputError


def test_observed_states_and_innovations_infer_the_same_relative_law():
    case = ObservedAutoregression(
        "inverse", opening=2, retention=Rational(1, 2), innovations=[1, 0, None, 0], closing=Rational(5, 4)
    )
    assert case.N == 4
    assert case.innovations == Vector(1, 0, 2, 0)


def test_numerical_views_rebind_law_parameters_alongside_vector_fields():
    class NumericalSequence(ObservedAutoregression, SymbolicModel):
        pass

    source = NumericalSequence("source", opening=1, innovations=[0, 0, 0])
    result = source.compile().evaluate(Known(source.retention, 2))
    assert tuple(map(float, result.levels)) == (1, 2, 4, 8)
    assert float(result.law.at(3).doit()) == 8
    source.levels = Vector(1, 2)
    with pytest.raises(ModelInputError, match="levels"):
        source.compile()
