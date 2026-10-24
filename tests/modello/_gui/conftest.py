"""Fresh observations and clients for the shared relative-window contract."""

import pytest
from sympy import Rational

from modello.gui import create_app
from tests.modello._models import DrivenSequence


@pytest.fixture
def relative_inputs():
    return {"retention": Rational(1, 2), "levels": [2, 2, 1, Rational(5, 2), Rational(5, 4)]}


@pytest.fixture
def relative_client(relative_inputs):
    return create_app(DrivenSequence, inputs=relative_inputs).test_client()
