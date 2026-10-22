"""Shared problem records, candidate lookup, and symbolic field labels."""

import json

import pytest
from sympy import (
    Dummy,
    Symbol,
)

from modello import (
    InstanceDummy,
    ModelInputError,
    Modello,
    ModelSolveError,
    ModelValidationError,
    component,
    inspect_model,
)
from modello._problem import Candidate, ConstraintKind, FieldValue

from ._models import Check, Reading


def test_error_labels_preserve_noncommutative_relationships():
    """Readable names must not turn an ordered product difference into zero."""

    class Ordered(Modello):
        scale = InstanceDummy("scale", positive=True)
        x = InstanceDummy("x", commutative=False)
        y = InstanceDummy("y", commutative=False)
        commutator = x * y - y * x

    with pytest.raises(ModelSolveError, match="0 solutions") as error:
        Ordered("O", scale=0)
    assert "commutator = x*y - y*x" in str(error.value.problem)


def test_input_constraint_keeps_string_enum_compatibility():
    direct = inspect_model(Reading, "direct", x=2)
    assert direct.constraints[-1].kind is ConstraintKind.INPUT
    assert str(direct.constraints[-1].kind) == "input"
    assert f"{direct.constraints[-1].kind:>7}" == "  input"
    assert json.dumps(direct.constraints[-1].kind) == '"input"'
    assert "[input 'x']" in str(direct)


def test_candidate_lookup_rejects_conflicting_alias_values(positive_problem):
    p = positive_problem
    candidate = Candidate((FieldValue(p.field("x"), 1), FieldValue(p.field("alias"), 2)))
    with pytest.raises(ModelValidationError, match="aliases"):
        candidate.value(p.field("x"))


def test_candidate_lookup_rejects_fields_from_another_problem(positive_problem):
    p = positive_problem
    other = inspect_model(p.model, p.name)
    candidate = Candidate((FieldValue(p.field("x"), 1),))
    with pytest.raises(KeyError):
        candidate.value(other.field("x"))


def test_candidate_lookup_accepts_recorded_aliases(positive_problem):
    candidate = Candidate((FieldValue(positive_problem.field("alias"), 1),))
    assert candidate.value(positive_problem.field("x")) == 1


@pytest.mark.parametrize("value", [pytest.param({}, id="mapping"), pytest.param(object(), id="opaque-object")])
def test_field_values_require_scalars(positive_problem, value):
    with pytest.raises(ModelInputError, match="scalar"):
        FieldValue(positive_problem.field("x"), value)


@pytest.mark.parametrize(
    "reference, left, right, offset",
    [
        pytest.param(("review", "check", "total"), "reading.x", "total - 1", -1, id="same-component"),
        pytest.param(("review", "x"), "check.reading.x", "x + 1", 1, id="parent-component"),
        pytest.param(("review", "peer", "x"), "check.reading.x", "peer.x + 1", 1, id="sibling-component"),
        pytest.param(("x",), "review.check.reading.x", "x + 1", 1, id="root"),
    ],
)
def test_cross_layer_equations_share_the_nearest_common_ancestor(reference, left, right, offset):
    class Review(Modello):
        check = component(Check)
        peer = component(Reading)
        x = InstanceDummy("x", real=True)

    class Batch(Modello):
        review = component(Review)
        x = InstanceDummy("x", real=True)

    expression = Batch
    for part in reference:
        expression = getattr(expression, part)
    source = Reading("source", x=expression + offset)
    check = Check("check", reading=source)
    review = Review("review", check=check)
    problem = inspect_model(Batch, "batch", review=review)
    rendered = str(problem)
    assert f"{left} = {right} [input 'review']" in rendered
    assert "squared = x**2 [definition]" in rendered
    # Rendering changes neither recorded paths nor the expressions they identify.
    copied = next(c for c in problem.constraints if c.path == ("review", "check", "reading", "x"))
    assert copied.right == problem.field(*reference).symbol + offset
    assert problem.format(copied.right) == f"{'.'.join(reference)} {'+' if offset > 0 else '-'} 1"


def test_bare_field_paths_cannot_be_confused_with_external_symbol_identities():
    positive = Symbol("x", positive=True)
    negative = Symbol("x", negative=True)
    independent = Dummy("x", real=True)
    dotted = Symbol("reading.x", real=True)

    class Names(Modello):
        x = InstanceDummy("x", real=True)
        reading = component(Reading)
        total = x + positive - negative + independent + dotted

    p = inspect_model(Names, "names")
    assert p.format(p.field("x").symbol) == "x"
    assert p.format(p.field("reading", "x").symbol) == "reading.x"
    assert p.format(dotted) == "external[reading.x]"
    labels = {p.format(symbol) for symbol in (positive, negative, independent)}
    assert labels == {"external[x#1]", "external[x#2]", "external[x#3]"}
    rendered = str(p)
    assert all(label in rendered for label in labels)
    assert "self." not in rendered
    assert p.constraints[0].right == p.field("x").symbol + positive - negative + independent + dotted
