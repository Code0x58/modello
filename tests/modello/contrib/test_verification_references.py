"""Typed field references select inputs; a read-only field view resolves values."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest
from sympy import Derivative, FiniteSet, Lambda, S

from modello import Index, InstanceDummy, Modello, Recurrence, Series, SolutionFamily, Vector, component, model_vector
from modello.contrib.verification import VerificationContext
from modello.errors import ModelDefinitionError, ModelInputError, ModelValidationError


class Reading(Modello):
    x = InstanceDummy("x", real=True)
    y = 2 * x


class Study(Modello):
    reading = component(Reading)
    x_alias = reading.x


def test_reference_observations_reuse_native_component_provenance_and_aliases():
    context = VerificationContext.for_model(Study)
    symbols = context.symbols
    observed = context.observe({symbols.reading: Reading("source", x=3)})
    assert observed.model.reading.y == 6
    assert observed.model.x_alias == 3
    assert context.model.x_alias == symbols.x_alias
    assert {item.input for item in observed.problem.constraints if item.input} == {"reading"}
    assert not any(item.input for item in observed.problem.without("reading").constraints)

    scalar = context.observe({symbols.reading.x: 4})
    assert scalar.model.reading.y == 8
    assert scalar.problem.constraints[-1].input == "x_alias"
    with pytest.raises(ModelInputError, match="More than one observation"):
        context.observe({symbols.reading.x: 4}, x_alias=5)


def test_references_are_scoped_hashable_keys_and_replacements_leave_the_source_intact():
    class Changed(Reading):
        y = -Reading.x

    context = VerificationContext.for_model(Study)
    other = VerificationContext.for_model(Study)
    names = {context.symbols.reading: "current", other.symbols.reading: "other"}
    assert names[context.symbols.reading] == "current"
    assert names[other.symbols.reading] == "other"
    observed = context.observe({context.symbols.reading.x: 3})
    changed = observed.replace(observed.symbols.reading, Changed)
    assert changed.model.reading.y == -3
    assert observed.model.reading.y == 6
    assert changed.model.reading.x == 3
    with pytest.raises(ModelValidationError, match="component references"):
        context.replace(other.symbols.reading, Changed)
    with pytest.raises(ModelValidationError, match="component references"):
        context.observe({other.symbols.reading: Reading("foreign", x=1)})
    with pytest.raises(ModelValidationError, match="symbols"):
        context.observe({other.symbols.reading.x: 1})
    with pytest.raises(ModelValidationError, match="component references"):
        changed.replace(observed.symbols.reading, Reading)
    with pytest.raises(ModelDefinitionError, match="Select a component"):
        context.replace(context.symbols.reading.x, Changed)


def test_resolved_fields_are_read_only_and_do_not_become_input_selectors():
    context = VerificationContext.for_model(Study)
    first = context.observe({context.symbols.reading.x: 3})
    second = context.observe({context.symbols.reading.x: 4})
    first_view = first.model
    assert first.model is first_view
    assert first_view.reading.x == 3
    assert second.model.reading.x == 4
    with pytest.raises(FrozenInstanceError):
        first_view.reading.x = 5
    with pytest.raises(AttributeError, match="has no model attribute"):
        getattr(first_view.reading, "unknown")
    with pytest.raises(ModelInputError, match="selected through"):
        first.observe({first_view.reading.x: 5})
    with pytest.raises(ModelInputError, match="Pass a mapping"):
        first.observe([])
    assert "reading" in dir(first_view)


def test_component_ownership_does_not_depend_on_having_scalar_fields():
    class Empty(Modello):
        pass

    class Container(Modello):
        child = component(Empty)

    context = VerificationContext.for_model(Container)
    other = VerificationContext.for_model(Container)
    assert context.symbols.child != other.symbols.child
    observed = context.observe({context.symbols.child: Empty("empty")})
    # Added observations retain the original problem's reference namespace.
    observed.replace(context.symbols.child, Empty)
    observed.replace(deepcopy(context.symbols.child), Empty)
    with pytest.raises(ModelValidationError, match="component references"):
        observed.replace(other.symbols.child, Empty)


def test_resolved_fields_keep_the_solution_family_instead_of_selecting_one(monkeypatch):
    class Quadratic(Modello):
        x = InstanceDummy("x", real=True)
        squared = x**2

    def unexpected(*args, **kwargs):
        raise AssertionError("Reading a field must not select a solution")

    monkeypatch.setattr(SolutionFamily, "one", unexpected)
    context = VerificationContext.for_model(Quadratic)
    observed = context.observe({context.symbols.squared: 1})
    assert observed.model.squared == observed.expression(observed.symbols.squared)
    assert observed.model.x == observed.expression(observed.symbols.x)
    assert observed.family.values(observed.symbols.x) == FiniteSet(-1, 1)


def test_values_keep_calculus_scopes_and_expression_can_still_protect_a_coordinate():
    class Curve(Modello):
        t = InstanceDummy("t", real=True)
        a = InstanceDummy("a", real=True)
        value = a * t**2
        slope = Derivative(value, t, evaluate=False)

    context = VerificationContext.for_model(Curve)
    symbols = context.symbols
    observed = context.observe({symbols.a: 2, symbols.t: 3})
    assert observed.model.value == 18
    assert observed.model.slope.doit() == 12
    assert observed.expression(symbols.value, wrt=symbols.t) == 2 * symbols.t**2


def test_nested_values_reuse_vectors_laws_and_model_collection_views():
    class Growth(Modello):
        n = Index("n")
        T = Series("T", n, real=True)
        opening = InstanceDummy("x", real=True)
        ratio = InstanceDummy("a", real=True)
        law = Recurrence(T[n], ratio * T[n - 1], initial={0: opening})
        terminal = law.at(3)
        samples = Vector(opening, ratio * opening)

    class Rows(Modello):
        n = Index("n", nonnegative=True)
        rows = model_vector(Growth, size=2)

    context = VerificationContext.for_model(Rows)
    observed = context.observe(
        {
            context.symbols.rows[0]: Growth("first", opening=1, ratio=2),
            context.symbols.rows[1]: Growth("second", opening=3, ratio=2),
        }
    )
    assert observed.model.rows[0].terminal.doit() == 8
    assert observed.model.rows[1].law.at(3).doit() == 24
    assert observed.model.rows[1].samples == Vector(3, 6)
    assert observed.model.rows.project(lambda row: row.opening) == Vector(1, 3)
    assert tuple(row.opening for row in observed.model.rows) == (1, 3)
    assert observed.model.rows[:1][0].opening == 1
    selected = observed.model.rows[observed.symbols.n].opening
    assert Lambda(observed.symbols.n, selected)(S.One).doit() == 3
    assert len(observed.model.rows) == 2
    assert bool(observed.model.rows) is True
    assert observed.model.rows[0].n == context.symbols.rows[0].n
    with pytest.raises(ModelInputError, match="selected through"):
        observed.observe({S.One: 2})
