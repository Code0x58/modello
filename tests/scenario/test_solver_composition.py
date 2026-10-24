"""Independent schemas, inverse problems, observations, and joint alternatives."""

import pytest
from sympy import Eq, Rational, S, Tuple

from modello import InstanceDummy, Modello, Vector, component, inspect_model, model_vector
from modello.errors import ModelInputError, SolutionCountError


class Measurement(Modello):
    input = InstanceDummy("input", real=True)
    gain = InstanceDummy("gain", positive=True)
    output = gain * input


@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse"])
def test_nested_vectors_retain_each_members_equations_and_independent_identities(solve_model, inverse):
    class Batch(Modello):
        readings = model_vector(Measurement, size=3)
        total = readings.project(lambda reading: reading.output).sum()

    entries = [Measurement("same", input=1, gain=2), Measurement("same", input=3, gain=2), None]
    entries[-1] = Measurement("same", gain=2, **({"output": 10} if inverse else {"input": 5}))
    case = solve_model(Batch, readings=entries)
    assert [(reading.input, reading.output) for reading in case.readings] == [(1, 2), (3, 6), (5, 10)]
    assert case.total == 18
    assert all(isinstance(reading, Measurement) for reading in case.readings)
    problem = inspect_model(Batch, "independent")
    assert problem.symbols.readings[0].gain != problem.symbols.readings[1].gain


def test_nonlinear_vector_observations_keep_joint_alternatives_instead_of_combining_marginals():
    class Pair(Modello):
        values = Vector("x", size=2, real=True)
        total = values.sum()
        product = values.product()

    problem = inspect_model(Pair, "roots", total=5, product=6)
    family = problem.solutions()
    assert {tuple(case.values) for case in family.cases()} == {(2, 3), (3, 2)}
    assert family.values(*problem.symbols.values).contains(Tuple(2, 2)) is S.false
    assert family.where(Eq(problem.symbols.values[0], 2)).one().values == Vector(2, 3)
    with pytest.raises(SolutionCountError) as failure:
        Pair("ambiguous", total=5, product=6)
    assert failure.value.solution_count == 2


def test_partial_observations_can_be_added_in_either_order_and_released_as_a_group():
    class Budget(Modello):
        values = Vector("x", size=3, nonnegative=True)
        total = values.sum()

    problem = inspect_model(Budget, "ordered", total=10)
    left = problem.solutions().observe(values=[2, None, None]).observe(values=[None, 3, None]).one()
    right = problem.solutions().observe(values=[None, 3, None]).observe(values=[2, None, None]).one()
    assert left.values == right.values == Vector(2, 3, 5)
    copied = inspect_model(Budget, "copied", values=[2, 3, None], total=10)
    released = copied.without("values").solutions()
    assert len(released.parameters) == 2
    assert released.observe(values=[1, 2, None]).one().values == Vector(1, 2, 7)


def test_inheritance_and_component_refinement_preserve_vector_formulas(solve_model):
    class Reading(Modello):
        samples = Vector("x", size=2, real=True)
        total = samples.sum()

    class PositiveReading(Reading):
        samples = Vector("x", size=2, positive=True)

    class Experiment(Modello):
        left = component(Reading)
        right = component(Reading)
        combined = left.total + right.total

    class PositiveExperiment(Experiment):
        left = component(PositiveReading)

    case = solve_model(
        PositiveExperiment,
        left=PositiveReading("left", samples=[1, 2]),
        right=Reading("right", samples=[-1, 4]),
    )
    assert case.combined == 6
    assert isinstance(case.left, PositiveReading)
    assert type(case.right) is Reading
    assert case.left.samples == Vector(1, 2)


@pytest.mark.parametrize("scalar", [Rational(1, 3), 2, -3])
def test_an_aggregate_can_infer_a_shared_parameter_across_typed_members(solve_model, scalar):
    class Entry(Modello):
        quantity = InstanceDummy("quantity", real=True)

    class Valuation(Modello):
        price = InstanceDummy("price", real=True)
        entries = model_vector(Entry, size=3)
        total = price * entries.project(lambda entry: entry.quantity).sum()

    entries = [Entry("entry", quantity=value) for value in (1, 2, 4)]
    assert solve_model(Valuation, entries=entries, total=7 * scalar).price == scalar


def test_exact_families_explicitly_reject_approximate_observations():
    class SumModel(Modello):
        values = Vector("x", size=2, real=True)
        total = values.sum()

    assert float(SumModel("numeric", values=[0.1, 0.2]).total) == pytest.approx(0.3)
    with pytest.raises(ModelInputError, match="exact"):
        inspect_model(SumModel, "approximate", values=[0.1, 0.2]).solutions()


@pytest.mark.parametrize("renamed", ["override", "nested-query", "additional-declaration"])
@pytest.mark.parametrize("multiplicative", [False, True], ids=["additive-noise", "multiplicative-noise"])
def test_renamed_laws_expand_named_coefficients_in_the_original_declaration_context(
    solve_model, renamed, multiplicative
):
    from sympy import E

    from modello import Coordinate, ItoProcess

    class Base(Modello):
        x = InstanceDummy("x", positive=True)
        t = Coordinate("t", real=True)
        mu = InstanceDummy("mu", real=True)
        coefficient = mu * x
        process = ItoProcess(x, coefficient, x if multiplicative else 1, t)
        mean = process.mean(1, 1)

    if renamed == "override":

        class Model(Base):
            process = Base.process.as_dummy()
    elif renamed == "nested-query":

        class Model(Base):
            mean = Base.process.as_dummy().mean(1, 1)
    else:

        class Model(Base):
            renamed_process = Base.process.as_dummy()
            mean = renamed_process.mean(1, 1)

    case = solve_model(Model, x=2, mu=1)
    assert case.mean == E
    assert case.coefficient == 2
    law = case.renamed_process if renamed == "additional-declaration" else case.process
    assert law.mean(1, 1).doit() == E


def test_distinct_process_declarations_keep_their_state_dependencies_even_with_a_shared_driver(solve_model):
    from modello import Coordinate, ItoProcess, Series

    class Coupled(Modello):
        x = InstanceDummy("x", positive=True)
        y = InstanceDummy("y", positive=True)
        t = Coordinate("t", real=True)
        W = Series("W", t, real=True)
        mu = InstanceDummy("mu", real=True)
        coefficient = mu * x
        first = ItoProcess(x, coefficient, x, t, W)
        second = ItoProcess(y, coefficient, y, t, W)
        transformed = second.drift(y**2)

    case = solve_model(Coupled, x=2, y=3, mu=1)
    assert case.transformed == 21
    assert case.first.state != case.second.state
    assert case.first.driver.symbol == case.second.driver.symbol


@pytest.mark.parametrize("with_first", [False, True], ids=["alone", "with-unrelated-law"])
def test_adding_an_unrelated_declaration_cannot_change_the_supported_exact_class(with_first):
    from modello import Coordinate, ItoProcess, Series

    class Coupled(Modello):
        x = Coordinate("x", positive=True)
        y = Coordinate("y", positive=True)
        t = Coordinate("t", real=True)
        W = Series("W", t, real=True)
        mu = InstanceDummy("mu", real=True)
        coefficient = mu * x
        if with_first:
            first = ItoProcess(x, coefficient, x, t, W)
        second = ItoProcess(y, coefficient, y, t, W)
        mean = second.mean(1, 1)

    with pytest.raises(ModelInputError, match="coefficients"):
        inspect_model(Coupled, "distinct", mu=1).solutions()


@pytest.mark.parametrize("with_first", [False, True], ids=["alone", "with-unrelated-law"])
def test_distinct_recurrences_do_not_rebind_a_shared_named_coefficient(with_first):
    from modello import Index, Recurrence, Series

    class Separate(Modello):
        n, m = Index("n"), Index("m")
        T = Series("T", n, real=True)
        alias = T.xreplace({n: m})
        coefficient = 2 * n
        if with_first:
            first = Recurrence(T[n], coefficient * T[n - 1], initial={0: 1})
        second = Recurrence(alias[m], coefficient * alias[m - 1], initial={0: 1})
        value = second.at(3)

    case = Separate("separate")
    assert case.value == 8 * Separate.n**3
    assert case.second.target.index == Separate.m
    problem = inspect_model(Separate, "family")
    assert problem.solutions().expression(problem.symbols.value) == 8 * Separate.n**3
