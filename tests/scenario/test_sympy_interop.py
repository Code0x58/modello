"""Native expressions and containers cross the model boundary without losing meaning."""

from copy import deepcopy

import pytest
from sympy import (
    Array,
    Derivative,
    Eq,
    Function,
    ImmutableMatrix,
    Lambda,
    Limit,
    Matrix,
    MatrixSymbol,
    Mul,
    MutableDenseNDimArray,
    Pow,
    Rational,
    S,
    SparseMatrix,
    Symbol,
    Tuple,
    lambdify,
    oo,
    solve,
    sympify,
)

from modello import (
    Coordinate,
    Index,
    InstanceDummy,
    Modello,
    Recurrence,
    Series,
    Vector,
    component,
    inspect_model,
    model_recurrence,
    model_vector,
    vector,
)
from modello.errors import ModelDefinitionError, ModelInputError, SolutionCountError


@pytest.fixture(
    params=[tuple, Array, MutableDenseNDimArray, Matrix, ImmutableMatrix, "row", "sparse"],
    ids=["tuple", "array", "mutable-array", "column", "immutable-column", "row", "sparse-column"],
)
def native_vector(request):
    if request.param == "row":
        return lambda entries: Matrix(1, len(entries), entries)
    if request.param == "sparse":
        return lambda entries: SparseMatrix(len(entries), 1, entries)
    return request.param


@pytest.mark.parametrize("entries", [(), (2,), (Rational(1, 2), -3, 4)])
def test_native_vector_round_trips_retain_order_and_values(native_vector, entries):
    native = native_vector(entries)
    converted = Vector(native)
    assert converted == Vector(entries)
    assert Vector(Array(converted)) == converted
    assert Vector(Matrix(converted)) == converted
    assert deepcopy(converted) == converted


def test_native_vector_round_trip_retains_independent_equal_named_quantities(native_vector):
    first, second = InstanceDummy("x"), InstanceDummy("x")
    converted = Vector(native_vector((first, second)))
    assert first != second
    assert converted.elements == (first, second)
    assert Vector(Array(converted)).elements == (first, second)


@pytest.mark.parametrize("entries", [(), (2,), (1, 2, 3)])
def test_native_observations_determine_shape_and_use_the_same_solver(native_vector, entries, solve_model):
    class Window(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        values = Vector("x", size=N, real=True)
        total = values.sum()

    case = solve_model(Window, values=native_vector(entries))
    assert case.N == len(entries)
    assert case.values == Vector(entries)
    assert case.total == sum(entries)


def test_native_declarations_are_vectors_and_mutable_inputs_are_snapshotted():
    a = Symbol("a", real=True)
    source = MutableDenseNDimArray([a, a + 1])

    class Declared(Modello):
        values = source
        total = values.sum()

    source[0] = 100
    problem = inspect_model(Declared, "declaration")
    assert isinstance(Declared.values, Vector)
    assert Declared("symbolic").total == 2 * a + 1
    assert problem.vector("values").shape == (2,)

    observed = Matrix([2, 3])
    captured = inspect_model(Declared, "observed", values=observed)
    observed[0] = 99
    assert [constraint.right for constraint in captured.constraints if constraint.input == "values"] == [2, 3]


@pytest.mark.parametrize(
    "value",
    [Array([[1, 2]]), Array([[[1]]]), Array(1), Matrix([[1, 2], [3, 4]]), Matrix(0, 2, []), MatrixSymbol("M", 2, 1)],
)
def test_tensor_and_abstract_matrix_inputs_are_not_silently_flattened(value):
    with pytest.raises(ModelInputError):
        Vector(value)

    class Window(Modello):
        values = Vector("x", size=2)

    with pytest.raises(ModelInputError):
        Window("bad-shape", values=value)


@pytest.fixture
def scalar_model():
    class Scalar(Modello):
        a = InstanceDummy("a", real=True)
        value = InstanceDummy("value", real=True)

    return Scalar


@pytest.mark.parametrize("parameter,expected", [(0, 0), (1, 1)])
def test_native_limit_observations_wait_for_parameter_values(scalar_model, solve_model, parameter, expected):
    t = Coordinate("t", real=True)
    a = scalar_model.a
    source = Limit(a * t / (a * t + 1), t, oo)
    case = solve_model(scalar_model, a=parameter, value=source)
    assert case.value == expected
    recorded = inspect_model(scalar_model, "source", value=source)
    assert next(constraint.right for constraint in recorded.constraints if constraint.input == "value").has(Limit)


@pytest.mark.parametrize("kind", ["input", "definition", "alias", "component", "vector", "series", "model-law"])
def test_native_cancellation_cannot_hide_a_source_denominator(kind):
    class Scalar(Modello):
        a = InstanceDummy("a", real=True)
        value = InstanceDummy("value", real=True)

    if kind == "input":
        model, observations = (
            Scalar,
            {"a": 0, "value": Mul(Scalar.a, Pow(Scalar.a, -1, evaluate=False), evaluate=False)},
        )
    elif kind == "definition":

        class Defined(Scalar):
            value = Mul(Scalar.a, Pow(Scalar.a, -1, evaluate=False), evaluate=False)

        model, observations = Defined, {"a": 0}
    elif kind == "alias":

        class Aliased(Modello):
            a = InstanceDummy("a", real=True)
            b = InstanceDummy("b", real=True)
            value = a / b
            a = b

        model, observations = Aliased, {"b": 0}
    elif kind == "component":

        class Inner(Scalar):
            t = Coordinate("t", real=True)
            value = Limit(Mul(Scalar.a, Pow(Scalar.a, -1, evaluate=False), t, evaluate=False), t, 1)

        class Outer(Modello):
            inner = component(Inner)
            a = inner.a

        model, observations = Outer, {"a": 0}
    else:
        if kind == "vector":

            class VectorAlias(Modello):
                a, b = InstanceDummy("a", real=True), InstanceDummy("b", real=True)
                values = Vector(a / b)
                a = b

            model = VectorAlias
        elif kind == "series":

            class SeriesAlias(Modello):
                a, b = InstanceDummy("a", real=True), InstanceDummy("b", real=True)
                n = Index("n")
                T = Series("T", n, real=True)
                law = Recurrence(T[n], T[n - 1] + a / b, initial={0: 1})
                a = b
                value = law.at(1)

            model = SeriesAlias
        else:

            class State(Modello):
                value = InstanceDummy("value", real=True)

            class ModelAlias(Modello):
                a, b = InstanceDummy("a", real=True), InstanceDummy("b", real=True)
                states = model_vector(State, size=2)
                n = Index("n")
                law = model_recurrence(states[n], value=states[n - 1].value + a / b)
                a = b

            model = ModelAlias
        observations = {"b": 0}
    assert inspect_model(model, "undefined", **observations).solutions().is_empty is True
    with pytest.raises(SolutionCountError):
        model("undefined", **observations)


def test_class_series_references_in_native_inputs_bind_to_the_current_problem(solve_model):
    class Path(Modello):
        opening = InstanceDummy("opening", real=True)
        result = InstanceDummy("result", real=True)
        n = Index("n")
        T = Series("T", n, real=True)
        law = Recurrence(T[n], T[n - 1] + 1, initial={0: opening})

    case = solve_model(Path, opening=2, result=Path.T[3])
    assert case.result == 5


def test_native_functions_resolve_and_differentiate_inside_models(solve_model):
    class Cube(Function):
        @classmethod
        def eval(cls, value):
            if value.is_number:
                return value**3

        def fdiff(self, argindex=1):
            return 3 * self.args[0] ** 2

    class Response(Modello):
        x = InstanceDummy("x", real=True)
        value = Cube(x)
        slope = Derivative(value, x)

    case = solve_model(Response, x=2)
    assert (case.value, case.slope) == (8, 12)


def test_native_solutions_can_return_as_checked_model_states():
    class Square(Modello):
        x = InstanceDummy("x", real=True)
        squared = x**2

    problem = inspect_model(Square, "roots", squared=4)
    x = problem.symbols.x
    roots = solve(Eq(x**2, 4), x)
    assert roots == [-2, 2]
    family = problem.solutions().where(x > 0)
    assert family.where(Eq(x, roots[0])).is_empty is True
    chosen = family.where(Eq(x, roots[1])).one()
    assert isinstance(chosen, Square) and chosen.x == 2


def test_exported_finite_vectors_compile_and_reenter_the_model(solve_model):
    class Response(Modello):
        gain = InstanceDummy("gain", real=True)
        values = Vector(1, 2, 3) * gain
        total = values.sum()

    problem = inspect_model(Response, "function")
    family = problem.solutions()
    output = family.expression(problem.symbols.values)
    calculate = lambdify(problem.symbols.gain, output.elements, modules="math")
    native_result = Array(calculate(2))
    case = solve_model(Response, values=native_result)
    assert case.gain == 2 and case.total == 12


@pytest.mark.parametrize("value", [Lambda(Symbol("u"), Symbol("u") ** 2), MatrixSymbol("M", 2, 2), Eq(Symbol("x"), 1)])
def test_function_matrix_and_boolean_objects_are_not_scalar_quantities(scalar_model, value):
    with pytest.raises(ModelInputError, match="scalar"):
        scalar_model("invalid", value=value)


def test_named_lambda_helpers_get_an_explanatory_declaration_error():
    with pytest.raises(ModelDefinitionError, match="Lambda.*function.*inline"):

        class Invalid(Modello):
            u = Coordinate("u", real=True)
            operation = Lambda(u, u**2)


@pytest.mark.parametrize("boundary", ["vector", "scale", "formula", "series-index", "recurrence", "history", "query"])
def test_function_objects_are_not_accepted_as_scalar_results(boundary):
    n = Index("n")
    T = Series("T", n)
    operation = Lambda(n, n**2)
    law = Recurrence(T[n], T[n - 1] + 1, initial={0: 1})
    with pytest.raises(ModelInputError):
        if boundary == "vector":
            Vector(operation)
        elif boundary == "scale":
            Vector(1, 2) * operation
        elif boundary == "formula":
            vector(operation, (n, 0, 1))
        elif boundary == "series-index":
            T[operation]
        elif boundary == "recurrence":
            Recurrence(T[n], operation, initial={0: 1})
        elif boundary == "history":
            Recurrence(T[n], T[n - 1] + 1, initial=Tuple(Tuple(0, operation)))
        else:
            law.at(operation)


def test_sympy_scalar_conversion_protocol_is_used_without_a_numeric_round_trip(scalar_model, solve_model):
    class ExactValue:
        def _sympy_(self):
            return Rational(1, 3)

    value = sympify(ExactValue())
    assert value == Rational(1, 3)
    assert solve_model(scalar_model, a=0, value=ExactValue()).value == value
    assert solve_model(scalar_model, a=0, value=value).value == Rational(1, 3)
    assert solve_model(scalar_model, a=0, value=S.One / 3).value == Rational(1, 3)
