"""Composition preserves domains, lexical scope, and declared dynamic dependencies."""

import pytest
from sympy import Eq, Piecewise, Rational, S, Sum, Symbol, simplify

from modello import (
    Coordinate,
    Index,
    InstanceDummy,
    ItoProcess,
    Modello,
    Recurrence,
    Series,
    Vector,
    inspect_model,
    model_recurrence,
    model_vector,
)
from modello.errors import ModelInputError


class State(Modello):
    value = InstanceDummy("value", real=True)


@pytest.fixture(params=["cyclic", "constant-last", "dynamic-last"])
def implicit_coefficients(request):
    def declare(*, query):
        class Implicit(Modello):
            x, y = InstanceDummy("x", real=True), InstanceDummy("y", real=True)
            c, d = InstanceDummy("c", real=True), InstanceDummy("d", real=True)
            enabled = InstanceDummy("enabled", integer=True)
            if request.param == "cyclic":
                c = d + y
                d = c / 2
            else:
                first = c
                first = 2 * y if request.param == "constant-last" else S(2)
                second = c
                second = S(2) if request.param == "constant-last" else 2 * y
            t = Coordinate("t", real=True)
            noise = ItoProcess(y, 0, 1, t)
            process = ItoProcess(x, Piecewise((c, Eq(enabled, 1)), (2, True)), 0, t)
            if query:
                value = process.variance(0, 1)

        return Implicit

    return declare


@pytest.mark.parametrize("same_family", [False, True])
@pytest.mark.parametrize("offset", [0, -1])
def test_an_explicit_recurrence_composes_in_one_doit_even_for_the_same_family(same_family, offset):
    n = Index("n")
    A, B = Series("A", n, real=True), Series("B", n, real=True)
    base = Recurrence(A[n], A[n - 1] + 1, initial={0: 0})
    target = A if same_family else B
    derived = Recurrence(target[n], base.at(n + offset) + 1, initial={0: 0})
    assert derived.at(3).doit() == 4 + offset


def test_a_causal_law_can_reuse_a_named_boundary_without_turning_it_into_a_relative_lag():
    n = Index("n")
    T = Series("T", n, real=True)
    law = Recurrence(T[n], T[n - 1] + T[0], initial={0: 2})
    assert law.at(3).doit() == 8
    k = Symbol("k", integer=True, nonnegative=True)
    assert simplify(law.at(k).doit() - 2 * (k + 1)) == 0


@pytest.mark.parametrize("point", [0, 1, 3])
def test_zero_history_does_not_erase_a_transition_denominator(point):
    n = Index("n")
    a = Symbol("a", real=True)
    T = Series("T", n, real=True)
    law = Recurrence(T[n], T[n - 1] / a, initial={0: 0})
    query = law.at(point).doit()
    assert query.subs(a, 2) == 0
    assert query.subs(a, 0) == (0 if point == 0 else S.NaN)


def test_symbolic_recurrence_queries_keep_denominator_domains_only_when_a_step_is_used():
    n, k = Index("n"), Index("k", nonnegative=True)
    a = Symbol("a", real=True)
    T = Series("T", n, real=True)
    law = Recurrence(T[n], T[n - 1] / a, initial={0: 0})
    query = law.at(k).doit()
    assert query.subs({a: 0, k: 0}) == 0
    assert query.subs({a: 0, k: 2}) is S.NaN


def test_families_preserve_a_denominator_before_observations_cancel_the_history():
    class Divided(Modello):
        a = InstanceDummy("a", real=True)
        n = Index("n")
        T = Series("T", n, real=True)
        law = Recurrence(T[n], T[n - 1] / a, initial={0: 0})
        value = law.at(2)

    family = inspect_model(Divided, "domain").solutions()
    assert family.parameter_domain.contains(0) is S.false
    assert family.observe(a=0).is_empty is True
    assert family.observe(a=2).one().value == 0


@pytest.mark.parametrize("count", [1, 2, 5])
def test_centered_boundary_value_equations_use_the_existing_joint_solver(solve_model, count):
    class Bridge(Modello):
        N = InstanceDummy("N", integer=True, positive=True)
        left = InstanceDummy("left", real=True)
        right = InstanceDummy("right", real=True)
        states = model_vector(State, size=N + 1)
        n = Index("n")
        law = model_recurrence(
            states[n],
            start=0,
            value=Piecewise(
                (left, Eq(n, 0)), (right, Eq(n, N)), ((states[n - 1].value + states[n + 1].value) / 2, True)
            ),
        )

    case = solve_model(Bridge, N=count, left=2, right=12)
    assert tuple(state.value for state in case.states) == tuple(2 + Rational(10 * i, count) for i in range(count + 1))


@pytest.mark.parametrize("prefix", [False, True], ids=["constant-sum", "free-upper-bound"])
def test_native_sum_binders_can_share_the_outer_coordinate_without_capture(solve_model, prefix):
    class Prefix(Modello):
        inputs = Vector("u", size=3, real=True)
        states = model_vector(State, size=4)
        n = Index("n")
        law = model_recurrence(
            states[n],
            start=0,
            value=Sum(inputs[n], (n, 0, n - 1)) if prefix else Sum(inputs[n], (n, 0, 2)),
        )

    case = solve_model(Prefix, inputs=[2, 3, 5])
    assert tuple(state.value for state in case.states) == ((0, 2, 5, 10) if prefix else (10, 10, 10, 10))


@pytest.mark.parametrize("start", [0, 1])
def test_relative_indices_inside_an_embedded_law_keep_their_finite_window(start):
    class Embedded(Modello):
        inputs = Vector("u", size=3, real=True)
        states = model_vector(State, size=3)
        n, k = Index("n"), Index("k")
        T = Series("T", k, real=True)
        law = model_recurrence(
            states[n],
            start=start,
            value=Recurrence(T[k], T[k - 1] + inputs[n - 1], initial={0: 0}).at(1),
        )

    family = inspect_model(
        Embedded, "embedded", inputs=[2, 3, 5], states=[State("opening", value=0), None, None]
    ).solutions()
    if start == 0:
        assert family.is_empty is True
    else:
        assert tuple(state.value for state in family.one().states[1:]) == (2, 3)


@pytest.mark.parametrize("point", [1, 2])
def test_evaluated_derivatives_keep_the_body_domain_at_the_substitution_point(point):
    from sympy import Derivative, Subs

    class Slope(Modello):
        x = Coordinate("x", real=True)
        at = InstanceDummy("at", real=True)
        value = Subs(Derivative((x**2 - 1) / (x - 1), x, evaluate=False), x, at)

    family = inspect_model(Slope, "slope", at=point).solutions()
    if point == 1:
        assert family.is_empty is True
    else:
        assert family.one().value == 1


@pytest.mark.parametrize("increment", [-2, -1, 0])
def test_exact_realizations_respect_the_state_domain_at_the_queried_endpoint(increment):
    x, t = Coordinate("x", positive=True), Coordinate("t", real=True)
    law = ItoProcess(x, 0, 1, t)
    assert law.exact(1, 1, increment).doit() == (S.NaN if increment < 0 else 1)


@pytest.mark.parametrize("as_field", [False, True], ids=["coordinate", "observed-field"])
@pytest.mark.parametrize("operation", ["drift", "diffusion", "generator"])
def test_one_state_transforms_reject_another_registered_state_even_after_point_observations(as_field, operation):
    class Coupled(Modello):
        x = InstanceDummy("x", real=True) if as_field else Coordinate("x", real=True)
        y = InstanceDummy("y", real=True) if as_field else Coordinate("y", real=True)
        t = Coordinate("t", real=True)
        W = Series("W", t, real=True)
        first = ItoProcess(x, 0, 1, t, W)
        second = ItoProcess(y, 0, 1, t, W)
        value = getattr(first, operation)(x * y)

    values = {"x": 2, "y": 3} if as_field else {}
    with pytest.raises(ModelInputError, match="another registered"):
        inspect_model(Coupled, "coupled", **values).solutions()


@pytest.mark.parametrize("operation", ["mean", "variance", "exact"])
@pytest.mark.parametrize("elapsed", [0, 1])
def test_exact_queries_do_not_freeze_a_dynamic_coefficient_but_retain_zero_time_identities(operation, elapsed):
    class Integrated(Modello):
        x = InstanceDummy("x", real=True)
        y = InstanceDummy("y", real=True)
        t = Coordinate("t", real=True)
        W = Series("W", t, real=True)
        noise = ItoProcess(y, 0, 1, t, W)
        process = ItoProcess(x, y, 0, t, W)
        value = (
            getattr(process, operation)(0, elapsed, 0)
            if operation == "exact"
            else getattr(process, operation)(0, elapsed)
        )

    problem = inspect_model(Integrated, "integrated", x=0, y=2)
    if elapsed:
        with pytest.raises(ModelInputError, match="constant coefficient"):
            problem.solutions()
    else:
        assert problem.solutions().one().value == 0


def test_implicit_coefficient_dependencies_survive_observations_and_returned_laws(solve_model, implicit_coefficients):
    observations = {"x": 0, "y": 1, "c": 2, "d": 1, "enabled": 1}
    with pytest.raises(ModelInputError, match="explicit parametrization"):
        solve_model(implicit_coefficients(query=True), **observations)

    case = solve_model(implicit_coefficients(query=False), **observations)
    assert case.process.drift(case.process.state).doit() == 2
    assert case.process.variance(0, 0).doit() == 0
    for law in (case.process, case.process.as_dummy()):
        with pytest.raises(NotImplementedError, match="explicit parametrization"):
            law.variance(0, 1).doit()

    # An inactive coefficient dependency requires no joint-process solver.
    observations["enabled"] = 0
    assert solve_model(implicit_coefficients(query=True), **observations).value == 0


def test_a_local_transform_can_use_adapted_coefficients_without_asking_for_joint_moments(solve_model):
    class Coupled(Modello):
        x = InstanceDummy("x", real=True)
        y = InstanceDummy("y", real=True)
        t = Coordinate("t", real=True)
        first = ItoProcess(y, 0, 1, t)
        second = ItoProcess(x, y, 1, t)
        value = second.drift(x**2)

    assert solve_model(Coupled, x=2, y=3).value == 13


@pytest.mark.parametrize("operation", ["drift", "diffusion", "generator"])
def test_local_ito_calculus_keeps_the_domain_of_an_explicit_rational_transform(operation):
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    law = ItoProcess(x, 0, 1, t)
    value = getattr(law, operation)((x**2 - 1) / (x - 1)).doit()
    assert value.subs(x, 1) is S.NaN
    assert value.subs(x, 2) == (1 if operation == "diffusion" else 0)


@pytest.mark.parametrize("operation", ["exact", "mean", "variance"])
def test_exact_ito_queries_keep_static_coefficient_denominators_even_when_initial_is_zero(operation):
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    a = Symbol("a", real=True)
    law = ItoProcess(x, x / a, 0, t)
    query = getattr(law, operation)(0, 1, 0) if operation == "exact" else getattr(law, operation)(0, 1)
    value = query.doit()
    assert value.subs(a, 0) is S.NaN
    assert value.subs(a, 2) == 0


def test_exact_ito_recognition_does_not_cancel_a_state_dependent_hole():
    x, t = Coordinate("x", real=True), Coordinate("t", real=True)
    law = ItoProcess(x, (x**2 - 1) / (x - 1), 0, t)
    with pytest.raises(NotImplementedError, match="throughout"):
        law.exact(0, 1, 0).doit()


@pytest.mark.parametrize("state_value", [0, -1])
def test_a_constant_transform_result_cannot_hide_an_invalid_state_observation(state_value):
    x, t = Coordinate("x", positive=True), Coordinate("t", real=True)
    law = ItoProcess(x, 0, x, t)
    assert law.drift(x / x).subs(x, state_value).doit() is S.NaN


@pytest.mark.parametrize("operator", ["integral", "sum", "derivative", "limit"])
def test_a_series_state_under_a_time_binder_is_not_silently_frozen(operator):
    from sympy import Derivative, Integral, Limit

    t = Coordinate("t", real=True)
    X = Series("X", t, real=True)
    law = ItoProcess(X[t], 0, 1, t)
    body = {
        "integral": lambda: Integral(X[t], (t, 0, 1)),
        "sum": lambda: Sum(X[t], (t, 0, 2)),
        "derivative": lambda: Derivative(X[t], t, evaluate=False),
        "limit": lambda: Limit(X[t], t, 0),
    }[operator]()
    with pytest.raises(ModelInputError, match="path or joint-process"):
        law.drift(body)
    with pytest.raises(ModelInputError, match="path or joint-process"):
        ItoProcess(X[t], body, 1, t)


def test_independently_scoped_model_components_do_not_hide_their_other_process_states():
    from modello import component

    class Cell(Modello):
        state = InstanceDummy("x", real=True)
        t = Coordinate("t", real=True)
        law = ItoProcess(state, 0, 1, t)

    class Pair(Modello):
        left = component(Cell)
        right = component(Cell)
        value = left.law.diffusion(left.state * right.state)

    with pytest.raises(ModelInputError, match="another registered"):
        inspect_model(Pair, "scoped", left=Cell("left", state=2), right=Cell("right", state=3)).solutions()


@pytest.mark.parametrize("product", [False, True])
def test_an_empty_reduction_does_not_evaluate_an_unused_unsupported_body(solve_model, product):
    from sympy import Product

    class Empty(Modello):
        x, t = Coordinate("x", real=True), Coordinate("t", real=True)
        i = Index("i")
        process = ItoProcess(x, -(x**3), 1, t)
        value = Product(process.mean(1, 1), (i, 0, -1)) if product else Sum(process.mean(1, 1), (i, 0, -1))

    assert solve_model(Empty).value == (1 if product else 0)


@pytest.mark.parametrize("weights", [[0], [0, 0], [1, -1]])
def test_finite_operator_materialization_cannot_erase_individual_term_poles(weights):
    n, j = Index("n"), Index("j")
    a = Symbol("a", real=True)
    T = Series("T", n, real=True)
    data = Vector(weights)
    law = Recurrence(T[n], Sum(data[j] * T[n - 1] / a, (j, 0, len(weights) - 1)), initial={0: 0})
    value = law.at(2).doit()
    assert value.subs(a, 2) == 0
    assert value.subs(a, 0) is S.NaN


def test_observing_zero_in_an_ordinary_model_does_not_cancel_a_denominator():
    class Ratio(Modello):
        numerator = InstanceDummy("x", real=True)
        denominator = InstanceDummy("a", real=True)
        ratio = numerator / denominator
        scaled = numerator * ratio

    family = inspect_model(Ratio, "ratio", numerator=0).solutions()
    assert family.parameter_domain.contains(0) is S.false
    assert family.observe(denominator=2).one().scaled == 0
    assert family.observe(denominator=0).is_empty is True


@pytest.mark.parametrize("pole", [Rational(1, 2), -1])
def test_parameterized_improper_integrals_are_evaluated_with_the_observed_pole_location(pole):
    from sympy import Integral, log

    class IntegralModel(Modello):
        pole = InstanceDummy("a", real=True)
        x = Coordinate("x", real=True)
        value = Integral(1 / (x - pole), (x, 0, 1))

    family = inspect_model(IntegralModel, "integral", pole=pole).solutions()
    if pole > 0:
        assert family.is_empty is True
    else:
        assert family.one().value == log(2)


def test_an_implicit_process_declaration_is_checked_as_part_of_the_same_model_context():
    class Integrated(Modello):
        x, y = InstanceDummy("x", real=True), InstanceDummy("y", real=True)
        t = Coordinate("t", real=True)
        noise_variance = ItoProcess(y, 0, 1, t).variance(0, 1)
        value = ItoProcess(x, y, 0, t).variance(0, 1)

    with pytest.raises(ModelInputError, match="constant coefficients"):
        inspect_model(Integrated, "implicit", x=0, y=0).solutions()


def test_returned_laws_do_not_freeze_other_processes_after_model_point_substitution(solve_model):
    class Coupled(Modello):
        x, y = InstanceDummy("x", real=True), InstanceDummy("y", real=True)
        t = Coordinate("t", real=True)
        noise = ItoProcess(y, 0, 1, t)
        integrated = ItoProcess(x, y, 0, t)
        local_drift = integrated.drift(x**2)

    case = solve_model(Coupled, x=2, y=3)
    assert case.local_drift == 12
    for law in (case.integrated, case.integrated.as_dummy()):
        with pytest.raises(NotImplementedError, match="constant coefficients"):
            law.variance(0, 1).doit()
        assert law.variance(0, 0).doit() == 0


def test_a_deterministic_calculus_result_can_be_reused_as_another_process_coefficient(solve_model):
    class Composed(Modello):
        x, y = InstanceDummy("x", real=True), InstanceDummy("y", real=True)
        t = Coordinate("t", real=True)
        source = ItoProcess(y, 0, 1, t)
        coefficient = source.generator(y**2)
        process = ItoProcess(x, coefficient, 0, t)
        mean = process.mean(0, 2)

    case = solve_model(Composed, x=0, y=3)
    assert case.coefficient == 1
    assert case.mean == 2
    assert case.process.mean(0, 2).doit() == 2


def test_explicit_discrete_recurrence_queries_are_ordinary_fixed_coefficients_for_ito_calculus():
    n, x, t = Index("n"), Coordinate("x", real=True), Coordinate("t", real=True)
    T = Series("T", n, real=True)
    law = Recurrence(T[n], T[n - 1] + 1, initial={0: 1})
    process = ItoProcess(x, 0, 1, t)
    assert process.drift(law.at(2) * x**2).doit() == 3
