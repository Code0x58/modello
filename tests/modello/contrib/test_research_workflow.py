"""Workflow issues exposed by applying research contracts to several models."""

from dataclasses import replace

import pytest
from sympy import Derivative, Eq, FiniteSet, Interval, Q, S, Subs, assuming, log

from modello import Coordinate, InstanceDummy
from modello.contrib.research import (
    Method,
    ResearchCase,
    ResearchError,
    ResearchModel,
    convex,
    exact_form,
    positive,
    requires,
)
from modello.errors import ModelDefinitionError, ModelInputError


def test_automatic_obligations_use_the_coordinates_declared_domain():
    class PositiveCurve(ResearchModel):
        x = Coordinate("x", positive=True)
        inverse = 1 / x
        slope = Derivative(log(x), x)

    case = ResearchCase.for_model(PositiveCurve)
    report = case.check()
    assert all(item.verdict == "strictly true" for item in report.checks)
    assert all(item.resolved.domain == Interval.open(0, S.Infinity) for item in report.checks)
    for path in ("inverse", "slope"):
        result = case.run(Method(path, lambda current, name=path: current.expression(name).doit(), uses=(path,)))
        assert result.data == 1 / PositiveCurve.x


def test_a_nonnegative_coordinate_still_requires_the_zero_denominator_check():
    class ClosedBoundary(ResearchModel):
        x = Coordinate("x", nonnegative=True)
        inverse = 1 / x

    case = ResearchCase.for_model(ClosedBoundary)
    assert case.check().checks[0].verdict == "strictly false"
    with pytest.raises(ResearchError):
        case.run(Method("inverse", lambda current: current.expression("inverse"), uses=("inverse",)))


def test_native_substitution_identities_evaluate_without_losing_their_domains():
    class Evaluated(ResearchModel):
        x = Coordinate("x", real=True)
        at = InstanceDummy("a", real=True)
        curve = x**2
        evaluated = Subs(curve, x, at)
        residual = evaluated - at**2
        bad = Subs(1 / x, x, 0)

    case = ResearchCase.for_model(Evaluated)
    report = case.require(
        exact_form(
            "residual", S.Zero, name="substitution_identity", consumer="test", reason="Evaluate a native substitution."
        ),
        exact_form("bad", S.Zero, name="undefined_point", consumer="test", reason="The pole cannot be erased."),
    ).check()
    assert report.explain("substitution_identity").verdict == "strictly true"
    assert report.explain("undefined_point").verdict == "strictly false"


def test_native_expected_derivatives_and_scalar_equalities_keep_their_premises():
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        at = InstanceDummy("a", real=True)
        curve = x**2
        slope = 2 * x
        scalar = InstanceDummy("b", real=True)

    case = ResearchCase.for_model(Curve)
    report = case.require(
        exact_form(
            "slope",
            Derivative(Curve.x**2, Curve.x),
            wrt="x",
            name="native_derivative",
            consumer="test",
            reason="Check both expressions.",
        ),
        exact_form(
            "scalar", S.One, name="parameter_equality", consumer="test", reason="Retain the parameter equality."
        ),
        exact_form(S.Zero, S.One, name="false_constant", consumer="test", reason="Refute unequal constants."),
    ).check()
    assert report.explain("native_derivative").verdict == "strictly true"
    assert report.explain("parameter_equality").verdict == "conditionally true"
    assert report.explain("false_constant").verdict == "strictly false"


def test_removing_an_assumption_rechecks_dependent_method_premises():
    class Parameter(ResearchModel):
        a = InstanceDummy("a", real=True)

    base = ResearchCase.for_model(Parameter)
    condition = base.symbols.a > 0
    positive_branch = base.assume(condition)
    requirement = positive(
        "a", name="positive_parameter", consumer="calculation", reason="This method needs a positive parameter."
    )
    method = Method("calculation", lambda current: current.expression("a"), (requirement,))
    assert positive_branch.run(method).report.explain("positive_parameter").verdict == "strictly true"
    relaxed = positive_branch.without_assumptions(condition, reason="Explore the zero and negative regimes.")
    assert positive_branch.assumptions == (condition,)
    assert relaxed.assumptions == ()
    with pytest.raises(ResearchError) as failure:
        relaxed.run(method)
    assert failure.value.report.explain("positive_parameter").verdict == "conditionally true"
    assert any("Explore the zero and negative regimes" in issue for issue in relaxed.check().issues)
    with pytest.raises(ModelInputError, match="not recorded"):
        relaxed.without_assumptions(condition, reason="Already absent.")
    with pytest.raises(ModelInputError, match="reason"):
        positive_branch.without_assumptions(condition, reason="")


def test_removing_a_case_requirement_cannot_waive_a_methods_own_requirement():
    class Parameter(ResearchModel):
        a = InstanceDummy("a", real=True)

    requirement = positive(
        "a", name="positive_parameter", consumer="calculation", reason="This operation needs positivity."
    )
    original = ResearchCase.for_model(Parameter).observe(a=-1).require(requirement)
    assert original.check().explain("positive_parameter").verdict == "strictly false"
    relaxed = original.without_requirements(
        "positive_parameter", reason="This exploratory report no longer requests positivity."
    )
    assert relaxed.check().checks == ()
    assert original.requirements == (requirement,)
    method = Method("calculation", lambda current: current.expression("a"), (requirement,))
    with pytest.raises(ResearchError) as failure:
        relaxed.run(method)
    assert failure.value.report.explain("positive_parameter").verdict == "strictly false"
    assert any("Removed requirement positive_parameter" in issue for issue in relaxed.check().issues)


def test_declaration_and_automatic_requirements_are_changed_at_their_source():
    class Annotated(ResearchModel):
        a = requires(InstanceDummy("a", real=True), positive("a", consumer="test", reason="Live declaration."))
        inverse = 1 / a

    case = ResearchCase.for_model(Annotated)
    for key in ("a/positive", "inverse/auto:1"):
        with pytest.raises(ModelInputError, match="Only case-added"):
            case.without_requirements(key, reason="Cannot erase a consumed contract.")


def test_identical_requirements_are_reused_and_conflicting_ids_remain_errors():
    class Parameter(ResearchModel):
        a = InstanceDummy("a", real=True)

    requirement = positive(
        "a", name="positive_parameter", consumer="calculation", reason="The same consumed requirement."
    )
    case = ResearchCase.for_model(Parameter).observe(a=2).require(requirement)
    result = case.run(Method("calculation", lambda current: current.expression("a"), (requirement,)))
    assert len(result.report.checks) == 1
    conflicting = positive("a", name="positive_parameter", consumer="another consumer", reason="A conflicting ID.")
    with pytest.raises(ModelDefinitionError, match="positive_parameter"):
        case.check(requirements=(conflicting,))


def test_case_bound_methods_reject_assumption_only_branches_before_execution():
    class Parameter(ResearchModel):
        a = InstanceDummy("a", real=True)

    case = ResearchCase.for_model(Parameter).observe(a=2)
    calls = []
    method = Method("captured formula", lambda current: calls.append(current) or current.expression("a")).bind(case)
    assert case.run(method).data == 2
    changed = case.assume(case.symbols.a > 0)
    assert changed.problem is case.problem
    with pytest.raises(ResearchError, match="different research case") as failure:
        changed.run(method)
    assert failure.value.case is changed
    assert calls == [case]
    with pytest.raises(ModelDefinitionError, match="Rebuild"):
        method.bind(changed)
    assert method.bind(case) is method
    rebuilt = Method("captured formula", lambda current: current.expression("a")).bind(changed)
    assert changed.run(rebuilt).data == 2


def test_method_factories_use_the_current_case_and_forward_execution_settings():
    class Parameter(ResearchModel):
        a = InstanceDummy("a", real=True)

    calls = []

    def factory(current):
        value = current.expression("a")
        requirement = positive("a", name="positive", consumer="calculation", reason="A positive parameter is needed.")

        def execute(inner, *, scale=1):
            calls.append(inner)
            return value * scale

        return Method("calculation", execute, (requirement,)).bind(current)

    base = ResearchCase.for_model(Parameter)
    first, second = base.observe(a=2), base.observe(a=3)
    assert first.run(factory, scale=4).data == 8
    assert second.run(factory).data == 3
    assert calls == [first, second]
    negative = base.observe(a=-1)
    with pytest.raises(ResearchError) as failure:
        negative.run(factory)
    assert failure.value.case is negative
    assert failure.value.report.explain("positive").verdict == "strictly false"
    assert calls == [first, second]


def test_a_factory_cannot_bypass_a_prepared_methods_context_guard():
    class Parameter(ResearchModel):
        a = InstanceDummy("a", real=True)

    case = ResearchCase.for_model(Parameter).observe(a=2)
    method = Method("captured", lambda current: 2).bind(case)
    changed = case.assume(case.symbols.a > 0)
    with pytest.raises(ResearchError, match="different research case") as failure:
        changed.run(lambda current: method)
    assert failure.value.case is changed


def test_a_factory_must_return_a_method_instead_of_a_computation_result():
    case = ResearchCase.for_model(ResearchModel)
    with pytest.raises(ModelDefinitionError, match="factory must return a Method"):
        case.run(lambda current: 1)  # type: ignore[arg-type, return-value]


def test_audits_preserve_the_requirement_and_formula_before_observations():
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = a * x

    requirement = exact_form(
        "value",
        lambda get: get("a") * get("x"),
        wrt="x",
        name="live_formula",
        consumer="calculation",
        reason="Retain the source identity as well as its checked specialization.",
    )
    report = ResearchCase.for_model(Curve).observe(a=2).require(requirement).check()
    audit = report.to_dict()["checks"][0]
    assert audit["declared_subject"] == "value"
    assert "a" in audit["expected_template"]
    assert "2" in audit["expected"]
    assert "Declared subject:" in report.to_html()
    assert "Expected template:" in report.to_html()


def test_an_inner_derivative_cannot_borrow_an_unrelated_outer_domain():
    class DifferentCoordinates(ResearchModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", positive=True)
        value = 1 / x

    case = ResearchCase.for_model(DifferentCoordinates).assume(DifferentCoordinates.x < 0)
    claim = exact_form(
        "value",
        Derivative(log(DifferentCoordinates.x), DifferentCoordinates.x, evaluate=False),
        wrt="y",
        domain=Interval.open(0, S.Infinity),
        name="cross_scope",
    )
    requirement = replace(claim, consumer="consumer", reason="The real logarithm's argument is a separate coordinate.")
    with pytest.raises(ResearchError) as failure:
        case.run(Method("consumer", lambda current: 42, (requirement,)))
    assert failure.value.report.explain("cross_scope").verdict == "unresolved"


def test_intrinsic_nonzero_coordinates_keep_zero_out_of_their_requested_domain():
    class Punctured(ResearchModel):
        x = Coordinate("x", real=True, nonzero=True)
        inverse = 1 / x
        square = x**2

    case = ResearchCase.for_model(Punctured)
    automatic = case.check().checks[0]
    assert automatic.verdict == "strictly true"
    assert automatic.resolved.domain == S.Reals - FiniteSet(0)
    impossible_domain = positive(
        "square", wrt="x", name="overbroad", consumer="test", reason="An intrinsic exclusion cannot be erased."
    )
    assert case.require(impossible_domain).check().explain("overbroad").verdict == "unresolved"


def test_reused_scalar_work_keeps_fresh_formula_payloads_and_each_reports_budget():
    class Parameter(ResearchModel):
        a = InstanceDummy("a", real=True)

    target = [2]
    requirement = exact_form(
        "a",
        lambda get: S(target[0]),
        name="live_target",
        consumer="test",
        reason="Resolve the actual formula before reuse.",
    )
    case = ResearchCase.for_model(Parameter).observe(a=2).require(requirement)
    first = case.check()
    assert first.explain("live_target").verdict == "strictly true"
    target[0] = 3
    second = case.check()
    assert second.explain("live_target").verdict == "strictly false"
    assert first.to_dict()["checks"][0]["expected"] == "2"
    assert second.to_dict()["checks"][0]["expected"] == "3"
    target[0] = 2
    extra = exact_form(
        "a", S(2), name="same_identity", consumer="another consumer", reason="The mathematical result can be shared."
    )
    populated = case.require(extra)
    assert all(item.verdict == "strictly true" for item in populated.check().checks)
    limited = populated.check(max_steps=1)
    assert limited.explain("live_target").verdict == "strictly true"
    assert limited.explain("same_identity").verdict == "unresolved"


@pytest.mark.parametrize("observed", (False, True))
def test_cached_resolution_follows_temporary_sympy_assumptions(observed):
    class Ratio(ResearchModel):
        a = InstanceDummy("a", real=True)
        numerator = a + 1
        denominator = a + 1
        ratio = numerator / denominator

    case = ResearchCase.for_model(Ratio)
    if observed:
        case = case.observe(a=case.symbols.a)
    requirement = exact_form("ratio", S.One, name="one", consumer="consumer", reason="Retain the source pole.")
    predicate = Q.nonzero(case.symbols.a + 1)
    with assuming(predicate):
        first = case.check(requirements=(requirement,))
        assert first.explain("one").verdict == "strictly true"
        assert predicate in first.assumptions
    unguarded = case.check(requirements=(requirement,))
    assert unguarded.explain("one").verdict == "conditionally true"
    assert unguarded.explain("one").expression.subs(case.symbols.a, -1) is S.NaN
    assert predicate not in unguarded.assumptions
    with assuming(predicate):
        assert case.check(requirements=(requirement,)).explain("one").verdict == "strictly true"


def test_global_assumptions_are_checked_against_observations():
    class Parameter(ResearchModel):
        a = InstanceDummy("a", real=True)
        value = a + 2

    case = ResearchCase.for_model(Parameter).observe(a=-1)
    requirement = positive("value", name="positive", consumer="consumer", reason="Skip contradictory premises.")
    with assuming(Q.positive(case.symbols.a)):
        report = case.check(requirements=(requirement,))
        assert report.consistent is False
        assert report.explain("positive").verdict == "unresolved"
        with pytest.raises(ResearchError):
            case.run(Method("consumer", lambda current: 1, (requirement,)))
    assert case.check(requirements=(requirement,)).explain("positive").verdict == "strictly true"
    symbolic = ResearchCase.for_model(Parameter)
    symbolic = symbolic.assume(symbolic.symbols.a < 0)
    with assuming(Q.positive(symbolic.symbols.a)):
        assert symbolic.check().consistent is False


def test_case_bound_methods_require_the_same_global_assumption_context():
    class Parameter(ResearchModel):
        a = InstanceDummy("a", real=True)

    case = ResearchCase.for_model(Parameter)
    calls = []
    with assuming(Q.positive(case.symbols.a)):
        method = Method("captured formula", lambda current: calls.append(current) or 1).bind(case)
        assert case.run(method).data == 1
    with pytest.raises(ResearchError, match="SymPy assumption context") as failure:
        case.run(method)
    assert failure.value.case is case
    assert calls == [case]
    with pytest.raises(ModelDefinitionError, match="Rebuild"):
        method.bind(case)
    with assuming(Q.positive(case.symbols.a)):
        assert method.bind(case) is method
        assert case.run(method).data == 1


@pytest.fixture(params=(0, 1))
def equal_native_assumptions(request):
    class Coordinates(ResearchModel):
        x = Coordinate("x", real=True)
        p = Coordinate("p", positive=True)

    conditions = tuple(
        Eq(Subs(Derivative(S.Zero, coordinate, evaluate=False), coordinate, request.param), 0, evaluate=False)
        for coordinate in (Coordinates.x, Coordinates.p)
    )
    assert conditions[0] == conditions[1]
    assert conditions[0].lhs.variables != conditions[1].lhs.variables
    return ResearchCase.for_model(Coordinates), conditions, request.param


@pytest.mark.parametrize("removed", (0, 1))
def test_assumption_removal_preserves_distinct_native_scopes(equal_native_assumptions, removed):
    base, conditions, point = equal_native_assumptions
    case = base.assume(*conditions)
    selected, other = conditions[removed], conditions[1 - removed]
    relaxed = case.without_assumptions(selected, selected, reason="Remove just this native scope.")
    assert len(relaxed.assumptions) == 1
    assert relaxed.assumptions[0].lhs.variables == other.lhs.variables
    with pytest.raises(ModelInputError, match="not recorded"):
        relaxed.without_assumptions(selected, reason="This scope is already absent.")
    method = Method("read", lambda current: 1)
    if point == 0 and removed == 0:
        assert relaxed.check().consistent is None
        with pytest.raises(ResearchError):
            relaxed.run(method)
    else:
        assert relaxed.run(method).data == 1
    empty = case.without_assumptions(*conditions, *conditions, reason="Remove both recorded scopes.")
    assert not empty.assumptions
    assert len(empty.changes) == 2
    assert empty.run(method).data == 1
    assert len(case.assumptions) == 2
    assert case.check().consistent is (None if point == 0 else True)


def test_bound_methods_distinguish_equal_native_assumption_scopes(equal_native_assumptions):
    case, conditions, _ = equal_native_assumptions
    calls = []
    with assuming(conditions[0]):
        method = Method("captured", lambda current: calls.append(current) or 1).bind(case)
        assert case.run(method).data == 1
    with assuming(conditions[1]):
        with pytest.raises(ModelDefinitionError, match="Rebuild"):
            method.bind(case)
        with pytest.raises(ResearchError, match="SymPy assumption context"):
            case.run(method)
    assert calls == [case]
    with assuming(conditions[0]):
        assert method.bind(case) is method
        assert case.run(method).data == 1


def test_singleton_convexity_uses_the_value_instead_of_ambient_curvature():
    class Curves(ResearchModel):
        x = Coordinate("x", real=True)
        concave = -(x**2)
        pole = 1 / x

    case = ResearchCase.for_model(Curves)
    for path, verdict in (("concave", "strictly true"), ("pole", "strictly false")):
        requirement = convex(
            path,
            wrt="x",
            domain=Interval(0, 0),
            name="singleton",
            consumer="consumer",
            reason="Convexity on a singleton requires a finite real value, without a curvature condition.",
        )
        assert case.check(requirements=(requirement,)).explain("singleton").verdict == verdict
    requirement = convex(
        "concave",
        wrt="x",
        domain=FiniteSet(-1, 1),
        name="disconnected",
        consumer="consumer",
        reason="The interval curvature test cannot refute unsupported disconnected-domain convexity.",
    )
    assert case.check(requirements=(requirement,)).explain("disconnected").verdict == "unresolved"
