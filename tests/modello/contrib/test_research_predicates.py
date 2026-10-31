"""Boolean truth cannot erase the domains of its scalar operands."""

from contextlib import nullcontext

import pytest
from sympy import (
    Abs,
    Add,
    And,
    Complement,
    Contains,
    Derivative,
    Eq,
    FiniteSet,
    Implies,
    Interval,
    Max,
    Min,
    Mul,
    Ne,
    Not,
    Or,
    Piecewise,
    Pow,
    Product,
    Q,
    S,
    Subs,
    arg,
    assuming,
    exp,
    log,
)

from modello import Coordinate, InstanceDummy, Modello
from modello.contrib.research import (
    Method,
    ResearchCase,
    ResearchError,
    ResearchModel,
    exact_form,
    positive,
    predicate,
    requires,
)
from modello.errors import ModelValidationError


class Parameters(Modello):
    a = InstanceDummy("a", real=True)
    b = InstanceDummy("b", real=True)


def assert_blocked(case, *, claim=None, uses=()):
    calls = []
    method = Method("read", lambda current: calls.append(current), () if claim is None else (claim,), uses=uses)
    with pytest.raises(ResearchError):
        case.run(method)
    assert not calls


@pytest.mark.parametrize("function", (log, arg))
@pytest.mark.parametrize(
    ("formula", "expected"),
    (
        (lambda value: Mul(0, value, evaluate=False), 0),
        (lambda value: Add(value, -value, evaluate=False), 0),
        (lambda value: Pow(value, 0, evaluate=False), 1),
    ),
)
def test_scalar_declarations_retain_partial_sources_before_simplification(function, formula, expected):
    class Curve(Modello):
        x = Coordinate("x", real=True)
        value = formula(function(x))

    case = ResearchCase.for_model(Curve)
    claim = exact_form("value", expected, wrt="x")
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == "strictly false"
    assert_blocked(case, claim=claim, uses=("value",))
    with pytest.raises(ModelValidationError):
        case.sample("value", wrt="x", points=(0,))
    assert tuple(map(float, case.sample("value", wrt="x", points=(1,)).values)) == (float(expected),)
    assert case.expression("value", wrt="x").subs(Curve.x, 0).doit() is S.NaN


def test_definition_reports_keep_declared_function_sources_before_inverse_simplification():
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        value = requires(exp(log(x), evaluate=False), exact_form("value", x, wrt=x))

    assert Curve.definition_report().explain("value/exact_form").verdict == "strictly false"
    case = ResearchCase.for_model(Curve)
    with pytest.raises(ModelValidationError):
        case.sample("value", wrt="x", points=(0,))
    assert case.sample("value", wrt="x", points=(-1, 1)).values == (-1.0, 1.0)


@pytest.mark.parametrize("ambient", (False, True))
@pytest.mark.parametrize("point", (0, 1))
def test_assumptions_validate_native_derivatives_before_certifying_other_requirements(ambient, point):
    class Curve(Modello):
        x = Coordinate("x", real=True)

    base = ResearchCase.for_model(Curve)
    premise = Eq(Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, point), int(bool(point)), evaluate=False)
    case = base if ambient else base.assume(premise)
    claim = positive(S.One)
    with assuming(premise) if ambient else nullcontext():
        report = case.check(requirements=(claim,))
        if point == 0:
            assert report.consistent is False
            assert report.explain(claim.name).verdict == "unresolved"
            assert any("undefined native operation" in issue for issue in report.issues)
            assert_blocked(case, claim=claim)
            assert_blocked(case)
        else:
            assert report.consistent is True
            assert report.explain(claim.name).verdict == "strictly true"
            assert case.run(Method("native", lambda current: 1, (claim,))).data == 1
    if ambient:
        assert base.check().consistent is True
        assert base.run(Method("left_context", lambda current: 1)).data == 1


def test_assumptions_cannot_certify_their_own_conditional_native_premises():
    class Curve(Modello):
        x = Coordinate("x", real=True)
        point = InstanceDummy("p", real=True)

    case = ResearchCase.for_model(Curve)
    p = case.symbols.point
    premise = Eq(Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, p), 1, evaluate=False)
    uncertain = case.assume(premise)
    assert uncertain.check().consistent is None
    assert uncertain.check().explain("model/auto:algebra").verdict == "unresolved"
    assert_blocked(uncertain)
    established = case.assume(And(p > 0, premise, evaluate=False))
    assert established.run(Method("positive_point", lambda current: 1)).data == 1


def test_assumptions_check_native_sources_erased_by_joint_observation_solving():
    class Curve(Parameters):
        x = Coordinate("x", real=True)

    base = ResearchCase.for_model(Curve)
    a, b = base.symbols.a, base.symbols.b
    undefined = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, 0)
    case = base.observe(a=b, b=2 * a + undefined).assume(Eq(a, 0, evaluate=False))
    report = case.check()
    assert report.consistent is False
    assert any("undefined native operation" in issue for issue in report.issues)
    assert_blocked(case)


@pytest.mark.parametrize("entry", ("requirement", "assumption", "ambient", "selector", "definition"))
def test_recursive_upstream_predicates_remain_inspectable_and_block_methods(entry):
    model = ResearchModel if entry == "definition" else Modello

    class Curve(model):
        i = Coordinate("i", integer=True)
        if entry in {"selector", "definition"}:
            value = Piecewise((1, Ne(Product(log(i), (i, 0, 1)), 1, evaluate=False)), (0, True), evaluate=False)

    base = ResearchCase.for_model(Curve)
    value = Product(log(Curve.i), (Curve.i, 0, 1))
    condition = Ne(value, 1, evaluate=False)
    case = base.assume(condition) if entry == "assumption" else base
    claim = predicate(condition)
    with assuming(condition) if entry == "ambient" else nullcontext():
        report = case.check(requirements=(claim,))
        assert report.explain(claim.name).verdict == "unresolved"
        assert_blocked(case, claim=claim)
        if entry != "requirement":
            assert report.consistent is None
            assert report.explain("model/auto:algebra").verdict == "unresolved"
            assert_blocked(case)
        if entry in {"selector", "definition"}:
            assert_blocked(case, uses=("value",))
            assert "unresolved" in report.to_json()
            with pytest.raises(NotImplementedError, match="source domains: RecursionError"):
                case.expression("value")
    if entry in {"selector", "definition"}:
        if entry == "definition":
            assert Curve.definition_report().consistent is None

        class Finite(model):
            i = Coordinate("i", integer=True)
            value = Piecewise((1, Eq(Product(i, (i, 1, 2)), 2, evaluate=False)), (0, True), evaluate=False)

        finite_case = ResearchCase.for_model(Finite)
        finite = exact_form("value", S.One)
        assert finite_case.run(Method("finite", lambda current: 1, (finite,), uses=("value",))).data == 1
        return
    finite = predicate(Eq(Product(Curve.i, (Curve.i, 1, 2)), 2, evaluate=False))
    assert base.check(requirements=(finite,)).explain(finite.name).verdict == "strictly true"


@pytest.mark.parametrize(
    "condition",
    (
        lambda a, b: Eq(1 / a, 1 / a, evaluate=False),
        lambda a, b: Ne(1 / a, 1 / b, evaluate=False),
        lambda a, b: Not(Eq(1 / a, 1 / b, evaluate=False), evaluate=False),
        lambda a, b: And(Ne(log(a), 1, evaluate=False), b > 0, evaluate=False),
        lambda a, b: Or(Ne(1 / a, 1 / b, evaluate=False), Eq(b, 0), evaluate=False),
        lambda a, b: Implies(Eq(b, 2), Ne(log(a), 1, evaluate=False), evaluate=False),
        lambda a, b: Q.finite(log(a)),
        lambda a, b: Ne(arg(a), 1, evaluate=False),
    ),
)
def test_predicates_check_scalar_domains_outside_boolean_truth(condition):
    case = ResearchCase.for_model(Parameters)
    claim = predicate(condition(case.symbols.a, case.symbols.b))
    invalid = case.observe(a=0, b=0)
    evidence = invalid.check(requirements=(claim,)).explain(claim.name)
    assert evidence.verdict in {"strictly false", "unresolved"}
    assert_blocked(invalid, claim=claim)
    valid = case.observe(a=1, b=2)
    assert valid.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"
    assert valid.run(Method("finite", lambda current: 1, (claim,))).data == 1


def test_negated_membership_cannot_certify_an_undefined_scalar():
    case = ResearchCase.for_model(Parameters)
    claim = predicate(Not(Contains(1 / case.symbols.a, S.Reals, evaluate=False), evaluate=False))
    invalid = case.observe(a=0)
    assert invalid.check(requirements=(claim,)).explain(claim.name).verdict != "strictly true"
    assert_blocked(invalid, claim=claim)
    assert case.observe(a=1).check(requirements=(claim,)).explain(claim.name).verdict == "strictly false"


def test_membership_allows_unbounded_intervals():
    case = ResearchCase.for_model(Parameters)
    claim = predicate(Contains(case.symbols.a, Interval.open(0, S.Infinity), evaluate=False))
    for value, expected in ((1, "strictly true"), (-1, "strictly false")):
        assert case.observe(a=value).check(requirements=(claim,)).explain(claim.name).verdict == expected


def test_negated_membership_retains_domains_of_scalar_set_elements():
    case = ResearchCase.for_model(Parameters)
    claim = predicate(Not(Contains(0, FiniteSet(1 / case.symbols.a), evaluate=False), evaluate=False))
    invalid = case.observe(a=0)
    assert invalid.check(requirements=(claim,)).explain(claim.name).verdict != "strictly true"
    assert_blocked(invalid, claim=claim)
    assert case.observe(a=1).check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"


@pytest.mark.parametrize("declaration_symbols", (False, True))
@pytest.mark.parametrize("through_field", (False, True))
def test_composite_set_predicates_retain_scalar_sources_before_model_expansion(declaration_symbols, through_field):
    class Curve(Modello):
        a = InstanceDummy("a", real=True)
        value = log(a)

    case = ResearchCase.for_model(Curve)
    symbols = Curve if declaration_symbols else case.symbols
    source = symbols.value if through_field else Mul(0, log(symbols.a), evaluate=False)
    domain = Complement(FiniteSet(0), FiniteSet(source), evaluate=False)
    claim = predicate(Not(Contains(0, domain, evaluate=False), evaluate=False))
    invalid = case.observe(a=0)
    assert invalid.check(requirements=(claim,)).explain(claim.name).verdict != "strictly true"
    assert_blocked(invalid, claim=claim)
    assert case.observe(a=1).check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"


def test_guarded_research_expressions_can_be_reused_in_solution_family_conditions():
    class Curve(Modello):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)

    case = ResearchCase.for_model(Curve)
    expression = case.expression(log(Derivative(Curve.x**2, Curve.x, evaluate=False)), wrt="x")
    condition = Eq(expression, 1, evaluate=False)
    assert case.family.where(condition).conditions == (condition,)


def test_predicate_identity_retains_a_sufficient_parameter_domain():
    case = ResearchCase.for_model(Parameters)
    claim = predicate(Eq(1 / case.symbols.a, 1 / case.symbols.a, evaluate=False))
    evidence = case.check(requirements=(claim,)).explain(claim.name)
    assert evidence.verdict == "conditionally true"
    assert evidence.condition == Ne(case.symbols.a, 0)
    assert_blocked(case, claim=claim)
    established = case.assume(case.symbols.a > 0)
    assert established.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"


def test_power_cancellation_retains_its_source_until_the_exponent_is_observed():
    case = ResearchCase.for_model(Parameters)
    power = Pow(Parameters.a, Parameters.b, evaluate=False)
    held = Add(power, -power, Parameters.a, Parameters.b, evaluate=False)
    claim = exact_form(held, Parameters.a + Parameters.b)
    invalid = case.observe(a=0, b=-1)
    assert invalid.check(requirements=(claim,)).explain(claim.name).verdict == "strictly false"
    assert_blocked(invalid, claim=claim)
    for valid in (case.observe(a=1, b=-1), case.observe(a=0, b=1)):
        assert valid.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"


@pytest.mark.parametrize("function", (log, arg))
@pytest.mark.parametrize("selector", (False, True))
@pytest.mark.parametrize("zero_coefficient", (False, True))
def test_declaration_predicates_keep_function_sources_until_binding(function, selector, zero_coefficient):
    def source(x):
        return Mul(0, function(x), evaluate=False) if zero_coefficient else function(x)

    a = InstanceDummy("a", real=True)
    condition = Eq(source(a), 0 if zero_coefficient else source(a), evaluate=False)
    value = Piecewise((1, condition), (-1, True), evaluate=False)

    class Choice(Modello):
        parameter = a
        chosen = value if selector else parameter

    case = ResearchCase.for_model(Choice)
    claim = (
        positive("chosen")
        if selector
        else predicate(
            Eq(source(Choice.parameter), 0 if zero_coefficient else source(Choice.parameter), evaluate=False)
        )
    )
    invalid = case.observe(parameter=0)
    assert invalid.check(requirements=(claim,)).explain(claim.name).verdict != "strictly true"
    assert_blocked(invalid, claim=claim, uses=("chosen",))
    valid = case.observe(parameter=1)
    assert valid.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"


@pytest.mark.parametrize("function", (log, arg))
@pytest.mark.parametrize(
    "wrapper",
    (lambda value: value, lambda value: Min(0, value, evaluate=False), lambda value: Max(0, value, evaluate=False)),
)
def test_rebinding_expected_templates_retains_scalar_source_without_adding_proof_guards(function, wrapper):
    case = ResearchCase.for_model(Parameters)
    template = wrapper(Mul(0, function(Parameters.a), evaluate=False))
    claim = exact_form(S.Zero, template)
    invalid = case.observe(a=0)
    assert invalid.check(requirements=(claim,)).explain(claim.name).verdict != "strictly true"
    assert_blocked(invalid, claim=claim)
    valid = case.observe(a=1).check(requirements=(claim,)).explain(claim.name)
    assert valid.verdict == "strictly true"
    assert valid.expected_template == wrapper(Mul(0, function(case.symbols.a), evaluate=False))


@pytest.mark.parametrize("ambient", (False, True))
@pytest.mark.parametrize(
    "condition",
    (
        lambda a, b: Ne(1 / a, 1 / b, evaluate=False),
        lambda a, b: Ne(log(a), 1, evaluate=False),
        lambda a, b: Or(Ne(log(a), 1, evaluate=False), Eq(b, 0), evaluate=False),
        lambda a, b: Implies(Eq(b, 2), Ne(arg(a), 1, evaluate=False), evaluate=False),
    ),
)
def test_assumptions_require_their_scalar_operands_to_be_defined(condition, ambient):
    base = ResearchCase.for_model(Parameters)
    premise = condition(base.symbols.a, base.symbols.b)
    case = (base if ambient else base.assume(premise)).observe(a=0, b=0)
    with assuming(premise) if ambient else nullcontext():
        report = case.check()
        assert report.consistent is not True
        assert_blocked(case)
    if ambient:
        assert case.check().consistent is True
        assert case.run(Method("unrestricted", lambda current: 1)).data == 1


@pytest.mark.parametrize("function", (log, arg))
@pytest.mark.parametrize("retained", (False, True))
def test_model_selectors_retain_scalar_domains_through_declaration_and_rebinding(function, retained):
    a = InstanceDummy("a", real=True)
    selector = Ne(function(a), 1, evaluate=False)
    value = Piecewise((1, selector), (-1, True), evaluate=False) if retained else None

    class Choice(Modello):
        parameter = a
        chosen = value if retained else Piecewise((1, Ne(function(parameter), 1, evaluate=False)), (-1, True))

    case = ResearchCase.for_model(Choice)
    claim = positive("chosen")
    invalid = case.observe(parameter=0)
    assert invalid.check(requirements=(claim,)).explain(claim.name).verdict != "strictly true"
    assert_blocked(invalid, uses=("chosen",))
    valid = case.observe(parameter=1)
    assert valid.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"
    assert valid.run(Method("chosen", lambda current: current.expression("chosen"), uses=("chosen",))).data == 1


def test_selectors_only_require_domains_on_reachable_branches():
    class Choice(Modello):
        switch = InstanceDummy("s", real=True)
        a = InstanceDummy("a", real=True)
        chosen = Piecewise((2, switch > 0), (1, Ne(log(a), 1, evaluate=False)), (-1, True))

    case = ResearchCase.for_model(Choice).observe(a=0)
    method = Method("chosen", lambda current: current.expression("chosen"), uses=("chosen",))
    assert case.observe(switch=1).run(method).data == 2
    assumed = case.assume(case.symbols.switch > 0).run(method).data
    assert assumed.subs(case.symbols.switch, 1) == 2
    assert assumed.subs(case.symbols.switch, -1) is S.NaN
    assert_blocked(case, uses=("chosen",))
    assert_blocked(case.observe(switch=-1), uses=("chosen",))


@pytest.mark.parametrize("positive_coordinate", (False, True))
def test_selector_domains_keep_curve_and_local_substitution_scopes(positive_coordinate):
    class Curve(Modello):
        x = Coordinate("x", **({"positive": True} if positive_coordinate else {"real": True}))
        curve = Piecewise((1, Ne(log(x), 1, evaluate=False)), (-1, True))
        local = Subs(curve, x, 1)

    case = ResearchCase.for_model(Curve)
    local = Method("local", lambda current: current.expression("local").doit(), uses=("local",))
    assert case.run(local).data == 1
    curve = Method("curve", lambda current: 1, uses=("curve",))
    if positive_coordinate:
        assert case.run(curve).data == 1
    else:
        assert_blocked(case, uses=("curve",))
        assert_blocked(case.assume(Curve.x > 0), uses=("curve",))
        with pytest.raises(ModelValidationError):
            case.sample("curve", wrt="x", points=(0,))
        assert case.expression("curve", wrt="x").subs(Curve.x, 0).doit() is S.NaN
    assert case.sample("curve", wrt="x", points=(1,)).values == (1.0,)


@pytest.mark.parametrize("through_observation", (False, True))
def test_predicates_preserve_source_domains_through_aliases_and_observations(through_observation):
    class Aliases(Modello):
        a = InstanceDummy("a", real=True)
        source = InstanceDummy("s", real=True) if through_observation else log(a)
        alias = source

    case = ResearchCase.for_model(Aliases)
    if through_observation:
        case = case.observe(source=log(case.symbols.a))
    claim = predicate(Ne(case.symbols.alias, 1, evaluate=False))
    invalid = case.observe(a=0)
    assert invalid.check(requirements=(claim,)).explain(claim.name).verdict != "strictly true"
    assert_blocked(invalid, claim=claim)
    assert case.observe(a=1).check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"
