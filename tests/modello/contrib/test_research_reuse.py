"""Reusing a case preserves identities through component changes."""

import pytest
from sympy import Abs, Add, Derivative, Eq, Integral, Limit, Mul, Product, S, Subs, Sum, exp

from modello import ConstraintKind, Coordinate, Index, InstanceDummy, Modello, Recurrence, Series, Vector, component
from modello.contrib.research import (
    Method,
    ResearchCase,
    ResearchError,
    exact_form,
    positive,
    predicate,
    requires,
    smooth,
)
from modello.contrib.stochastic import ItoProcess
from modello.errors import ModelDefinitionError, ModelInputError, ModelValidationError


@pytest.fixture
def curves():
    class Old(Modello):
        x = Coordinate("x", real=True)
        value = x**2

    class New(Modello):
        x = Coordinate("z", real=True)
        value = -(x**2)

    return Old, New


@pytest.mark.parametrize(
    "operation, expected",
    (
        (lambda value, x: Derivative(value, x), lambda x: -2 * x),
        (lambda value, x: Integral(value, (x, 0, 1)), lambda x: -S.One / 3),
        (lambda value, x: Sum(value, (x, 1, 2)), lambda x: -5),
        (lambda value, x: Product(value, (x, 1, 2)), lambda x: 4),
        (lambda value, x: Limit(value, x, 1), lambda x: -1),
        (lambda value, x: Subs(value, x, 2), lambda x: -4),
        (lambda value, x: Subs(value, x**2, 1), lambda x: -1),
    ),
    ids=("derivative", "integral", "sum", "product", "limit", "substitution", "compound_substitution"),
)
def test_replacement_rebinds_coordinates_used_only_inside_native_operators(curves, operation, expected):
    old, new = curves

    class Parent(Modello):
        curve = component(old)
        value = operation(curve.value, old.x)

    source = ResearchCase.for_model(Parent)
    changed = source.replace("curve", new)
    expression = changed.expression("value")
    assert not expression.has(old.x)
    assert expression.doit() == expected(new.x)
    assert source.expression("value").has(old.x)


def test_replacement_rebinds_native_observations_and_requirement_payloads(curves):
    old, new = curves

    class Parent(Modello):
        curve = component(old)
        observed = InstanceDummy("observed", real=True)

    source = ResearchCase.for_model(Parent)
    slope = Derivative(source.symbols.curve.value, old.x)
    claim = exact_form(slope, 0, name="stationary")
    changed = source.observe(observed=slope).require(claim).replace("curve", new)
    assert changed.expression("observed").doit() == -2 * new.x
    assert changed.requirements[0].subject == Derivative(changed.symbols.curve.value, new.x)
    evidence = changed.check().explain("stationary")
    assert evidence.expression.doit() == -2 * new.x
    assert evidence.verdict == "conditionally true"
    expected = exact_form("observed", slope, name="observed_slope")
    changed = source.observe(observed=slope).require(expected).replace("curve", new)
    assert changed.check().explain("observed_slope").verdict == "strictly true"


def test_replacement_rebinds_native_coordinates_in_vector_definitions(curves):
    old, new = curves

    class Parent(Modello):
        curve = component(old)
        values = Vector((Derivative(curve.value, old.x),))

    changed = ResearchCase.for_model(Parent).replace("curve", new)
    assert changed.expression("values.0").doit() == -2 * new.x


def test_replacement_rebinds_substitutions_inside_equal_parent_expressions(curves):
    old, new = curves

    class Parent(Modello):
        curve = component(old)

    native = Subs(Derivative(S.Zero, old.x, evaluate=False), old.x, 1)
    subject = Add(native, S.One, evaluate=False)
    source = ResearchCase.for_model(Parent).require(exact_form(subject, S.One))
    changed = source.replace("curve", new)
    assert next(iter(changed.requirements[0].subject.atoms(Subs))).variables == (new.x,)
    assert next(iter(source.requirements[0].subject.atoms(Subs))).variables == (old.x,)
    assert changed.check().explain("exact_form").verdict == "strictly true"


def test_replacement_rejects_stale_substitution_scopes_from_cached_function_constructors(curves):
    old, new = curves

    class Parent(Modello):
        curve = component(old)

    native = Subs(Derivative(S(17), old.x, evaluate=False), old.x, 0)
    source = ResearchCase.for_model(Parent).require(exact_form(exp(native), S.One))
    exp(native, evaluate=False)  # Prime the held constructor used while rebinding.
    assert next(iter(source.requirements[0].subject.atoms(Subs))).variables == (old.x,)
    with pytest.raises(ModelInputError, match="operation scopes"):
        source.replace("curve", new)
    assert next(iter(source.requirements[0].subject.atoms(Subs))).variables == (old.x,)


@pytest.mark.parametrize("declaration", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_replacement_discovers_each_coordinate_of_equal_native_substitutions(declaration, reverse):
    class A(Modello):
        x = Coordinate("x", real=True)
        value = x**2

    class B(Modello):
        x = Coordinate("y", real=True)
        value = x**2

    class Positive(Modello):
        x = Coordinate("p", positive=True)
        value = x**2

    first = Subs(Derivative(S.Zero, A.x, evaluate=False), A.x, 0)
    second = Subs(Derivative(S.Zero, B.x, evaluate=False), B.x, 0)
    assert first == second
    # Retain the requested argument order despite native mathematical equality.
    held = Add._from_args((second, first) if reverse else (first, second))

    class Parent(Modello):
        a = component(A)
        b = component(B)
        value = held if declaration else S.Zero

    claim = exact_form("value" if declaration else held, S.Zero)
    source = ResearchCase.for_model(Parent).require(claim)
    assert source.check().explain(claim.name).verdict == "strictly true"
    try:
        changed = source.replace("a", Positive)
    except ModelInputError as error:
        assert "operation scopes" in str(error)
    else:
        assert changed.check().explain(claim.name).verdict == "unresolved"
        expression = changed.expression("value") if declaration else changed.requirements[0].subject
        assert not expression.has(A.x)
        assert expression.has(Positive.x)
        with pytest.raises(ResearchError):
            changed.run(Method("changed", lambda current: 0, changed.requirements, uses=("value",)))
    assert source.check().explain(claim.name).verdict == "strictly true"


def test_replacement_preserves_derivative_premises_in_a_held_zero_product(curves):
    old, _ = curves

    class Positive(Modello):
        x = Coordinate("p", positive=True)
        value = x**2

    class Parent(Modello):
        curve = component(old)

    native = Subs(Derivative(S.Zero, old.x, evaluate=False), old.x, 0)
    source = ResearchCase.for_model(Parent).require(exact_form(Mul(S.Zero, native, evaluate=False), S.Zero))
    changed = source.replace("curve", Positive)
    assert changed.check().explain("exact_form").verdict == "unresolved"
    with pytest.raises(ResearchError):
        changed.run(Method("zero", lambda current: 0, changed.requirements))
    assert source.check().explain("exact_form").verdict == "strictly true"


@pytest.mark.parametrize("where", ("declaration", "assumption", "requirement", "expected", "hint"))
def test_replacement_rejects_ambiguous_shared_native_coordinates(curves, where):
    old, new = curves

    class Parent(Modello):
        a = component(old)
        b = component(old)
        if where == "declaration":
            slope = Derivative(a.value, old.x)

    case = ResearchCase.for_model(Parent)
    shared = Derivative(case.symbols.b.value, old.x)
    if where == "assumption":
        case = case.assume(Eq(shared, 0))
    elif where == "requirement":
        case = case.require(exact_form(case.symbols.a.value + shared, 0, name="joined"))
    elif where == "expected":
        case = case.require(exact_form("a.value", shared, name="joined"))
    elif where == "hint":
        case = case.with_hint("bound", upper=case.symbols.a.value + shared)
    with pytest.raises(ModelDefinitionError, match="shared|joins"):
        case.replace("a", new)


@pytest.mark.parametrize("where", ("requirement", "hint"))
def test_replacement_rejects_ambiguous_expanded_native_payloads(curves, where):
    old, new = curves

    class Parent(Modello):
        a = component(old)
        b = component(old)

    source = ResearchCase.for_model(Parent)
    expanded = Derivative(source.expression("b.value"), old.x)
    if where == "requirement":
        source = source.require(exact_form(expanded, 0))
    else:
        source = source.with_hint("bound", upper=expanded)
    with pytest.raises(ModelDefinitionError, match="joins"):
        source.replace("a", new)


def test_replacement_preserves_a_named_unchanged_siblings_hint_scope(curves):
    old, new = curves

    class Parent(Modello):
        a = component(old)
        b = component(old)

    source = ResearchCase.for_model(Parent)
    hint = Derivative(source.symbols.b.value, old.x)
    changed = source.with_hint("bound", upper=hint).replace("a", new)
    assert dict(changed.hints)["bound"] == Derivative(changed.symbols.b.value, old.x)


@pytest.mark.parametrize("observed", (False, True))
def test_replacement_rejects_splitting_aliased_fields(observed):
    class Old(Modello):
        a = InstanceDummy("a", real=True)
        b = a

    class New(Modello):
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)

    class Parent(Modello):
        curve = component(Old)

    source = ResearchCase.for_model(Parent)
    if observed:
        source = source.observe(curve=Old("old", a=2))
    with pytest.raises(ModelDefinitionError, match="splits aliased fields"):
        source.replace("curve", New)
    assert source.problem.model is Parent


def test_replacement_preserves_aliases_and_their_observations():
    class Old(Modello):
        a = InstanceDummy("a", real=True)
        b = a

    class New(Modello):
        a = InstanceDummy("a", real=True)
        b = a

    class Parent(Modello):
        curve = component(Old)

    source = ResearchCase.for_model(Parent).observe(curve=Old("old", a=2))
    changed = source.replace("curve", New)
    assert changed.symbols.curve.a == changed.symbols.curve.b
    assert changed.expression("curve.a") == changed.expression("curve.b") == 2


@pytest.mark.parametrize("query", ("series", "recurrence"))
def test_replacement_rebinds_named_declarations_in_case_payloads(query):
    class Old(Modello):
        n = Index("n")
        opening = InstanceDummy("opening", real=True)
        values = Series("values", n, real=True)
        law = Recurrence(values[n], 2 * values[n - 1], initial={0: opening})
        terminal = law.at(2)

    class New(Modello):
        n = Index("position")
        opening = InstanceDummy("opening", real=True)
        values = Series("values", n, real=True)
        law = Recurrence(values[n], 3 * values[n - 1], initial={0: opening})
        terminal = law.at(2)

    class Parent(Modello):
        curve = component(Old)
        sibling = component(Old)
        observed = InstanceDummy("observed", real=True)

    original = ResearchCase.for_model(Parent)
    value = original.symbols.curve.values[2] if query == "series" else original.symbols.curve.law.at(2)
    source = (
        original.observe(observed=value).require(exact_form("curve.terminal", value)).with_hint("upper", upper=value)
    )
    changed = source.replace("curve", New)
    opening = changed.symbols.curve.opening
    assert changed.expression("observed").doit() == changed.expression("curve.terminal").doit() == 9 * opening
    assert changed.expression("sibling.terminal").doit() == 4 * changed.symbols.sibling.opening
    assert source.expression("observed").doit() == 4 * source.symbols.curve.opening
    expected = changed.requirements[0].expected
    assert changed.expression(expected).doit() == 9 * opening
    assert changed.expression(dict(changed.hints)["upper"]).doit() == 9 * opening
    result = changed.run(Method("query", lambda current: current.expression("observed").doit(), uses=("observed",)))
    assert result.data == 9 * opening


def test_replacement_rejects_removed_or_incompatible_declarations():
    class Old(Modello):
        n = Index("n")
        values = Series("values", n)

    class Removed(Modello):
        pass

    class Continuous(Modello):
        n = Coordinate("n", real=True)
        values = Series("values", n)

    class Parent(Modello):
        curve = component(Old)

    source = ResearchCase.for_model(Parent)
    with pytest.raises(ModelDefinitionError, match="recorded declaration"):
        source.replace("curve", Removed)
    with pytest.raises(ModelDefinitionError, match="argument signature"):
        source.replace("curve", Continuous)


def test_named_declaration_binders_do_not_join_shared_parent_coordinates():
    class Old(Modello):
        n = Index("n")
        values = Series("values", n)
        law = Recurrence(values[n], 2 * values[n - 1], initial={0: 1})

    class New(Modello):
        n = Index("position")
        values = Series("values", n)
        law = Recurrence(values[n], 3 * values[n - 1], initial={0: 1})

    class Parent(Modello):
        a = component(Old)
        b = component(Old)
        named = a.values[2]

    source = ResearchCase.for_model(Parent)
    changed = source.replace("a", New)
    assert changed.expression("named").doit() == 9
    assert source.expression("named").doit() == 4
    assert changed.expression(changed.symbols.b.values[2]).doit() == 4


def test_replacement_rejects_splitting_aliased_declarations():
    class Old(Modello):
        n = Index("n")
        first = Series("first", n)
        second = first

    class New(Modello):
        n = Index("n")
        first = Series("first", n)
        second = Series("second", n)

    class Parent(Modello):
        curve = component(Old)

    with pytest.raises(ModelDefinitionError, match="splits aliased"):
        ResearchCase.for_model(Parent).replace("curve", New)


@pytest.mark.parametrize("original_kind", ("field", "declaration"))
def test_replacement_cannot_change_a_recorded_paths_kind(original_kind):
    class Scalar(Modello):
        value = InstanceDummy("value", real=True)

    class Declared(Modello):
        n = Index("n")
        value = Series("value", n)

    old, new = (Scalar, Declared) if original_kind == "field" else (Declared, Scalar)

    class Parent(Modello):
        curve = component(old)

    source = ResearchCase.for_model(Parent)
    if original_kind == "field":
        source = source.observe(curve=Scalar("old", value=2))
    with pytest.raises(ModelDefinitionError, match=f"recorded {original_kind}"):
        source.replace("curve", new)


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("derived", (False, True))
def test_replacement_cannot_turn_a_coordinate_path_into_a_model_field(nested, derived):
    class OldCurve(Modello):
        x = Coordinate("x", real=True)
        value = x**2

    class NewCurve(Modello):
        parameter = InstanceDummy("parameter", real=True)
        x = 2 * parameter if derived else InstanceDummy("x", real=True)
        value = x**2

    class OldWrapper(Modello):
        inner = component(OldCurve)

    class NewWrapper(Modello):
        inner = component(NewCurve)

    old, new = (OldWrapper, NewWrapper) if nested else (OldCurve, NewCurve)

    class Parent(Modello):
        curve = component(old)
        slope = Derivative(curve.inner.value if nested else curve.value, OldCurve.x)

    source = ResearchCase.for_model(Parent)
    with pytest.raises(ModelDefinitionError, match="coordinate paths to model fields") as failure:
        source.replace("curve", new)
    assert ("inner.x" if nested else "x") in str(failure.value)
    assert source.expression("slope").doit() == 2 * OldCurve.x
    assert source.problem.model is Parent


def test_replacement_checks_coordinate_alias_paths_even_when_the_primary_path_survives():
    class Old(Modello):
        x = Coordinate("x", real=True)
        argument = x
        value = x**2

    class New(Modello):
        x = Coordinate("x", real=True)
        argument = InstanceDummy("argument", real=True)
        value = x**2

    class Parent(Modello):
        curve = component(Old)

    with pytest.raises(ModelDefinitionError, match="coordinate paths to model fields: argument"):
        ResearchCase.for_model(Parent).replace("curve", New)


@pytest.fixture
def processes():
    class Old(Modello):
        x = Coordinate("x", real=True)
        t = Coordinate("t", real=True)
        process = ItoProcess(x, 5, 3, t)

    class New(Modello):
        x = Coordinate("z", real=True)
        t = Coordinate("s", real=True)
        process = ItoProcess(x, 7, 4, t)

    return Old, New


@pytest.mark.parametrize("operation, expected", (("mean", 30), ("variance", 64), ("exact", 34)))
def test_process_solution_queries_rebind_without_joining_sibling_local_coordinates(processes, operation, expected):
    old, new = processes

    class Parent(Modello):
        a = component(old)
        b = component(old)
        named = a.process.mean(2, 4)
        observed = InstanceDummy("observed", real=True)

    source = ResearchCase.for_model(Parent)
    query = (
        getattr(source.symbols.a.process, operation)(2, 4, 1)
        if operation == "exact"
        else getattr(source.symbols.a.process, operation)(2, 4)
    )
    changed = source.observe(observed=query).replace("a", new)
    assert changed.expression("named").doit() == 30
    assert changed.expression("observed").doit() == expected
    assert changed.expression(changed.symbols.b.process.mean(2, 4)).doit() == 22


@pytest.mark.parametrize("operation", ("drift", "initial"))
def test_process_queries_keep_external_coordinate_join_checks(processes, operation):
    old, new = processes

    class Parent(Modello):
        a = component(old)
        b = component(old)
        observed = InstanceDummy("observed", real=True)

    source = ResearchCase.for_model(Parent)
    process = source.symbols.a.process
    query = process.drift(old.x**2) if operation == "drift" else process.mean(old.x, 1)
    with pytest.raises(ModelDefinitionError, match="joins"):
        source.observe(observed=query).replace("a", new)


@pytest.fixture
def fixed_series():
    class Old(Modello):
        n = Index("n")
        values = Series("values", n)
        law = Recurrence(values[n], 2 * values[n - 1], initial={0: 1})

    class New(Modello):
        n = Index("position")
        values = Series("values", n)
        law = Recurrence(values[n], 3 * values[n - 1], initial={0: 1})

    return Old, New


@pytest.mark.parametrize("composed", (False, True))
@pytest.mark.parametrize("explicit", (False, True))
def test_named_law_changes_invalidate_inferred_values_and_preserve_inputs(fixed_series, composed, explicit):
    old, new = fixed_series

    class Parent(Modello):
        curve = component(old)
        sibling = component(old)
        n = Index("outer")
        outer = Series("outer", n)
        composed_law = Recurrence(outer[n], outer[n - 1] + curve.values[1], initial={0: 0})
        value = composed_law.at(2) if composed else curve.values[2]
        alias = value

    class Wrapper(Modello):
        model = component(Parent)

    point = Parent("old", value=4) if explicit else Parent("old")
    source = ResearchCase.for_model(Wrapper).observe(model=point)
    changed = source.replace("model.curve", new)
    if explicit:
        assert any(item.kind is ConstraintKind.INPUT and item.right == 4 for item in changed.problem.constraints)
        assert changed.check().consistent is False
    else:
        expected = 6 if composed else 9
        assert changed.check().consistent is True
        assert changed.expression("model.value").doit() == changed.expression("model.alias").doit() == expected
        assert any("Invalidated inferred observation" in item for item in changed.changes)
        assert changed.expression(changed.symbols.model.sibling.values[2]).doit() == 4
        assert source.expression("model.value").doit() == 4


@pytest.mark.parametrize("native", (False, True))
def test_native_series_queries_retain_the_same_observation_sources_as_fields(native):
    class Curve(Modello):
        x = Coordinate("x", real=True)
        n = Index("n")
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        values = Series("values", n)
        law = Recurrence(values[n], a * values[n - 1], initial={0: 1})
        first = law.at(1)

    source = ResearchCase.for_model(Curve)
    case = source.observe(
        a=source.symbols.b, b=2 * source.symbols.a + Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, 0)
    )
    subject = case.symbols.values[1] if native else "first"
    claim = exact_form(subject, S.Zero, wrt="x")
    evidence = case.check(requirements=(claim,)).explain(claim.name)
    assert evidence.verdict == "strictly false"
    assert evidence.source_operands


def test_a_method_consumes_component_contracts_reached_through_named_laws(fixed_series):
    old, _ = fixed_series
    restricted = requires(old, predicate(S.false, name="admissible"))

    class Parent(Modello):
        curve = component(restricted)
        value = curve.values[2]

    case = ResearchCase.for_model(Parent)
    calls = []
    with pytest.raises(ResearchError) as failure:
        case.run(Method("read", lambda current: calls.append(current), uses=("value",)))
    assert failure.value.report.explain("curve/admissible").verdict == "strictly false"
    assert not calls


@pytest.mark.parametrize("alpha_renamed", (False, True))
def test_explicit_named_law_selection_does_not_consume_an_alternative_laws_contract(alpha_renamed):
    class Curve(Modello):
        n = Index("n")
        values = Series("values", n)
        chosen = Recurrence(values[n], 2 * values[n - 1], initial={0: 1})
        other = chosen.as_dummy() if alpha_renamed else chosen
        alternative = requires(
            Recurrence(other.target, 3 * other.target.series[other.index - 1], initial={0: 1}),
            predicate(S.false, name="not_selected"),
        )
        value = chosen.at(2)

    case = ResearchCase.for_model(Curve)
    result = case.run(Method("read", lambda current: current.expression("value").doit(), uses=("value",)))
    assert result.data == 4
    assert result.report.explain("alternative/not_selected").verdict == "strictly false"


@pytest.mark.parametrize("observed", (False, True))
def test_ambiguous_series_queries_remain_inspectable_and_block_required_methods(observed):
    class Curve(Modello):
        n = Index("n")
        values = Series("values", n)
        first = Recurrence(values[n], 2 * values[n - 1], initial={0: 1})
        second = Recurrence(values[n], 3 * values[n - 1], initial={0: 1})
        input = InstanceDummy("input")

    case = ResearchCase.for_model(Curve)
    subject = case.symbols.values[2]
    if observed:
        case = case.observe(input=subject)
        subject = "input"
    claim = exact_form(subject, 4)
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == "unresolved"
    calls = []
    with pytest.raises(ResearchError) as failure:
        case.run(Method("read", lambda current: calls.append(current), (claim,), uses=("input",) if observed else ()))
    assert failure.value.report.explain(claim.name).verdict == "unresolved"
    assert not calls


def test_alpha_renamed_declaration_aliases_consume_the_same_contracts():
    class Curve(Modello):
        n = Index("n")
        values = Series("values", n)
        chosen = Recurrence(values[n], 2 * values[n - 1], initial={0: 1})
        alias = requires(chosen.as_dummy(), predicate(S.false, name="required"))
        value = chosen.at(2)

    case = ResearchCase.for_model(Curve)
    calls = []
    with pytest.raises(ResearchError) as failure:
        case.run(Method("read", lambda current: calls.append(current), uses=("value",)))
    assert failure.value.report.explain("alias/required").verdict == "strictly false"
    assert not calls


@pytest.mark.parametrize("split", (False, True))
def test_replacement_preserves_alpha_renamed_declaration_aliases(fixed_series, split):
    old, _ = fixed_series

    class Old(old):
        alias = old.law.as_dummy()

    class New(Modello):
        n = Index("position")
        values = Series("values", n)
        law = Recurrence(values[n], 3 * values[n - 1], initial={0: 1})
        alpha = law.as_dummy()
        alias = Recurrence(alpha.target, 4 * alpha.target.series[alpha.index - 1], initial={0: 1}) if split else alpha

    class Parent(Modello):
        curve = component(Old)

    source = ResearchCase.for_model(Parent)
    if split:
        with pytest.raises(ModelDefinitionError, match="splits aliased"):
            source.replace("curve", New)
    else:
        changed = source.replace("curve", New)
        assert changed.expression(changed.symbols.curve.law.at(2)).doit() == 9
        assert changed.expression(changed.symbols.curve.alias.at(2)).doit() == 9


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("shadow", (None, lambda self: None))
def test_shadowed_coordinate_names_are_not_available_through_research_paths(shadow, nested):
    class Original(Modello):
        x = Coordinate("x", real=True)
        value = x**2

    class Hidden(Original):
        x = shadow
        value = S.One

    class Parent(Modello):
        curve = component(Hidden)

    case = ResearchCase.for_model(Parent if nested else Hidden)
    prefix = "curve." if nested else ""
    assert case.expression(prefix + "value") == 1
    assert case.check(requirements=(positive(prefix + "value"),)).checks[0].verdict == "strictly true"
    with pytest.raises(ModelDefinitionError, match="Unknown scalar or coordinate path"):
        case.expression(prefix + "x")
    with pytest.raises(ModelDefinitionError, match="Unknown scalar or coordinate path"):
        case.check(requirements=(smooth(prefix + "value", wrt=prefix + "x"),))
    with pytest.raises(ModelValidationError):
        case.expression(prefix + "value", wrt=Original.x)


def test_coordinate_lookup_preserves_inheritance_and_rebinding_to_an_independent_field():
    class Original(Modello):
        x = Coordinate("x", real=True)
        value = requires(x**2, smooth("value", wrt=x))

    class Inherited(Original):
        pass

    inherited = ResearchCase.for_model(Inherited)
    assert inherited.expression("x") == Original.x
    assert inherited.check().checks[0].verdict == "strictly true"

    class Observable(Original):
        x = InstanceDummy("x", real=True)
        value = x**2

    case = ResearchCase.for_model(Observable)
    evidence = case.check().checks[0]
    assert case.expression("x") == case.symbols.x
    assert evidence.coordinate == case.symbols.x
    assert evidence.verdict == "strictly true"


def test_hiding_a_coordinate_name_preserves_explicit_symbols_used_by_the_formula():
    class Original(Modello):
        x = Coordinate("x", real=True)
        value = x**2

    class Hidden(Original):
        x = None

    case = ResearchCase.for_model(Hidden)
    with pytest.raises(ModelDefinitionError):
        case.expression("x")
    assert case.expression("value", wrt=Original.x) == Original.x**2
    claim = smooth("value", wrt=Original.x)
    assert case.check(requirements=(claim,)).checks[0].verdict == "strictly true"


@pytest.mark.parametrize("hidden_first", (False, True))
def test_coordinate_lookup_respects_multiple_inheritance_precedence(hidden_first):
    class Hidden(Modello):
        x = None

    class Visible(Modello):
        x = Coordinate("x", real=True)

    class HiddenFirst(Hidden, Visible):
        pass

    class VisibleFirst(Visible, Hidden):
        pass

    case = ResearchCase.for_model(HiddenFirst if hidden_first else VisibleFirst)
    if hidden_first:
        with pytest.raises(ModelDefinitionError):
            case.expression("x")
    else:
        assert case.expression("x") == Visible.x


@pytest.mark.parametrize("hidden_first", (False, True))
def test_ordinary_mixin_attributes_take_part_in_coordinate_shadowing(hidden_first):
    class HiddenMixin:
        x = None

    class Visible(Modello):
        x = Coordinate("x", real=True)

    class HiddenFirst(HiddenMixin, Visible):
        pass

    class VisibleFirst(Visible, HiddenMixin):
        pass

    case = ResearchCase.for_model(HiddenFirst if hidden_first else VisibleFirst)
    if hidden_first:
        with pytest.raises(ModelDefinitionError):
            case.expression("x")
    else:
        assert case.expression("x") == Visible.x


def test_native_coordinate_annotations_rebind_coordinates_inherited_from_ordinary_mixins():
    class CoordinateMixin:
        x = Coordinate("x", real=True)

    class Original(CoordinateMixin, Modello):
        value = requires(CoordinateMixin.x**2, smooth("value", wrt=CoordinateMixin.x))

    class Renamed(Original):
        x = Coordinate("new_x", real=True)
        value = x**2

    inherited = ResearchCase.for_model(Original)
    assert inherited.expression("x") == CoordinateMixin.x
    assert inherited.check().checks[0].verdict == "strictly true"
    changed = ResearchCase.for_model(Renamed)
    evidence = changed.check().checks[0]
    assert evidence.coordinate == Renamed.x
    assert evidence.verdict == "strictly true"
