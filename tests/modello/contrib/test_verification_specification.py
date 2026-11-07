# mypy: check-untyped-defs
"""Named proof declarations keep their Python schema across context updates."""

from dataclasses import dataclass

import pytest
from sympy import Interval, S

from modello import InstanceDummy, Modello, component
from modello.contrib.verification import (
    InferenceRule,
    Obligation,
    VerificationContext,
    equal_to,
    positive,
    supremum_below,
)
from modello.errors import ModelDefinitionError, ModelInputError, ModelValidationError


class Curve(Modello):
    x = InstanceDummy("x", real=True)
    value = x**2


class Study(Modello):
    curve = component(Curve)


@dataclass(frozen=True)
class Claims:
    increasing_input: Obligation
    conclusion: Obligation
    implication: InferenceRule
    identities: tuple[Obligation, ...] = ()


def claims(symbols: Study) -> Claims:
    premise = positive(symbols.curve.x)
    conclusion = Obligation("original_conclusion", symbols.curve.value, "rule")
    return Claims(
        premise,
        conclusion,
        InferenceRule("original_rule", conclusion, (premise,), "A positive input has a positive square."),
        (equal_to(S.One, S.One), equal_to(S.One, S.One)),
    )


def test_catalog_names_and_rule_references_come_from_the_declared_schema():
    context = VerificationContext.for_model(Study).with_specification(claims)
    declared = context.specification
    assert declared.increasing_input.name == "increasing_input"
    assert declared.identities[1].name == "identities[1]"
    assert declared.implication.conclusion is declared.conclusion
    assert declared.implication.premises == (declared.increasing_input,)
    assert context.obligations == (declared.increasing_input, declared.conclusion, *declared.identities)
    assert context.rules == (declared.implication,)
    assert context.explain(declared.increasing_input).verdict == "conditionally true"
    assert context.explain(declared.identities[1]).id == "identities[1]"
    observed = context.observe({context.symbols.curve.x: 2})
    assert observed.explain(observed.specification.implication).verdict == "strictly true"
    assert observed.verify().explain(observed.specification.conclusion).kind == "rule"
    assert observed.verify().explain(observed.specification.identities[0]).id == "identities[0]"
    # Observation edits preserve scalar identities and the named declarations.
    assert observed.explain(declared.increasing_input).verdict == "strictly true"
    assert len(observed.obligations) == len(context.obligations)
    assert claims(context.symbols).increasing_input.name == "positive"


def test_catalog_replacement_rebuilds_names_and_preserves_only_added_goals_and_rules():
    @dataclass(frozen=True)
    class ValueClaim:
        value: Obligation

    class Decreasing(Curve):
        value = -Curve.x

    base = VerificationContext.for_model(Study)
    context = base.with_specification(lambda symbols: ValueClaim(positive(symbols.curve.value)))
    context = context.observe({context.symbols.curve.x: 2})
    context = context.with_obligations(positive(context.symbols.curve.x, name="extra"))
    changed = context.replace(context.symbols.curve, Decreasing)
    assert changed.explain(changed.specification.value).verdict == "strictly false"
    assert context.explain(context.specification.value).verdict == "strictly true"
    assert len(changed.obligations) == 2
    assert changed.explain("extra").verdict == "strictly true"
    removed = changed.with_specification(lambda symbols: None)
    assert removed.specification is None
    assert tuple(item.name for item in removed.obligations) == ("extra",)
    with pytest.raises(ModelInputError, match="Change the specification"):
        changed.without_obligations(changed.specification.value, reason="Change the named declaration instead.")
    assert removed.without_obligations(removed.obligations[0], reason="Retire the added goal.").obligations == ()


def test_catalog_factories_must_use_current_symbols_after_replacement():
    @dataclass(frozen=True)
    class OneClaim:
        claim: Obligation

    context = VerificationContext.for_model(Study)
    original = context.symbols.curve.value
    captured = context.with_specification(lambda symbols: OneClaim(positive(original)))
    with pytest.raises(ModelValidationError, match="context's symbols"):
        captured.replace(captured.symbols.curve, Curve)


def test_an_inference_can_be_explained_without_adding_its_conclusion_as_a_goal():
    @dataclass(frozen=True)
    class Rules:
        implication: InferenceRule

    base = VerificationContext.for_model(Curve)
    conclusion = Obligation("conclusion", base.symbols.value, "rule")
    rule = InferenceRule("rule", conclusion, (positive(base.symbols.x),), "A positive input has a positive square.")
    with pytest.raises(ModelDefinitionError, match="Register"):
        base.explain(rule)
    context = base.with_specification(lambda symbols: Rules(rule))
    observed = context.observe({context.symbols.x: 2})
    assert observed.obligations == ()
    assert observed.explain(observed.specification.implication).verdict == "strictly true"
    assert observed.obligations == ()


def test_catalog_names_distinct_rule_premises_created_with_the_same_helper():
    class Parameters(Modello):
        x = InstanceDummy("x", real=True)
        y = InstanceDummy("y", real=True)

    @dataclass(frozen=True)
    class TwoPremises:
        positive_x: Obligation
        positive_y: Obligation
        implication: InferenceRule

    def specification(symbols: Parameters) -> TwoPremises:
        first, second = positive(symbols.x), positive(symbols.y)
        rule = InferenceRule(
            "original_rule", Obligation("combined", symbols.x + symbols.y, "rule"), (first, second), "Test rule"
        )
        return TwoPremises(first, second, rule)

    context = VerificationContext.for_model(Parameters).with_specification(specification)
    declared = context.specification
    assert declared.implication.premises == (declared.positive_x, declared.positive_y)
    evidence = context.observe(x=2, y=3).explain(declared.implication)
    assert evidence.verdict == "strictly true"
    assert tuple(item.id for item in evidence.premises) == ("combined/positive_x", "combined/positive_y")
    report = context.observe(x=2, y=-1).verify(obligations=(declared.implication.conclusion,))
    assert report.consistent
    inapplicable = report.explain(declared.implication)
    assert inapplicable.verdict == "unresolved"
    assert inapplicable.premises[1].verdict == "strictly false"


def test_named_hints_use_the_obligation_record():
    @dataclass(frozen=True)
    class Bounds:
        ceiling: Obligation

    context = VerificationContext.for_model(Curve).with_specification(
        lambda symbols: Bounds(supremum_below(symbols.value, 2, wrt=symbols.x, domain=Interval(0, 1)))
    )
    bounded = context.with_bound_hint(context.specification.ceiling, upper=1)
    assert bounded.explain(bounded.specification.ceiling).verdict == "strictly true"


@pytest.mark.parametrize(
    "factory", [lambda symbols: {"goal": positive(symbols.x)}, lambda symbols: [positive(symbols.x)]]
)
def test_a_catalog_requires_a_frozen_dataclass(factory):
    with pytest.raises(ModelDefinitionError, match="frozen dataclass"):
        VerificationContext.for_model(Curve).with_specification(factory)


def test_shared_declarations_keep_one_name_and_unsupported_fields_fail_clearly():
    @dataclass(frozen=True)
    class Aliases:
        first: Obligation
        second: Obligation

    base = VerificationContext.for_model(Curve)
    shared = positive(base.symbols.x)
    context = base.with_specification(lambda symbols: Aliases(shared, shared))
    assert context.specification.first is context.specification.second
    assert len(context.obligations) == 1

    @dataclass
    class Mutable:
        goal: Obligation

    with pytest.raises(ModelDefinitionError, match="frozen dataclasses"):
        context.with_specification(lambda symbols: Mutable(positive(symbols.x)))
