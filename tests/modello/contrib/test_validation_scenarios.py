"""Domain and recurrence scenarios independent of any frame library."""

from decimal import Decimal
from fractions import Fraction

import pytest
from sympy import Derivative, Eq, Integral, Lambda, Mul, Piecewise, Pow, Rational, S, Sum, Symbol

from examples.accounting_breaks import Ledger, accounting_rows
from examples.portfolio.models import HoldingState
from modello import Index, InstanceDummy, Modello, Recurrence, Series, inspect_model, model_recurrence, model_vector
from modello.contrib.validation import WindowValidator
from modello.diagnostics import Candidate, FieldValue
from modello.errors import ModelInputError


def test_second_order_evolution_and_moving_sums_use_observed_history():
    class Row(Modello):
        level = InstanceDummy("level", integer=True)
        rolling = InstanceDummy("rolling", integer=True)

    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n, j = Index("n"), Index("j")
        evolution = model_recurrence(rows[n], start=2, level=2 * rows[n - 1].level - rows[n - 2].level)
        reduction = model_recurrence(rows[n], start=2, rolling=Sum(rows[j].level, (j, n - 2, n)))

    validator = WindowValidator(History, "rows")
    results = list(validator.iter_results(((1, 0), (3, 0), (5, 9), (8, 16), (11, 24))))
    assert validator.window_size == 3
    assert [result.valid for result in results] == [None, None, True, False, True]
    errors = [
        (result.position, check.rule, check.expected, check.residual)
        for result in results
        for check in result.checks
        if check.invalid
    ]
    assert errors == [(3, "evolution", 7, 1)]


def test_current_row_cycles_need_no_dependency_scheduler():
    class Row(Modello):
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)

    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n = Index("n")
        first = model_recurrence(rows[n], start=0, a=rows[n].b + n)
        second = model_recurrence(rows[n], start=0, b=rows[n].a - n)

    assert WindowValidator(History, "rows").audit(((2, 2), (4, 3), (6, 4))).valid is True


@pytest.mark.parametrize("projected,valid", [(30, True), (29, False), (None, None)])
def test_row_template_queries_retain_the_complete_field_scope(projected, valid):
    class Row(Modello):
        quantity = InstanceDummy("quantity", real=True)
        cash = InstanceDummy("cash", real=True)
        j = Index("j", nonnegative=True)
        trace = Series("trace", j, real=True)
        growth = Recurrence(trace[j], 2 * trace[j - 1], initial={0: quantity})
        projected = growth.at(1)

    class Ledger(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n = Index("n")
        posting = model_recurrence(rows[n], quantity=rows[n - 1].quantity - rows[n].cash)

    results = list(WindowValidator(Ledger, "rows").iter_results(((20, 0, 40), (15, 5, projected))))
    assert results[0].assumed
    assert results[1].valid == valid
    domains = {check.field: check.valid for check in results[1].checks if check.kind == "domain"}
    assert domains["quantity"] and domains["cash"]
    definition = next(check for check in results[1].checks if check.kind == "definition")
    assert definition.expected == 30


@pytest.mark.parametrize(
    "squared,gradient,integral,after,valid",
    [
        (4, 4, Rational(8, 3), 5, True),
        (4, 0, Rational(8, 3), 5, False),
        (4, 4, 0, 5, False),
        (5, 4, Rational(8, 3), 6, False),
    ],
)
def test_named_operator_definitions_use_core_semantics_without_fitting_observations(
    squared, gradient, integral, after, valid
):
    class Row(Modello):
        x = InstanceDummy("x", real=True)
        squared = x**2
        gradient = Derivative(squared, x, evaluate=False)
        reciprocal = 1 / Derivative(squared, x, evaluate=False)
        integral = Integral(squared, (x, 0, x))
        after = squared + 1

    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n = Index("n")
        identity = model_recurrence(rows[n], start=0, x=rows[n].x)

    values = {
        "x": 2,
        "squared": squared,
        "gradient": gradient,
        "reciprocal": Rational(1, 4),
        "integral": integral,
        "after": after,
    }
    problem = inspect_model(Row, "comparison")
    candidate = Candidate(tuple(FieldValue(problem.field(name), value) for name, value in values.items()))
    core = problem.check(candidate).verdict
    result = WindowValidator(History, "rows")((values,), position=0)
    assert (core == "satisfied") == result.valid == valid
    checks = {check.field: check for check in result.checks if check.kind == "definition"}
    assert checks["gradient"].expected == 4
    assert checks["reciprocal"].expected == Rational(1, 4)
    assert checks["integral"].expected == Rational(8, 3)
    assert checks["after"].expected == squared + 1


@pytest.mark.parametrize(
    "denominator,conditional,valid", [(0, False, False), (2, False, True), (0, True, True), (2, True, True)]
)
def test_operator_resolution_preserves_source_poles_and_inactive_branches(denominator, conditional, valid):
    def formula(p, constant, x):
        term = Mul(Pow(p, -1, evaluate=False), Derivative(constant, x, evaluate=False), evaluate=False)
        return Piecewise((0, Eq(p, 0)), (term, True)) if conditional else term

    class Row(Modello):
        x = InstanceDummy("x", real=True)
        p = InstanceDummy("p", real=True)
        constant = S.One
        term = formula(p, constant, x)

    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n = Index("n")
        identity = model_recurrence(rows[n], start=0, x=rows[n].x)

    values = {"x": 2, "p": denominator, "constant": 1, "term": 0}
    problem = inspect_model(Row, "pole")
    candidate = Candidate(tuple(FieldValue(problem.field(name), value) for name, value in values.items()))
    result = WindowValidator(History, "rows")((values,), position=0)
    assert (problem.check(candidate).verdict == "satisfied") == result.valid == valid


def test_laws_share_observations_with_independent_indices_and_starts():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(HoldingState, size=N)
        n, m = Index("n"), Index("m")
        first = model_recurrence(rows[n], quantity=rows[n - 1].quantity - n)
        second = model_recurrence(rows[m], start=2, cash=rows[m - 2].cash + m)

    validator = WindowValidator(History, "rows")
    rows = ((40, 0), (39, 0), (37, 2), (34, 3))
    results = list(validator.iter_results(rows))
    assert [result.valid for result in results] == [None, None, True, True]
    for position, result in enumerate(results):
        window = rows[max(0, position + 1 - validator.window_size) : position + 1]
        assert validator(window, position=position) == result
    assert not results[1].assumed and not results[1].complete


def test_an_inactive_branch_can_ignore_a_missing_predecessor():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(
            rows[n],
            quantity=Piecewise((0, Eq(rows[n].cash, 0)), (rows[n - 1].quantity, True)),
            cash=rows[n - 1].cash,
        )

    result = WindowValidator(History, "rows")(((None, 0), (0, 0)), position=1)
    assert result.valid is True and result.complete


def test_distinct_missing_cells_cannot_cancel_each_other():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quantity=rows[n - 1].quantity - rows[n - 1].cash)

    result = WindowValidator(History, "rows")(((None, None), (0, 0)), position=1)
    assert result.valid is None and not result.invalid
    assert result.checks[-1].expected is None


@pytest.mark.parametrize(
    "amounts",
    [
        ((Decimal("0.30"), 0), (Decimal("0.20"), Decimal("0.10")), (Decimal("0.10"), Decimal("0.10"))),
        ((Fraction(1), 0), (Fraction(2, 3), Fraction(1, 3)), (Fraction(1, 3), Fraction(1, 3))),
        ((2**60 + 2, 0), (2**60 + 1, 1), (2**60, 1)),
    ],
)
def test_exact_amounts_keep_their_precision(validator, amounts):
    opening = accounting_rows()[0]
    rows = [dict(opening, cash=balance, nav=balance, cash_change=-debit) for balance, debit in amounts]
    results = list(validator.iter_results(rows))
    assert [result.valid for result in results] == [None, True, True]
    if isinstance(amounts[0][0], Decimal):
        check = next(check for check in results[1].checks if check.kind == "recurrence" and check.field == "cash")
        assert check.expected == Rational(1, 5)


def test_equation_tolerances_are_explicit():
    opening = accounting_rows()[0]
    rows = [
        dict(opening, cash=balance, nav=balance, cash_change=movement)
        for balance, movement in ((0.3, 0), (0.2, -0.1), (0.100001, -0.1))
    ]
    exact = list(WindowValidator(Ledger, "rows").iter_results(rows))
    assert [result.valid for result in exact] == [None, False, False]
    approximate = WindowValidator(Ledger, "rows", atol=Decimal("0.00001"))
    assert [result.valid for result in approximate.iter_results(rows)] == [None, True, True]


@pytest.mark.parametrize("value", [-1, S.Infinity, S.NaN, 1j, Symbol("t", positive=True), "0.01"])
def test_invalid_tolerance_configuration_is_rejected(value):
    with pytest.raises(ModelInputError, match="tolerances"):
        WindowValidator(Ledger, "rows", atol=value)


@pytest.mark.parametrize("bad", ["100", True, [100], {"amount": 100}, S.Infinity, 1j, Lambda(Symbol("x"), Symbol("x"))])
def test_invalid_numeric_cells_are_proven_domain_violations(validator, rows, bad):
    rows[1]["cash"] = bad
    result = validator(rows[:2], position=1)
    assert result.invalid
    problem = next(check for check in result.checks if check.invalid)
    assert problem.field == "cash" and problem.kind == "domain"


def test_symbolic_cells_do_not_prove_numeric_validity(validator, rows):
    rows[1]["cash"] = Symbol("x", real=True)
    result = validator(rows[:2], position=1)
    assert result.valid is None and not result.invalid


@pytest.mark.parametrize(
    "field,value,valid",
    [
        ("holding.price", Symbol("negative_price", negative=True), False),
        ("cash", Symbol("nonreal_cash", real=False), False),
        ("holding.price", Symbol("positive_price", positive=True), None),
    ],
)
def test_symbolic_domains_preserve_proven_violations(validator, rows, field, value, valid):
    rows[1][field] = value
    result = validator(rows[:2], position=1)
    check = next(check for check in result.checks if check.kind == "domain" and check.field == field)
    assert check.valid == valid
    assert result.valid == valid


@pytest.mark.parametrize("assumptions", [{"even": True}, {"prime": True}])
def test_domain_checks_retain_assumptions_beyond_the_expected_set(assumptions):
    class Row(Modello):
        value = InstanceDummy("value", **assumptions)

    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], start=0, value=rows[n].value)

    result = next(WindowValidator(History, "rows").iter_results(((9,),)))
    domain, equation = result.checks
    assert domain.expected.contains(9) is S.true
    assert domain.invalid and equation.valid
    assert result.invalid and result.complete


def test_source_poles_are_retained_when_the_numerator_is_zero():
    class Row(Modello):
        numerator = InstanceDummy("numerator", real=True)
        denominator = InstanceDummy("denominator", real=True)
        quotient = Mul(numerator, Pow(denominator, -1, evaluate=False), evaluate=False)

    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quotient=rows[n - 1].quotient)

    result = WindowValidator(History, "rows")(((1, 1, 1), (0, 0, 0)), position=1)
    assert result.valid is False
    assert next(check for check in result.checks if check.invalid).reason == "undefined expression"


def test_parameter_definitions_expand_without_inference():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        retention = InstanceDummy("retention", positive=True)
        multiplier = 2 * retention
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quantity=multiplier * rows[n - 1].quantity - rows[n].cash)

    validator = WindowValidator(History, "rows", parameters={"retention": Fraction(1, 2)})
    assert validator(((100, 0), (90, 10)), position=1).valid is True
    with pytest.raises(ModelInputError, match="Unbound"):
        WindowValidator(History, "rows")
    with pytest.raises(ModelInputError, match="Parameter"):
        WindowValidator(History, "rows", parameters={"retention": -1})


@pytest.mark.parametrize("inherited", [False, True])
def test_parent_parameters_do_not_bind_to_reused_row_declarations(inherited):
    class History(HoldingState if inherited else Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        if not inherited:
            rate = HoldingState.quantity
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quantity=HoldingState.quantity * rows[n - 1].quantity)

    with pytest.raises(ModelInputError, match="Unbound"):
        WindowValidator(History, "rows")

    parameter = "quantity" if inherited else "rate"
    validator = WindowValidator(History, "rows", parameters={parameter: 2})
    assert validator(((1, 0), (2, 0)), position=1).valid
    result = validator(((1, 0), (3, 0)), position=1)
    assert result.invalid and result.checks[-1].expected == 2
    problem = inspect_model(History, "comparison", N=2)
    values = {("N",): 2, (parameter,): 2}
    if inherited:
        values[("cash",)] = 0
    values.update(
        (("rows", str(position), field), value)
        for position, row in enumerate(((1, 0), (3, 0)))
        for field, value in zip(validator.fields, row, strict=True)
    )
    candidate = Candidate(tuple(FieldValue(problem.field(*path), value) for path, value in values.items()))
    assert problem.check(candidate).verdict == "violated"


@pytest.mark.parametrize(
    "query,rate,quantity,valid",
    [
        (lambda squared, rate: Derivative(squared, rate, evaluate=False), 2, 24, True),
        (lambda squared, rate: Derivative(squared, rate, evaluate=False), 2, 20, False),
        (lambda squared, rate: 1 / Derivative(squared, rate, evaluate=False), 2, Rational(81, 4), True),
        (lambda squared, rate: 1 / Derivative(squared, rate, evaluate=False), 2, 20, False),
        (lambda squared, rate: 1 / Derivative(squared, rate, evaluate=False), 0, 20, False),
        (lambda squared, rate: Integral(squared, (rate, 0, rate)), 2, Rational(68, 3), True),
        (lambda squared, rate: Integral(squared, (rate, 0, rate)), 2, 28, False),
    ],
)
def test_recurrence_operators_resolve_parent_definitions_before_parameters(query, rate, quantity, valid):
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rate = InstanceDummy("rate", real=True)
        squared = rate**2
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quantity=rows[n - 1].quantity + query(squared, rate))

    rows = ((20, 0), (quantity, 0))
    validator = WindowValidator(History, "rows", parameters={"rate": rate})
    result = list(validator.iter_results(rows))[-1]
    problem = inspect_model(History, "comparison", N=len(rows))
    values = {("N",): len(rows), ("rate",): rate, ("squared",): rate**2}
    values.update(
        (("rows", str(position), name), value)
        for position, row in enumerate(rows)
        for name, value in zip(validator.fields, row, strict=True)
    )
    candidate = Candidate(tuple(FieldValue(problem.field(*path), value) for path, value in values.items()))
    assert (problem.check(candidate).verdict == "satisfied") == result.valid == valid


def test_recurrence_queries_use_the_parent_template_context():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rate = InstanceDummy("rate", real=True)
        squared = rate**2
        j = Index("j", nonnegative=True)
        trace = Series("trace", j, real=True)
        growth = Recurrence(trace[j], trace[j - 1] + squared, initial={0: 0})
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quantity=rows[n - 1].quantity + trace[1])

    validator = WindowValidator(History, "rows", parameters={"rate": 2})
    assert validator(((20, 0), (24, 0)), position=1).valid
    assert validator(((20, 0), (20, 0)), position=1).invalid


@pytest.mark.parametrize(
    "rate,alias,consistent",
    [(1, 1.0, True), (Decimal("1"), Fraction(1), True), (1, 2, False), (2, 1, False)],
)
def test_parameter_aliases_must_agree(rate, alias, consistent):
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rate = InstanceDummy("rate", positive=True)
        alias = rate
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quantity=rate * rows[n - 1].quantity)

    if consistent:
        validator = WindowValidator(History, "rows", parameters={"rate": rate, "alias": alias})
        assert [result.valid for result in validator.iter_results(((20, 0), (20, 0)))] == [None, True]
    else:
        with pytest.raises(ModelInputError, match="Conflicting values for parameter"):
            WindowValidator(History, "rows", parameters={"rate": rate, "alias": alias})


@pytest.mark.parametrize("value", [1.0, 2])
def test_declared_parameter_alias_definitions_must_agree(value):
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rate = InstanceDummy("rate", positive=True)
        alias = rate
        rate = S.One
        alias = S(value)
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quantity=rate * rows[n - 1].quantity)

    if value == 1:
        assert WindowValidator(History, "rows")(((20, 0), (20, 0)), position=1).valid
    else:
        with pytest.raises(ModelInputError, match="definitions"):
            WindowValidator(History, "rows")


def test_resolvable_scalar_constants_are_checked_as_configuration_even_when_unused():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        unused = InstanceDummy("unused", positive=True)
        unused = S.NegativeOne
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quantity=rows[n - 1].quantity)

    with pytest.raises(ModelInputError, match="Defined parameter 'unused'.*domain"):
        WindowValidator(History, "rows")


@pytest.mark.parametrize(
    "definition,rate,invalid",
    [
        (lambda rate: 1 / rate, 0, True),
        (lambda rate: Mul(0, Pow(rate, -1, evaluate=False), evaluate=False), 0, True),
        (lambda rate: 1 / rate, 2, False),
        (lambda rate: Piecewise((0, Eq(rate, 0)), (1 / rate, True)), 0, False),
    ],
)
def test_unused_scalar_definitions_must_remain_defined(definition, rate, invalid):
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rate = InstanceDummy("rate", real=True)
        unused = definition(rate)
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quantity=rows[n - 1].quantity - rows[n].cash)

    if invalid:
        with pytest.raises(ModelInputError, match="Defined parameter 'unused'.*domain"):
            WindowValidator(History, "rows", parameters={"rate": rate})
    else:
        validator = WindowValidator(History, "rows", parameters={"rate": rate})
        assert validator(((20, 0), (15, 5)), position=1).valid


@pytest.mark.parametrize("parameters,problem", [({}, "domain"), ({"rate": 0}, "domain")])
def test_derived_parameters_require_numeric_dependencies_with_valid_domains(parameters, problem):
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rate = InstanceDummy("rate", real=True)
        intermediate = InstanceDummy("intermediate", positive=True)
        intermediate = -(rate**2) - 1
        multiplier = intermediate + rate**2 + 2
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quantity=multiplier * rows[n - 1].quantity)

    with pytest.raises(ModelInputError, match=problem):
        WindowValidator(History, "rows", parameters=parameters)


def test_core_definition_expansion_can_resolve_a_constant_without_fitting_parameters():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rate = InstanceDummy("rate", positive=True)
        intermediate = rate + 1
        coefficient = intermediate - rate
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quantity=coefficient * rows[n - 1].quantity - rows[n].cash)

    validator = WindowValidator(History, "rows")
    assert validator(((40, 0), (30, 10)), position=1).valid
    assert validator(((40, 0), (31, 10)), position=1).invalid


@pytest.mark.parametrize(
    "expression",
    [
        lambda rows, n: rows[n + 1].quantity,
        lambda rows, n: rows[0].quantity,
        lambda rows, n: rows[2 * n].quantity,
        lambda rows, n: rows[n - 2].quantity,
    ],
)
def test_laws_must_have_fixed_causal_lags_covered_by_start(expression):
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quantity=expression(rows, n))

    with pytest.raises(ModelInputError, match="fixed past lags"):
        WindowValidator(History, "rows")


def test_another_vector_cannot_silently_supply_row_inputs():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(HoldingState, size=N)
        other = model_vector(HoldingState, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quantity=other[n - 1].quantity)

    with pytest.raises(ModelInputError, match="selected row vector"):
        WindowValidator(History, "rows")
