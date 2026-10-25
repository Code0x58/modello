"""Domain and recurrence scenarios independent of any frame library."""

from decimal import Decimal
from fractions import Fraction

import pytest
from sympy import Eq, Mul, Piecewise, Pow, Rational, S, Sum, Symbol

from modello import Index, InstanceDummy, Modello, model_recurrence, model_vector
from modello.contrib.validation import WindowValidator
from modello.errors import ModelInputError

from ._models import CashLedger, CashRow


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
    errors = [check for result in results for check in result.checks if check.invalid]
    assert [(check.position, check.rule, check.expected, check.residual) for check in errors] == [
        (3, "evolution", 7, 1)
    ]


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


def test_laws_share_observations_with_independent_indices_and_starts():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n, m = Index("n"), Index("m")
        first = model_recurrence(rows[n], cash=rows[n - 1].cash - n)
        second = model_recurrence(rows[m], start=2, debit=rows[m - 2].debit + m)

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
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(
            rows[n],
            cash=Piecewise((0, Eq(rows[n].debit, 0)), (rows[n - 1].cash, True)),
            debit=rows[n - 1].debit,
        )

    result = WindowValidator(History, "rows")(((None, 0), (0, 0)), position=1)
    assert result.valid is True and result.complete


def test_distinct_missing_cells_cannot_cancel_each_other():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=rows[n - 1].cash - rows[n - 1].debit)

    result = WindowValidator(History, "rows")(((None, None), (0, 0)), position=1)
    assert result.valid is None and not result.invalid
    assert result.checks[-1].expected is None


@pytest.mark.parametrize(
    "rows",
    [
        ((Decimal("0.30"), 0), (Decimal("0.20"), Decimal("0.10")), (Decimal("0.10"), Decimal("0.10"))),
        ((Fraction(1), 0), (Fraction(2, 3), Fraction(1, 3)), (Fraction(1, 3), Fraction(1, 3))),
        ((2**60 + 2, 0), (2**60 + 1, 1), (2**60, 1)),
    ],
)
def test_exact_amounts_keep_their_precision(rows):
    results = list(WindowValidator(CashLedger, "rows").iter_results(rows))
    assert [result.valid for result in results] == [None, True, True]
    if isinstance(rows[0][0], Decimal):
        assert results[1].checks[-1].expected == Rational(1, 5)


def test_equation_tolerances_are_explicit():
    rows = ((0.3, 0), (0.2, 0.1), (0.100001, 0.1))
    exact = list(WindowValidator(CashLedger, "rows").iter_results(rows))
    assert [result.valid for result in exact] == [None, False, False]
    approximate = WindowValidator(CashLedger, "rows", atol=Decimal("0.00001"))
    assert [result.valid for result in approximate.iter_results(rows)] == [None, True, True]


@pytest.mark.parametrize("value", [-1, S.Infinity, S.NaN, 1j, Symbol("t", positive=True), "0.01"])
def test_invalid_tolerance_configuration_is_rejected(value):
    with pytest.raises(ModelInputError, match="tolerances"):
        WindowValidator(CashLedger, "rows", atol=value)


@pytest.mark.parametrize("bad", ["100", True, [100], {"amount": 100}, S.Infinity, 1j])
def test_invalid_numeric_cells_are_proven_domain_violations(bad):
    result = WindowValidator(CashLedger, "rows")(((100, 0), (bad, 0)), position=1)
    assert result.valid is False
    problem = next(check for check in result.checks if check.invalid)
    assert problem.field == "cash" and problem.kind == "domain"


def test_symbolic_cells_do_not_prove_numeric_validity():
    result = WindowValidator(CashLedger, "rows")(((100, 0), (Symbol("x", real=True), 0)), position=1)
    assert result.valid is None and not result.invalid


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
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=multiplier * rows[n - 1].cash - rows[n].debit)

    validator = WindowValidator(History, "rows", parameters={"retention": Fraction(1, 2)})
    assert validator(((100, 0), (90, 10)), position=1).valid is True
    with pytest.raises(ModelInputError, match="Unbound"):
        WindowValidator(History, "rows")
    with pytest.raises(ModelInputError, match="Parameter"):
        WindowValidator(History, "rows", parameters={"retention": -1})


@pytest.mark.parametrize(
    "rate,alias,consistent",
    [(1, 1.0, True), (Decimal("1"), Fraction(1), True), (1, 2, False), (2, 1, False)],
)
def test_parameter_aliases_must_agree(rate, alias, consistent):
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rate = InstanceDummy("rate", positive=True)
        alias = rate
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=rate * rows[n - 1].cash)

    if consistent:
        validator = WindowValidator(History, "rows", parameters={"rate": rate, "alias": alias})
        assert [result.valid for result in validator.iter_results(((20, 0), (20, 0)))] == [None, True]
    else:
        with pytest.raises(ModelInputError, match="Conflicting values for parameter"):
            WindowValidator(History, "rows", parameters={"rate": rate, "alias": alias})


@pytest.mark.parametrize("parameters,problem", [({}, "Unbound"), ({"rate": 0}, "domain")])
def test_derived_parameters_require_numeric_dependencies_with_valid_domains(parameters, problem):
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rate = InstanceDummy("rate", real=True)
        intermediate = InstanceDummy("intermediate", positive=True)
        intermediate = -(rate**2) - 1
        multiplier = intermediate + rate**2 + 2
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=multiplier * rows[n - 1].cash)

    with pytest.raises(ModelInputError, match=problem):
        WindowValidator(History, "rows", parameters=parameters)


@pytest.mark.parametrize(
    "expression",
    [
        lambda rows, n: rows[n + 1].cash,
        lambda rows, n: rows[0].cash,
        lambda rows, n: rows[2 * n].cash,
        lambda rows, n: rows[n - 2].cash,
    ],
)
def test_laws_must_have_fixed_causal_lags_covered_by_start(expression):
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=expression(rows, n))

    with pytest.raises(ModelInputError, match="fixed past lags"):
        WindowValidator(History, "rows")


def test_another_vector_cannot_silently_supply_row_inputs():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        other = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=other[n - 1].cash)

    with pytest.raises(ModelInputError, match="selected row vector"):
        WindowValidator(History, "rows")
