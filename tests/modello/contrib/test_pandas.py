"""Observed recurrence checks, independent row identities, and exact amounts."""

from decimal import Decimal
from fractions import Fraction
from itertools import islice

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sympy import Eq, Mul, Piecewise, Pow, Rational, S, Sum, Symbol

from modello import Index, InstanceDummy, Modello, Vector, component, model_recurrence, model_vector
from modello.contrib.pandas import FrameValidator
from modello.errors import ModelInputError, SolutionCountError


class CashRow(Modello):
    cash = InstanceDummy("cash", real=True)
    debit = InstanceDummy("debit", real=True)


class CashLedger(Modello):
    N = InstanceDummy("N", integer=True, nonnegative=True)
    rows = model_vector(CashRow, size=N)
    n = Index("n")
    posting = model_recurrence(rows[n], cash=rows[n - 1].cash - rows[n].debit)


def test_breaks_use_observed_history_and_keep_duplicate_unsorted_labels():
    frame = pd.DataFrame(
        {"cash": [100, 90, 85, 83], "debit": [0, 10, 2, 2], "metadata": [object()] * 4},
        index=pd.Index(["z", "a", "a", "b"], name="event"),
    )
    before = frame.copy(deep=True)
    validator = FrameValidator(CashLedger, "rows")
    report = validator.validate(frame)
    assert validator.window_size == 2
    assert not report.valid
    assert report.unknown.empty
    assert report.breaks[["position", "index", "field", "expected", "observed", "residual"]].values.tolist() == [
        [2, "a", "cash", 88, 85, -3]
    ]
    postings = report.checks.loc[report.checks["rule"].eq("posting")]
    assert postings["position"].tolist() == [1, 2, 3]
    assert postings["verdict"].tolist() == ["satisfied", "violated", "satisfied"]
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_empty_singleton_and_valid_frames(count):
    frame = pd.DataFrame({"cash": [100, 90, 88, 86][:count], "debit": [0, 10, 2, 2][:count]})
    report = FrameValidator(CashLedger, "rows").validate(frame)
    assert report.valid
    assert report.breaks.empty
    assert report.unknown.empty
    assert list(report.checks.columns) == [
        "position",
        "index",
        "field",
        "rule",
        "observed",
        "expected",
        "residual",
        "verdict",
        "reason",
    ]
    assert len(report.checks.loc[report.checks["rule"].eq("posting")]) == max(0, count - 1)


def test_the_adapter_checks_the_same_law_as_the_joint_model_solver():
    rows = [CashRow("opening", cash=100, debit=0), CashRow("purchase", cash=90, debit=10)]
    joint = CashLedger("whole", rows=rows)
    assert joint.rows[-1].cash == 90
    assert FrameValidator(CashLedger, "rows").validate(pd.DataFrame({"cash": [100, 90], "debit": [0, 10]})).valid
    with pytest.raises(SolutionCountError):
        CashLedger("broken", rows=[rows[0], CashRow("bad", cash=89, debit=10)])
    assert not FrameValidator(CashLedger, "rows").validate(pd.DataFrame({"cash": [100, 89], "debit": [0, 10]})).valid


def test_the_source_window_can_be_replaced_without_changing_its_relative_law():
    class ShortLedger(Modello):
        rows = model_vector(CashRow, size=2)
        n = Index("n")
        posting = model_recurrence(rows[n], cash=rows[n - 1].cash - rows[n].debit)

    report = FrameValidator(ShortLedger, "rows").validate(pd.DataFrame({"cash": [100, 90, 88], "debit": [0, 10, 2]}))
    assert report.valid
    assert report.checks.loc[report.checks["rule"].eq("posting"), "position"].tolist() == [1, 2]


def test_second_order_laws_and_moving_sums_share_a_three_row_window():
    class Row(Modello):
        level = InstanceDummy("level", integer=True)
        rolling = InstanceDummy("rolling", integer=True)

    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n, j = Index("n"), Index("j")
        evolution = model_recurrence(rows[n], start=2, level=2 * rows[n - 1].level - rows[n - 2].level)
        reduction = model_recurrence(rows[n], start=2, rolling=Sum(rows[j].level, (j, n - 2, n)))

    frame = pd.DataFrame({"level": [1, 3, 5, 8, 11], "rolling": [0, 0, 9, 16, 24]})
    validator = FrameValidator(History, "rows")
    report = validator.validate(frame)
    assert validator.window_size == 3
    assert report.breaks[["position", "rule", "expected", "residual"]].values.tolist() == [[3, "evolution", 7, 1]]
    assert report.checks.loc[report.checks["rule"].eq("evolution"), "position"].tolist() == [2, 3, 4]


def test_laws_keep_separate_starts_and_current_row_dependencies():
    class Row(Modello):
        a = InstanceDummy("a", integer=True)
        b = InstanceDummy("b", integer=True)

    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n = Index("n")
        first = model_recurrence(rows[n], a=rows[n - 1].a + rows[n].b)
        second = model_recurrence(rows[n], start=2, b=rows[n - 1].b + rows[n - 2].b)

    validator = FrameValidator(History, "rows")
    assert validator.window_size == 3
    report = validator.validate(pd.DataFrame({"a": [0, 1, 3, 6], "b": [1, 1, 2, 3]}))
    assert report.valid
    assert report.checks.loc[report.checks["rule"].eq("first"), "position"].tolist() == [1, 2, 3]
    assert report.checks.loc[report.checks["rule"].eq("second"), "position"].tolist() == [2, 3]


def test_zero_order_and_implicit_current_row_equations_need_no_schedule():
    class Row(Modello):
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)

    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n = Index("n")
        first = model_recurrence(rows[n], start=0, a=rows[n].b + n)
        second = model_recurrence(rows[n], start=0, b=rows[n].a - n)

    validator = FrameValidator(History, "rows")
    assert validator.window_size == 1
    assert validator.validate(pd.DataFrame({"a": [2, 4, 6], "b": [2, 3, 4]})).valid


@pytest.mark.parametrize("padding", ["", None])
def test_nested_multiindex_columns_row_equations_and_nested_law_targets(padding):
    class Holding(Modello):
        quantity = InstanceDummy("quantity", integer=True)
        price = InstanceDummy("price", positive=True)
        value = quantity * price

    class Row(Modello):
        holding = component(Holding)
        cash = InstanceDummy("cash", real=True)
        nav = cash + holding.value

    class Ledger(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n = Index("n")
        holding = model_recurrence(rows[n].holding, quantity=rows[n - 1].holding.quantity + 1)

    frame = pd.DataFrame(
        [[0, 10, 0, 100, 99], [1, 10, 10, 90, 100], [2, 10, 20, 80, 100]],
        columns=pd.MultiIndex.from_tuples(
            [("holding", "quantity"), ("holding", "price"), ("holding", "value"), ("cash", padding), ("nav", padding)]
        ),
    )
    report = FrameValidator(Ledger, "rows").validate(frame)
    assert report.unknown.empty
    assert report.breaks[["position", "field", "rule", "expected"]].values.tolist() == [[0, "nav", "definition", 100]]
    dotted = frame.copy()
    dotted.columns = ["holding.quantity", "holding.price", "holding.value", "cash", "nav"]
    assert_frame_equal(FrameValidator(Ledger, "rows").validate(dotted).checks, report.checks)


def test_fixed_vector_fields_and_aliases_are_observations_too():
    class Row(Modello):
        amount = InstanceDummy("amount", real=True)
        alias = amount
        quantities = Vector("quantities", size=2, integer=True)

    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n = Index("n")
        posting = model_recurrence(rows[n], amount=rows[n - 1].amount + rows[n].quantities[0])

    frame = pd.DataFrame({"amount": [1, 3], "alias": [2, 3], "quantities.0": [0, 2], "quantities.1": [10, 20]})
    report = FrameValidator(History, "rows").validate(frame)
    assert report.breaks[["position", "field", "rule", "expected"]].values.tolist() == [[0, "alias", "alias", 1]]
    frame["alias"] = frame["amount"]
    frame = frame.rename(columns={"quantities.0": "quantities[0]", "quantities.1": "quantities[1]"})
    assert FrameValidator(History, "rows").validate(frame).valid


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, pd.NaT, S.NaN, Decimal("NaN"), Decimal("sNaN")])
def test_missing_cells_are_unknown_without_filling_or_long_term_propagation(missing):
    frame = pd.DataFrame({"cash": [100, missing, 90, 88], "debit": [0, 10, 2, 2]}, dtype=object)
    report = FrameValidator(CashLedger, "rows").validate(frame)
    assert not report.valid
    assert report.breaks.empty
    assert report.unknown[["position", "rule"]].values.tolist() == [[1, "domain"], [1, "posting"], [2, "posting"]]
    postings = report.checks.loc[report.checks["rule"].eq("posting")]
    assert postings["verdict"].tolist() == ["unknown", "unknown", "satisfied"]
    assert postings.iloc[0]["expected"] == 90
    assert postings.iloc[1]["expected"] is None


def test_an_inactive_branch_does_not_need_its_missing_input():
    class Branched(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        posting = model_recurrence(rows[n], cash=Piecewise((0, Eq(rows[n].debit, 0)), (rows[n - 1].cash, True)))

    frame = pd.DataFrame({"cash": [None, 0], "debit": [0, 0]})
    report = FrameValidator(Branched, "rows").validate(frame)
    assert report.checks.loc[report.checks["rule"].eq("posting"), "verdict"].tolist() == ["satisfied"]
    assert not report.valid  # The missing seed is still an unknown domain check.


def test_exact_decimal_fraction_and_large_integer_inputs_keep_their_precision():
    validator = FrameValidator(CashLedger, "rows")
    for amounts in (
        ([Decimal("0.30"), Decimal("0.20"), Decimal("0.10")], [Decimal("0"), Decimal("0.10"), Decimal("0.10")]),
        ([Fraction(1), Fraction(2, 3), Fraction(1, 3)], [0, Fraction(1, 3), Fraction(1, 3)]),
        ([2**60 + 2, 2**60 + 1, 2**60], [0, 1, 1]),
    ):
        cash, debit = amounts
        frame = pd.DataFrame({"cash": cash, "debit": debit})
        assert validator.validate(frame).valid
    checks = validator.validate(
        pd.DataFrame({"cash": [Decimal("0.30"), Decimal("0.20")], "debit": [0, Decimal("0.10")]})
    )
    assert checks.checks.iloc[-1]["expected"] == Rational(1, 5)


def test_float_tolerances_apply_only_to_equations():
    frame = pd.DataFrame({"cash": [0.3, 0.2, 0.100001], "debit": [0, 0.1, 0.1]})
    assert len(FrameValidator(CashLedger, "rows").validate(frame).breaks) == 2
    assert FrameValidator(CashLedger, "rows", atol=Decimal("0.00001")).validate(frame).valid
    assert FrameValidator(CashLedger, "rows", rtol=Decimal("0.0001")).validate(frame).valid


@pytest.mark.parametrize("raw", [-1, float("inf"), float("nan"), 1j, Symbol("t", positive=True), "0.01"])
def test_tolerances_must_be_finite_nonnegative_numbers(raw):
    with pytest.raises(ModelInputError, match="tolerances"):
        FrameValidator(CashLedger, "rows", atol=raw)


@pytest.mark.parametrize("bad", ["100", True, [100], {"amount": 100}, S.Infinity, 1j, Decimal("Infinity")])
def test_bad_numeric_cells_report_domain_breaks_without_evaluating_text(bad):
    frame = pd.DataFrame({"cash": [bad], "debit": [0]}, dtype=object)
    report = FrameValidator(CashLedger, "rows").validate(frame)
    assert not report.valid
    assert report.breaks[["position", "field", "rule"]].values.tolist() == [[0, "cash", "domain"]]


def test_symbolic_cells_remain_unknown():
    report = FrameValidator(CashLedger, "rows").validate(pd.DataFrame({"cash": [Symbol("x", real=True)], "debit": [0]}))
    assert not report.valid
    assert report.breaks.empty
    assert len(report.unknown) == 1


def test_source_poles_are_not_hidden_by_a_zero_numerator():
    class Row(Modello):
        numerator = InstanceDummy("numerator", real=True)
        denominator = InstanceDummy("denominator", real=True)
        quotient = Mul(numerator, Pow(denominator, -1, evaluate=False), evaluate=False)

    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], quotient=rows[n - 1].quotient)

    frame = pd.DataFrame({"numerator": [0], "denominator": [0], "quotient": [0]})
    report = FrameValidator(History, "rows").validate(frame)
    assert report.breaks["reason"].tolist() == ["undefined expression"]


def test_scalar_parameters_and_named_constant_definitions():
    class Interest(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        retention = InstanceDummy("retention", positive=True)
        multiplier = 2 * retention
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=multiplier * rows[n - 1].cash - rows[n].debit)

    validator = FrameValidator(Interest, "rows", parameters={"retention": Fraction(1, 2)})
    assert validator.validate(pd.DataFrame({"cash": [100, 90], "debit": [0, 10]})).valid
    with pytest.raises(ModelInputError, match="Unbound"):
        FrameValidator(Interest, "rows")
    for parameters in ({"retention": -1}, {"retention": None}, {"retention": "0.5"}, {"multiplier": 1}, {"unknown": 1}):
        with pytest.raises(ModelInputError, match="Parameter"):
            FrameValidator(Interest, "rows", parameters=parameters)
    with pytest.raises(ModelInputError, match="size parameters"):
        FrameValidator(Interest, "rows", parameters={"N": 2, "retention": 1})


def test_explicit_column_mapping_ignores_unmapped_metadata():
    frame = pd.DataFrame({"Cash Balance": [100, 90], "Cash Debit": [0, 10], "comment": ["seed", "posting"]})
    report = FrameValidator(CashLedger, "rows").validate(frame, columns={"cash": "Cash Balance", "debit": "Cash Debit"})
    assert report.valid


def test_explicit_column_mapping_can_disambiguate_normalized_paths():
    frame = pd.DataFrame([[100, 1, 0], [90, 1, 10]], columns=pd.Index(["cash", ("cash", ""), "debit"]))
    validator = FrameValidator(CashLedger, "rows")
    with pytest.raises(ModelInputError, match="Ambiguous columns"):
        validator.validate(frame)
    assert validator.validate(frame, columns={"cash": "cash"}).valid


def test_multiindex_row_labels_and_column_mapping_require_complete_labels():
    frame = pd.DataFrame(
        [[100, 0], [90, 10]],
        index=pd.MultiIndex.from_tuples([("account", "opening"), ("account", "purchase")]),
        columns=pd.MultiIndex.from_tuples([("cash", "observed"), ("debit", "observed")]),
    )
    validator = FrameValidator(CashLedger, "rows")
    report = validator.validate(frame, columns={"cash": ("cash", "observed"), "debit": ("debit", "observed")})
    assert report.valid
    assert report.checks.iloc[-1]["index"] == ("account", "purchase")
    with pytest.raises(ModelInputError, match="complete column label"):
        validator.validate(frame, columns={"cash": "cash", "debit": "debit"})


def test_field_assumptions_are_checked_even_when_equations_pass_with_tolerance():
    class WholeCash(CashRow):
        cash = InstanceDummy("cash", integer=True, nonnegative=True)

    class Ledger(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(WholeCash, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=rows[n - 1].cash - rows[n].debit)

    report = FrameValidator(Ledger, "rows", atol=1).validate(pd.DataFrame({"cash": [1, -0.1], "debit": [0, 1]}))
    assert report.breaks[["position", "field", "rule"]].values.tolist() == [[1, "cash", "domain"]]
    assert report.checks.iloc[-1]["verdict"] == "satisfied"


def test_invalid_bindings_raise_input_errors():
    validator = FrameValidator(CashLedger, "rows")
    for frame, columns, match in (
        (pd.DataFrame({"cash": [1]}), None, "Missing frame columns: debit"),
        (pd.DataFrame([[1, 2]], columns=["cash", "cash"]), None, "unique"),
        (pd.DataFrame({"cash": [1], "debit": [0]}), {"unknown": "cash"}, "Unknown row field"),
        (pd.DataFrame({"cash": [1], "debit": [0]}), {"cash": "missing"}, "Missing frame column"),
    ):
        with pytest.raises(ModelInputError, match=match):
            validator.validate(frame, columns=columns)
    with pytest.raises(ModelInputError, match="DataFrame"):
        validator.validate([{"cash": 1, "debit": 0}])
    with pytest.raises(ModelInputError, match="Modello class"):
        FrameValidator(object, "rows")
    with pytest.raises(ModelInputError, match="model_vector"):
        FrameValidator(CashLedger, "unknown")


@pytest.mark.parametrize(
    "expression",
    [
        lambda rows, n: rows[n + 1].cash,
        lambda rows, n: rows[0].cash,
        lambda rows, n: rows[2 * n].cash,
        lambda rows, n: rows[n - 2].cash,
    ],
)
def test_unbounded_or_uncovered_references_are_rejected(expression):
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=expression(rows, n))

    with pytest.raises(ModelInputError, match="fixed past lags"):
        FrameValidator(History, "rows")


def test_another_model_vector_cannot_silently_supply_row_data():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        other = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=other[n - 1].cash)

    with pytest.raises(ModelInputError, match="selected row vector"):
        FrameValidator(History, "rows")
    with pytest.raises(ModelInputError, match="No model_recurrence"):
        FrameValidator(History, "other")


def test_iterating_checks_is_lazy_and_never_constructs_or_solves_models(monkeypatch):
    frame = pd.DataFrame({"cash": [100, 89, 87], "debit": [0, 10, 2]})
    validator = FrameValidator(CashLedger, "rows")
    consumed = []
    original = frame.itertuples

    def observe_rows(*args, **kwargs):
        for row in original(*args, **kwargs):
            consumed.append(row)
            yield row

    monkeypatch.setattr(frame, "itertuples", observe_rows)

    def forbidden(*args, **kwargs):
        raise AssertionError("Validation must not construct or solve a model")

    monkeypatch.setattr(Modello, "__init__", forbidden)
    monkeypatch.setattr("modello.solve", forbidden)
    checks = validator.iter_checks(frame)
    assert all(check.verdict == "satisfied" for check in islice(checks, 2))
    assert len(consumed) == 1
    remainder = list(checks)
    assert next(check for check in remainder if check.rule == "posting").observed == 89
    # Reusing a plan starts a fresh independent history.
    assert validator.validate(pd.DataFrame({"cash": [5, 4], "debit": [0, 1]})).valid
