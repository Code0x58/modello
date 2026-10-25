"""Observed recurrence checks, independent row identities, and exact amounts."""

from decimal import Decimal
from fractions import Fraction
from itertools import islice

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sympy import Eq, Mul, Piecewise, Pow, Rational, S, Sum, Symbol

from modello import Index, InstanceDummy, Modello, Vector, component, model_recurrence, model_vector
from modello.contrib.pandas import FrameValidator
from modello.contrib.validation import WindowValidator
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
    report = validator.report(frame)
    assert validator.window_size == 2
    assert not report["verdict"].isin(("satisfied", "assumed")).all()
    assert report.loc[report["verdict"].eq("unknown")].empty
    assert report.loc[report["verdict"].eq("violated")][
        ["position", "index", "field", "expected", "observed", "residual"]
    ].values.tolist() == [[2, "a", "cash", 88, 85, -3]]
    postings = report.loc[(report["rule"].eq("posting") & report["verdict"].ne("assumed"))]
    assert postings["position"].tolist() == [1, 2, 3]
    assert postings["verdict"].tolist() == ["satisfied", "violated", "satisfied"]
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_empty_singleton_and_valid_frames(count):
    frame = pd.DataFrame({"cash": [100, 90, 88, 86][:count], "debit": [0, 10, 2, 2][:count]})
    report = FrameValidator(CashLedger, "rows").report(frame)
    assert report["verdict"].isin(("satisfied", "assumed")).all()
    assert report.loc[report["verdict"].eq("violated")].empty
    assert report.loc[report["verdict"].eq("unknown")].empty
    assert list(report.columns) == [
        "position",
        "index",
        "field",
        "rule",
        "observed",
        "expected",
        "residual",
        "verdict",
        "reason",
        "start",
        "kind",
    ]
    assert len(report.loc[(report["rule"].eq("posting") & report["verdict"].ne("assumed"))]) == max(0, count - 1)


def test_the_adapter_checks_the_same_law_as_the_joint_model_solver():
    rows = [CashRow("opening", cash=100, debit=0), CashRow("purchase", cash=90, debit=10)]
    joint = CashLedger("whole", rows=rows)
    assert joint.rows[-1].cash == 90
    assert (
        FrameValidator(CashLedger, "rows")
        .report(pd.DataFrame({"cash": [100, 90], "debit": [0, 10]}))["verdict"]
        .isin(("satisfied", "assumed"))
        .all()
    )
    with pytest.raises(SolutionCountError):
        CashLedger("broken", rows=[rows[0], CashRow("bad", cash=89, debit=10)])
    assert (
        not FrameValidator(CashLedger, "rows")
        .report(pd.DataFrame({"cash": [100, 89], "debit": [0, 10]}))["verdict"]
        .isin(("satisfied", "assumed"))
        .all()
    )


def test_the_source_window_can_be_replaced_without_changing_its_relative_law():
    class ShortLedger(Modello):
        rows = model_vector(CashRow, size=2)
        n = Index("n")
        posting = model_recurrence(rows[n], cash=rows[n - 1].cash - rows[n].debit)

    report = FrameValidator(ShortLedger, "rows").report(pd.DataFrame({"cash": [100, 90, 88], "debit": [0, 10, 2]}))
    assert report["verdict"].isin(("satisfied", "assumed")).all()
    assert report.loc[(report["rule"].eq("posting") & report["verdict"].ne("assumed")), "position"].tolist() == [1, 2]


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
    report = validator.report(frame)
    assert validator.window_size == 3
    assert report.loc[report["verdict"].eq("violated")][
        ["position", "rule", "expected", "residual"]
    ].values.tolist() == [[3, "evolution", 7, 1]]
    assert report.loc[(report["rule"].eq("evolution") & report["verdict"].ne("assumed")), "position"].tolist() == [
        2,
        3,
        4,
    ]


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
    report = validator.report(pd.DataFrame({"a": [0, 1, 3, 6], "b": [1, 1, 2, 3]}))
    assert report["verdict"].isin(("satisfied", "assumed")).all()
    assert report.loc[(report["rule"].eq("first") & report["verdict"].ne("assumed")), "position"].tolist() == [1, 2, 3]
    assert report.loc[(report["rule"].eq("second") & report["verdict"].ne("assumed")), "position"].tolist() == [2, 3]


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
    assert (
        validator.report(pd.DataFrame({"a": [2, 4, 6], "b": [2, 3, 4]}))["verdict"].isin(("satisfied", "assumed")).all()
    )


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
        [[0, 10, 0, 100, 99], [1, 10, 10, 90, 99], [2, 10, 20, 80, 100]],
        columns=pd.MultiIndex.from_tuples(
            [("holding", "quantity"), ("holding", "price"), ("holding", "value"), ("cash", padding), ("nav", padding)]
        ),
    )
    report = FrameValidator(Ledger, "rows").report(frame)
    assert report.loc[report["verdict"].eq("unknown")].empty
    assert report.loc[report["verdict"].eq("violated")][["position", "field", "rule", "expected"]].values.tolist() == [
        [1, "nav", "definition", 100]
    ]
    dotted = frame.copy()
    dotted.columns = ["holding.quantity", "holding.price", "holding.value", "cash", "nav"]
    assert_frame_equal(FrameValidator(Ledger, "rows").report(dotted), report)


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

    frame = pd.DataFrame({"amount": [1, 3], "alias": [2, 4], "quantities.0": [0, 2], "quantities.1": [10, 20]})
    report = FrameValidator(History, "rows").report(frame)
    assert report.loc[report["verdict"].eq("violated")][["position", "field", "rule", "expected"]].values.tolist() == [
        [1, "alias", "alias", 3]
    ]
    frame["alias"] = frame["amount"]
    frame = frame.rename(columns={"quantities.0": "quantities[0]", "quantities.1": "quantities[1]"})
    assert FrameValidator(History, "rows").report(frame)["verdict"].isin(("satisfied", "assumed")).all()


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, pd.NaT, S.NaN, Decimal("NaN"), Decimal("sNaN")])
def test_missing_cells_are_unknown_without_filling_or_long_term_propagation(missing):
    frame = pd.DataFrame({"cash": [100, missing, 90, 88], "debit": [0, 10, 2, 2]}, dtype=object)
    report = FrameValidator(CashLedger, "rows").report(frame)
    assert not report["verdict"].isin(("satisfied", "assumed")).all()
    assert report.loc[report["verdict"].eq("violated")].empty
    assert report.loc[report["verdict"].eq("unknown")][["position", "rule"]].values.tolist() == [
        [1, "domain"],
        [1, "posting"],
        [2, "posting"],
    ]
    postings = report.loc[(report["rule"].eq("posting") & report["verdict"].ne("assumed"))]
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
    report = FrameValidator(Branched, "rows").report(frame)
    assert report.loc[(report["rule"].eq("posting") & report["verdict"].ne("assumed")), "verdict"].tolist() == [
        "satisfied"
    ]
    assert report["verdict"].isin(("satisfied", "assumed")).all()


def test_exact_decimal_fraction_and_large_integer_inputs_keep_their_precision():
    validator = FrameValidator(CashLedger, "rows")
    for amounts in (
        ([Decimal("0.30"), Decimal("0.20"), Decimal("0.10")], [Decimal("0"), Decimal("0.10"), Decimal("0.10")]),
        ([Fraction(1), Fraction(2, 3), Fraction(1, 3)], [0, Fraction(1, 3), Fraction(1, 3)]),
        ([2**60 + 2, 2**60 + 1, 2**60], [0, 1, 1]),
    ):
        cash, debit = amounts
        frame = pd.DataFrame({"cash": cash, "debit": debit})
        assert validator.report(frame)["verdict"].isin(("satisfied", "assumed")).all()
    checks = validator.report(pd.DataFrame({"cash": [Decimal("0.30"), Decimal("0.20")], "debit": [0, Decimal("0.10")]}))
    assert checks.iloc[-1]["expected"] == Rational(1, 5)


def test_float_tolerances_apply_only_to_equations():
    frame = pd.DataFrame({"cash": [0.3, 0.2, 0.100001], "debit": [0, 0.1, 0.1]})
    assert FrameValidator(CashLedger, "rows").report(frame)["verdict"].eq("violated").sum() == 2
    assert (
        FrameValidator(CashLedger, "rows", atol=Decimal("0.00001"))
        .report(frame)["verdict"]
        .isin(("satisfied", "assumed"))
        .all()
    )
    assert (
        FrameValidator(CashLedger, "rows", rtol=Decimal("0.0001"))
        .report(frame)["verdict"]
        .isin(("satisfied", "assumed"))
        .all()
    )


@pytest.mark.parametrize("raw", [-1, float("inf"), float("nan"), 1j, Symbol("t", positive=True), "0.01"])
def test_tolerances_must_be_finite_nonnegative_numbers(raw):
    with pytest.raises(ModelInputError, match="tolerances"):
        FrameValidator(CashLedger, "rows", atol=raw)


@pytest.mark.parametrize("bad", ["100", True, [100], {"amount": 100}, S.Infinity, 1j, Decimal("Infinity")])
def test_bad_numeric_cells_report_domain_breaks_without_evaluating_text(bad):
    frame = pd.DataFrame({"cash": [100, bad], "debit": [0, 0]}, dtype=object)
    report = FrameValidator(CashLedger, "rows").report(frame)
    assert not report["verdict"].isin(("satisfied", "assumed")).all()
    domains = report.loc[report["rule"].eq("domain") & report["verdict"].eq("violated")]
    assert domains[["position", "field"]].values.tolist() == [[1, "cash"]]


def test_symbolic_cells_remain_unknown():
    report = FrameValidator(CashLedger, "rows").report(
        pd.DataFrame({"cash": [100, Symbol("x", real=True)], "debit": [0, 0]})
    )
    assert not report["verdict"].isin(("satisfied", "assumed")).all()
    assert report.loc[report["verdict"].eq("violated")].empty
    assert len(report.loc[report["verdict"].eq("unknown")]) == 2


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

    frame = pd.DataFrame({"numerator": [1, 0], "denominator": [1, 0], "quotient": [1, 0]})
    report = FrameValidator(History, "rows").report(frame)
    assert report.loc[report["rule"].eq("definition") & report["verdict"].ne("assumed"), "reason"].tolist() == [
        "undefined expression"
    ]


def test_scalar_parameters_and_named_constant_definitions():
    class Interest(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        retention = InstanceDummy("retention", positive=True)
        multiplier = 2 * retention
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=multiplier * rows[n - 1].cash - rows[n].debit)

    validator = FrameValidator(Interest, "rows", parameters={"retention": Fraction(1, 2)})
    assert (
        validator.report(pd.DataFrame({"cash": [100, 90], "debit": [0, 10]}))["verdict"]
        .isin(("satisfied", "assumed"))
        .all()
    )
    with pytest.raises(ModelInputError, match="Unbound"):
        FrameValidator(Interest, "rows")
    for parameters in ({"retention": -1}, {"retention": None}, {"retention": "0.5"}, {"multiplier": 1}, {"unknown": 1}):
        with pytest.raises(ModelInputError, match="Parameter"):
            FrameValidator(Interest, "rows", parameters=parameters)
    with pytest.raises(ModelInputError, match="size parameters"):
        FrameValidator(Interest, "rows", parameters={"N": 2, "retention": 1})


def test_explicit_column_mapping_ignores_unmapped_metadata():
    frame = pd.DataFrame({"Cash Balance": [100, 90], "Cash Debit": [0, 10], "comment": ["seed", "posting"]})
    report = FrameValidator(CashLedger, "rows").report(frame, columns={"cash": "Cash Balance", "debit": "Cash Debit"})
    assert report["verdict"].isin(("satisfied", "assumed")).all()


def test_explicit_column_mapping_can_disambiguate_normalized_paths():
    frame = pd.DataFrame([[100, 1, 0], [90, 1, 10]], columns=pd.Index(["cash", ("cash", ""), "debit"]))
    validator = FrameValidator(CashLedger, "rows")
    with pytest.raises(ModelInputError, match="Ambiguous columns"):
        validator.report(frame)
    assert validator.report(frame, columns={"cash": "cash"})["verdict"].isin(("satisfied", "assumed")).all()


def test_multiindex_row_labels_and_column_mapping_require_complete_labels():
    frame = pd.DataFrame(
        [[100, 0], [90, 10]],
        index=pd.MultiIndex.from_tuples([("account", "opening"), ("account", "purchase")]),
        columns=pd.MultiIndex.from_tuples([("cash", "observed"), ("debit", "observed")]),
    )
    validator = FrameValidator(CashLedger, "rows")
    report = validator.report(frame, columns={"cash": ("cash", "observed"), "debit": ("debit", "observed")})
    assert report["verdict"].isin(("satisfied", "assumed")).all()
    assert report.iloc[-1]["index"] == ("account", "purchase")
    with pytest.raises(ModelInputError, match="complete column label"):
        validator.report(frame, columns={"cash": "cash", "debit": "debit"})


def test_field_assumptions_are_checked_even_when_equations_pass_with_tolerance():
    class WholeCash(CashRow):
        cash = InstanceDummy("cash", integer=True, nonnegative=True)

    class Ledger(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(WholeCash, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=rows[n - 1].cash - rows[n].debit)

    report = FrameValidator(Ledger, "rows", atol=1).report(pd.DataFrame({"cash": [1, -0.1], "debit": [0, 1]}))
    assert report.loc[report["verdict"].eq("violated")][["position", "field", "rule"]].values.tolist() == [
        [1, "cash", "domain"]
    ]
    assert report.iloc[-1]["verdict"] == "satisfied"


def test_invalid_bindings_raise_input_errors():
    validator = FrameValidator(CashLedger, "rows")
    for frame, columns, match in (
        (pd.DataFrame({"cash": [1]}), None, "Missing frame columns: debit"),
        (pd.DataFrame([[1, 2]], columns=["cash", "cash"]), None, "unique"),
        (pd.DataFrame({"cash": [1], "debit": [0]}), {"unknown": "cash"}, "Unknown row field"),
        (pd.DataFrame({"cash": [1], "debit": [0]}), {"cash": "missing"}, "Missing frame column"),
    ):
        with pytest.raises(ModelInputError, match=match):
            validator.report(frame, columns=columns)
    with pytest.raises(ModelInputError, match="DataFrame"):
        validator.report([{"cash": 1, "debit": 0}])
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
    assert all(check.verdict == "assumed" for check in islice(checks, 3))
    assert len(consumed) == 1
    remainder = list(checks)
    assert next(check for check in remainder if check.rule == "posting").observed == 89
    # Reusing a plan starts a fresh independent history.
    assert (
        validator.report(pd.DataFrame({"cash": [5, 4], "debit": [0, 1]}))["verdict"]
        .isin(("satisfied", "assumed"))
        .all()
    )


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([10, 3, 3, 27], name="event"),
        pd.DatetimeIndex(["2026-01-01", "2026-01-03", "2026-01-08", "2026-01-20"], tz="Europe/London", name="at"),
        pd.MultiIndex.from_tuples([("a", 2), ("b", 5), ("b", 5), ("a", 9)], names=["source", "event"]),
    ],
)
def test_boolean_validation_preserves_every_position_and_index_metadata(index):
    frame = pd.DataFrame({"cash": [100, 90, 85, 83], "debit": [0, 10, 2, 2]}, index=index)
    before = frame.copy(deep=True)
    expected = pd.Series([pd.NA, True, False, True], index=index, dtype="boolean", name="valid")
    assert_series_equal(FrameValidator(CashLedger, "rows").validate(frame), expected)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_second_order_short_frames_keep_assumed_history_as_missing(count):
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], start=2, cash=2 * rows[n - 1].cash - rows[n - 2].cash)

    frame = pd.DataFrame({"cash": [1, 3, 5, 7][:count], "debit": [0] * count}, index=pd.RangeIndex(0, count * 3, 3))
    validator = FrameValidator(History, "rows")
    assert validator.window_size == 3
    expected = pd.Series(([pd.NA] * 2 + [True] * 2)[:count], index=frame.index, dtype="boolean", name="valid")
    assert_series_equal(validator.validate(frame), expected)
    assert validator.first_invalid(frame) is None
    assert validator.report(frame)["position"].tolist() == [position for position in range(count) for _ in range(3)]


def test_initial_values_have_explicit_assumed_checks():
    frame = pd.DataFrame({"cash": ["assumed"], "debit": [None]})
    validator = FrameValidator(CashLedger, "rows")
    assert validator.first_invalid(frame) is None
    report = validator.report(frame)
    assert report["verdict"].tolist() == ["assumed", "assumed", "assumed"]
    assert report["start"].tolist() == [1, 1, 1]
    assert_series_equal(validator.validate(frame), pd.Series([pd.NA], dtype="boolean", name="valid"))


def test_missing_dependencies_can_interrupt_and_then_restore_validation():
    frame = pd.DataFrame({"cash": [100, 90, None, 86, 84], "debit": [0, 10, 2, 2, 2]}, index=[0, 3, 7, 10, 20])
    validator = FrameValidator(CashLedger, "rows")
    expected = pd.Series([pd.NA, True, pd.NA, pd.NA, True], index=frame.index, dtype="boolean", name="valid")
    assert_series_equal(validator.validate(frame), expected)
    assert validator.first_invalid(frame) is None
    assert validator.report(frame)["verdict"].eq("unknown").any()


def test_first_invalid_continues_past_unknowns_to_a_later_proven_break():
    frame = pd.DataFrame(
        {"cash": [100, None, 90, 87], "debit": [0, 10, 2, 2]}, index=["seed", "gap", "a", "b"], dtype=object
    )
    validator = FrameValidator(CashLedger, "rows")
    first = validator.first_invalid(frame)
    assert first is not None
    assert (first.position, first.index, first.field, first.observed, first.expected, first.residual) == (
        3,
        "b",
        "cash",
        87,
        88,
        -1,
    )
    expected = pd.Series([pd.NA, pd.NA, pd.NA, False], index=frame.index, dtype="boolean", name="valid")
    assert_series_equal(validator.validate(frame), expected)


def test_first_invalid_stops_before_reading_any_later_row(monkeypatch):
    frame = pd.DataFrame({"cash": [100, 90, 85, 83], "debit": [0, 10, 2, 2]}, index=[0, 10, 10, 40])
    original = frame.itertuples
    consumed = []

    def read_rows(*args, **kwargs):
        for row in original(*args, **kwargs):
            if len(consumed) == 3:
                raise AssertionError("The first violation has already been found")
            consumed.append(row)
            yield row

    monkeypatch.setattr(frame, "itertuples", read_rows)
    first = FrameValidator(CashLedger, "rows").first_invalid(frame)
    assert first is not None
    assert (first.position, first.index, first.expected, first.residual) == (2, 10, 88, -3)
    assert len(consumed) == 3


def test_multiple_laws_preserve_incomplete_coverage_but_still_report_known_breaks():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        cash_law = model_recurrence(rows[n], cash=rows[n - 1].cash - rows[n].debit)
        debit_law = model_recurrence(rows[n], start=3, debit=2 * rows[n - 1].debit)

    frame = pd.DataFrame({"cash": [100, 99, 97, 93], "debit": [0, 1, 2, 4]})
    validator = FrameValidator(History, "rows")
    assert_series_equal(
        validator.validate(frame), pd.Series([pd.NA, pd.NA, pd.NA, True], dtype="boolean", name="valid")
    )
    assert validator.first_invalid(frame) is None
    frame["cash"] = [100, 98, 96, 92]
    assert_series_equal(
        validator.validate(frame), pd.Series([pd.NA, False, pd.NA, True], dtype="boolean", name="valid")
    )
    first = validator.first_invalid(frame)
    assert first is not None
    assert (first.position, first.rule, first.expected) == (1, "cash_law", 99)


@pytest.mark.parametrize("cash,debit", [(None, "bad"), (S.Infinity, None)])
def test_a_proven_violation_takes_precedence_over_unknown_checks(cash, debit):
    frame = pd.DataFrame({"cash": [100, cash], "debit": [0, debit]}, dtype=object)
    validator = FrameValidator(CashLedger, "rows")
    assert_series_equal(validator.validate(frame), pd.Series([pd.NA, False], dtype="boolean", name="valid"))
    first = validator.first_invalid(frame)
    assert first is not None
    assert (first.position, first.rule) == (1, "domain")


def test_a_zero_order_law_can_validate_the_first_row():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], start=0, cash=rows[n].debit + n)

    frame = pd.DataFrame({"cash": [0, 2, 4], "debit": [0, 1, 2]}, index=[0, 10, 40])
    validator = FrameValidator(History, "rows")
    expected = pd.Series([True, True, True], index=frame.index, dtype="boolean", name="valid")
    assert_series_equal(validator.validate(frame), expected)
    frame.loc[0, "cash"] = 1
    first = validator.first_invalid(frame)
    assert first is not None
    assert (first.position, first.expected, first.residual) == (0, 0, 1)


def test_a_later_start_is_distinct_from_the_required_window_size():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], start=4, cash=rows[n - 1].cash - rows[n].debit)

    frame = pd.DataFrame({"cash": [100, -999, 1, 10, 9], "debit": [0, "trusted", 0, 0, 1]})
    validator = FrameValidator(History, "rows")
    assert validator.window_size == 2
    assert_series_equal(
        validator.validate(frame), pd.Series([pd.NA, pd.NA, pd.NA, pd.NA, True], dtype="boolean", name="valid")
    )
    assert validator.first_invalid(frame) is None


@pytest.mark.parametrize("method", ["validate", "report", "first_invalid", "audit"])
def test_native_apis_bind_the_schema_even_when_no_transition_is_available(method):
    operation = getattr(FrameValidator(CashLedger, "rows"), method)
    with pytest.raises(ModelInputError, match="Missing frame columns"):
        operation(pd.DataFrame({"cash": []}))
    with pytest.raises(ModelInputError, match="DataFrame"):
        operation([])
    frame = pd.DataFrame({"Cash": [100, 90], "Debit": [0, 10]})
    columns = {"cash": "Cash", "debit": "Debit"}
    result = operation(frame, columns=columns)
    if method == "validate":
        assert_series_equal(result, pd.Series([pd.NA, True], dtype="boolean", name="valid"))
    elif method == "report":
        assert result["verdict"].isin(("satisfied", "assumed")).all()
    elif method == "audit":
        assert (result.row_count, result.assumed_rows, result.checked_rows, result.unknown_rows) == (2, 1, 1, 0)
        assert result.valid is None
    else:
        assert result is None


def test_pandas_projects_and_normalises_rows_before_the_public_window_function(monkeypatch):
    frame = pd.DataFrame(
        {
            "cash": pd.array([100, 90, None, 86, 84], dtype="Int64"),
            "debit": [0, 10, 2, 2, 2],
            "metadata": ["ignored"] * 5,
        }
    )
    frame.index = pd.Index([0, 3, 3, 10, 20], name="event")
    validator = FrameValidator(CashLedger, "rows")
    calls = []
    original = validator.window.check_window

    def capture(window, *, position, index):
        calls.append((window, position, index))
        return original(window, position=position, index=index)

    monkeypatch.setattr(validator.window, "check_window", capture)
    results = list(validator.iter_results(frame))
    assert [result.valid for result in results] == [None, True, None, None, True]
    assert [position for _, position, _ in calls] == [0, 1, 2, 3, 4]
    assert [label for _, _, label in calls] == [0, 3, 3, 10, 20]
    assert calls[0][0] == ((100, 0),)
    assert calls[3][0] == ((None, 2), (86, 2))
    summary = validator.audit(frame)
    assert (summary.row_count, summary.checked_rows, summary.assumed_rows, summary.unknown_rows) == (5, 2, 1, 2)
    assert summary.first_invalid is None and summary.valid is None


def test_report_makes_initial_assumptions_and_start_metadata_visible():
    frame = pd.DataFrame({"cash": [100, 90], "debit": [0, 10]})
    validator = FrameValidator(CashLedger, "rows")
    report = validator.report(frame)
    assert validator.first_test_position == 1
    assert report.loc[report["position"].eq(0), "verdict"].tolist() == ["assumed"] * 3
    assert report.loc[report["position"].eq(1), "verdict"].tolist() == ["satisfied"] * 3
    assert report["start"].tolist() == [1] * 6
    assert report["kind"].tolist() == ["domain", "domain", "recurrence"] * 2
    summary = validator.audit(frame)
    assert (summary.assumed_rows, summary.checked_rows, summary.unknown_rows) == (1, 1, 0)
    assert summary.valid is None and not summary.complete


def test_pandas_can_wrap_a_prepared_window_plan_without_recompiling(monkeypatch):
    window = WindowValidator(CashLedger, "rows", atol=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("The existing plan must be reused")

    monkeypatch.setattr(WindowValidator, "__init__", forbidden)
    validator = FrameValidator.from_window(window)
    assert validator.window is window
    frame = pd.DataFrame({"cash": [100, 89], "debit": [0, 10]})
    assert_series_equal(validator.validate(frame), pd.Series([pd.NA, True], dtype="boolean", name="valid"))
    assert validator.first_invalid(frame) is None
    with pytest.raises(ModelInputError, match="WindowValidator"):
        FrameValidator.from_window(object())
