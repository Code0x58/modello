"""Portable window contracts and strict coverage for downstream consumers."""

import pickle
import subprocess
import sys
from collections import deque
from decimal import Decimal
from itertools import chain

import pytest
from sympy import S

from examples.validate_stream import CashLedger, CashRow
from modello import Index, InstanceDummy, Modello, component, model_recurrence, model_vector
from modello.contrib.validation import ValidationSummary, WindowValidator
from modello.errors import ModelInputError


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_first_validation_is_at_p_and_requires_p_plus_one_rows(order):
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], start=order, cash=n if order == 0 else rows[n - order].cash + order)

    validator = WindowValidator(History, "rows")
    assert validator.fields == ("cash", "debit")
    assert validator.field_paths == (("cash",), ("debit",))
    assert (validator.order, validator.window_size, validator.first_test_position) == (order, order + 1, order)
    assert validator.fully_testable_position == order
    results = list(validator.iter_results((n, 0) for n in range(order + 2)))
    assert [result.valid for result in results] == [None] * order + [True, True]
    for result in results[:order]:
        assert result.assumed and not result.complete
        assert {check.verdict for check in result.checks} == {"assumed"}
        assert {check.start for check in result.checks} == {order}
    first = results[order]
    assert first.position == order and first.complete and not first.assumed
    assert {check.kind for check in first.checks} == {"domain", "recurrence"}


def test_plain_mappings_and_projected_sequences_have_the_same_results():
    validator = WindowValidator(CashLedger, "rows")
    tuples = ((100, 0), (90, 10), (85, 2), (83, 2))
    mappings = ({"debit": debit, "cash": cash, "metadata": "ignored"} for cash, debit in tuples)
    expected = list(validator.iter_results(tuples))
    assert list(validator.iter_results(mappings)) == expected
    assert [result.valid for result in expected] == [None, True, False, True]
    problem = expected[2].first_invalid
    assert problem is not None
    assert (problem.position, problem.expected, problem.residual, problem.start, problem.kind) == (
        2,
        88,
        -3,
        1,
        "recurrence",
    )


def test_windows_are_stateless_reentrant_and_keep_observed_predecessors():
    validator = WindowValidator(CashLedger, "rows")
    final = validator.check_window(((85, 2), (83, 2)), position=3)
    broken = validator.check_window(((90, 10), (85, 2)), position=2)
    other_account = validator.check_window(((50, 0), (45, 5)), position=1)
    assert final.valid is True and broken.valid is False and other_account.valid is True
    assert validator.check_window(((85, 2), (83, 2)), position=3) == final
    assert broken.first_invalid.expected == 88


def test_position_is_the_absolute_coordinate_not_the_window_length_or_label():
    class TimeDriven(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=rows[n - 1].cash - n)

    validator = WindowValidator(TimeDriven, "rows")
    correct = validator.check_window(((97, 0), (94, 0)), position=3, index="event-at-100")
    assert correct.valid is True
    assert correct.index == "event-at-100"
    wrong = validator.check_window(((97, 0), (94, 0)), position=1)
    assert wrong.valid is False and wrong.first_invalid.expected == 96


@pytest.mark.parametrize("position", [-1, True, 1.5, None, "1"])
def test_invalid_absolute_positions_are_rejected(position):
    with pytest.raises(ModelInputError, match="absolute nonnegative integer"):
        WindowValidator(CashLedger, "rows").check_window(((100, 0),), position=position)


@pytest.mark.parametrize(
    "window,position,required", [([], 0, 1), ([(1, 0)], 10, 2), ([(1, 0)] * 3, 2, 2), ([(1, 0)] * 2, 0, 1)]
)
def test_wrong_window_sizes_are_configuration_errors(window, position, required):
    with pytest.raises(ModelInputError, match=f"requires exactly {required} window rows"):
        WindowValidator(CashLedger, "rows").check_window(window, position=position)


@pytest.mark.parametrize("row", ["bad", 1, (100,), (100, 0, 0)])
def test_wrong_projected_row_shapes_are_rejected(row):
    with pytest.raises(ModelInputError, match="row|fields"):
        WindowValidator(CashLedger, "rows").check_window((row,), position=0)


def test_missing_mapping_keys_are_distinct_from_explicit_nulls():
    validator = WindowValidator(CashLedger, "rows")
    with pytest.raises(ModelInputError, match="Missing row fields: debit"):
        validator.check_window(({"cash": 100},), position=0)
    result = validator.check_window(({"cash": 100, "debit": 0}, {"cash": None, "debit": 10}), position=1)
    assert result.valid is None and not result.complete and not result.assumed
    assert result.first_invalid is None
    assert any(check.verdict == "unknown" for check in result.checks)


def test_labels_are_metadata_and_explicit_none_is_preserved():
    validator = WindowValidator(CashLedger, "rows")
    assert validator.check_window(((100, 0), (90, 10)), position=1).index == 1
    assert validator.check_window(((100, 0), (90, 10)), position=1, index=None).index is None
    results = list(validator.iter_results(((100, 0), (90, 10), (88, 2)), labels=[100, 5, 5]))
    assert [result.position for result in results] == [0, 1, 2]
    assert [result.index for result in results] == [100, 5, 5]
    assert [result.valid for result in results] == [None, True, True]


@pytest.mark.parametrize("labels", [[], ["one"], ["one", "two", "extra"]])
def test_label_stream_length_must_match_rows(labels):
    with pytest.raises(ValueError):
        list(WindowValidator(CashLedger, "rows").iter_results(((100, 0), (90, 10)), labels=labels))


def test_stream_scan_snapshots_reused_mutable_row_containers():
    def stream():
        row = [100, 0]
        yield row
        row[:] = [90, 10]
        yield row
        row[:] = [88, 2]
        yield row

    assert [result.valid for result in WindowValidator(CashLedger, "rows").iter_results(stream())] == [None, True, True]


def test_batch_chaining_and_external_window_management_preserve_history():
    validator = WindowValidator(CashLedger, "rows")
    batches = (((100, 0), (90, 10)), ((85, 2),), ((83, 2),))
    expected = list(validator.iter_results(chain.from_iterable(batches)))
    window = deque(maxlen=validator.window_size)
    actual = []
    position = 0
    for batch in batches:
        for row in batch:
            window.append(row)
            actual.append(validator.check_window(tuple(window), position=position))
            position += 1
    assert actual == expected
    # A new scan resets position and history; this must be a deliberate partition boundary.
    assert next(validator.iter_results(batches[1])).assumed
    with pytest.raises(ModelInputError, match="exactly 2 window rows"):
        validator.check_window(batches[1], position=2)


def test_audit_makes_successful_transitions_and_assumed_history_distinct():
    summary = WindowValidator(CashLedger, "rows").audit(((100, 0), (90, 10), (88, 2)))
    assert (
        summary.row_count,
        summary.checked_rows,
        summary.assumed_rows,
        summary.unknown_rows,
        summary.violated_rows,
    ) == (3, 2, 1, 0, 0)
    assert summary.first_invalid is None
    assert not summary.complete and summary.valid is None


@pytest.mark.parametrize("rows", [(), ((100, 0),)])
def test_empty_and_history_only_audits_never_claim_complete_success(rows):
    summary = WindowValidator(CashLedger, "rows").audit(rows)
    assert summary.row_count == summary.assumed_rows == len(rows)
    assert summary.checked_rows == summary.unknown_rows == summary.violated_rows == 0
    assert summary.first_invalid is None
    assert not summary.complete and summary.valid is None


def test_audit_reports_missing_coverage_that_can_resume_after_a_gap():
    rows = ((100, 0), (90, 10), (None, 2), (86, 2), (84, 2))
    summary = WindowValidator(CashLedger, "rows").audit(rows)
    assert (summary.row_count, summary.checked_rows, summary.assumed_rows, summary.unknown_rows) == (5, 2, 1, 2)
    assert summary.first_invalid is None and summary.valid is None


def test_proven_violation_does_not_hide_incomplete_checks_on_the_same_row():
    validator = WindowValidator(CashLedger, "rows")
    result = validator.check_window(((100, 0), (None, "bad")), position=1)
    assert result.valid is False and not result.complete
    assert result.first_invalid.field == "debit"
    summary = validator.audit(((100, 0), (None, "bad")))
    assert (summary.checked_rows, summary.assumed_rows, summary.unknown_rows, summary.violated_rows) == (0, 1, 0, 1)
    assert summary.valid is False and not summary.complete


def test_audit_reads_past_the_first_failure_to_measure_all_coverage():
    summary = WindowValidator(CashLedger, "rows").audit(((100, 0), (85, 10), (None, 2), (70, 2)))
    assert summary.row_count == 4 and summary.unknown_rows == 2
    assert summary.violated_rows == 1
    assert summary.first_invalid.position == 1


@pytest.mark.parametrize("positions", [(1,), (0, 2), (0, 0), (1, 0)])
def test_audit_aggregation_rejects_omitted_reordered_or_duplicated_positions(positions):
    validator = WindowValidator(CashLedger, "rows")
    rows = ((100, 0), (90, 10), (88, 2))
    results = list(validator.iter_results(rows))
    with pytest.raises(ModelInputError, match="every position from 0 in order"):
        ValidationSummary.from_results(results[position] for position in positions)


def test_periodic_stream_audits_must_name_their_interval_explicitly():
    validator = WindowValidator(CashLedger, "rows")
    results = list(validator.iter_results(((100, 0), (90, 10), (88, 2), (86, 2))))
    with pytest.raises(ModelInputError, match="expected 0"):
        ValidationSummary.from_results(results[2:])
    audit = ValidationSummary.from_results(results[2:], start_position=2)
    assert (audit.start_position, audit.stop_position, audit.row_count) == (2, 4, 2)
    assert audit.valid is True and audit.complete
    assert audit.assumed_rows == 0  # Earlier history is outside this explicit interval.


@pytest.mark.parametrize("start", [-1, True, 1.5])
def test_invalid_audit_intervals_are_rejected(start):
    with pytest.raises(ModelInputError, match="Audit start_position"):
        ValidationSummary.from_results((), start_position=start)


def test_zero_order_audits_can_prove_every_row_without_a_seed_assumption():
    class Identity(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], start=0, cash=rows[n].debit)

    validator = WindowValidator(Identity, "rows")
    summary = validator.audit(((0, 0), (10, 10)))
    assert summary.complete and summary.valid is True and summary.assumed_rows == 0
    bad = validator.audit(((0, 0), (9, 10)))
    assert bad.complete and bad.valid is False


def test_mixed_law_starts_are_visible_in_each_check_not_just_the_mask():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        first = model_recurrence(rows[n], cash=rows[n - 1].cash - rows[n].debit)
        second = model_recurrence(rows[n], start=2, debit=rows[n - 2].debit + 2)

    validator = WindowValidator(History, "rows")
    assert (validator.window_size, validator.first_test_position, validator.fully_testable_position) == (3, 1, 2)
    result = validator.check_window(((100, 0), (99, 1)), position=1)
    assert result.valid is None and not result.assumed and not result.complete
    late = next(check for check in result.checks if check.rule == "second")
    assert (late.start, late.kind, late.verdict) == (2, "recurrence", "assumed")
    bad = validator.check_window(((100, 0), (98, 1)), position=1)
    assert bad.valid is False and not bad.complete
    assert bad.first_invalid.rule == "first"
    summary = validator.audit(((100, 0), (99, 1), (97, 2)))
    assert (summary.checked_rows, summary.assumed_rows, summary.unknown_rows) == (1, 1, 1)


@pytest.mark.parametrize("missing", [None, float("nan"), S.NaN, Decimal("NaN")])
def test_portable_nulls_produce_unknown_checks(missing):
    result = WindowValidator(CashLedger, "rows").check_window(((100, 0), (missing, 0)), position=1)
    assert result.valid is None and result.first_invalid is None


def test_prepared_validator_can_be_serialised_for_a_worker():
    original = WindowValidator(CashLedger, "rows")
    restored = pickle.loads(pickle.dumps(original))
    window = ((Decimal("0.30"), 0), (Decimal("0.20"), Decimal("0.10")))
    assert restored.check_window(window, position=1) == original.check_window(window, position=1)


def test_derived_parameter_assumptions_are_configuration_requirements():
    class InvalidParameters(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rate = InstanceDummy("rate", positive=True)
        rate = S.NegativeOne
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=rate * rows[n - 1].cash)

    with pytest.raises(ModelInputError, match="Defined parameter 'rate'.*domain"):
        WindowValidator(InvalidParameters, "rows")


def test_nested_flat_mappings_and_aliases_need_no_frame_adapter():
    class Holding(Modello):
        quantity = InstanceDummy("quantity", integer=True)
        price = InstanceDummy("price", positive=True)
        value = quantity * price

    class Row(Modello):
        holding = component(Holding)
        cash = InstanceDummy("cash", real=True)
        spendable = cash
        debit = InstanceDummy("debit", real=True)
        nav = cash + holding.value

    class Ledger(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(Row, size=N)
        n = Index("n")
        law = model_recurrence(
            rows[n], cash=rows[n - 1].cash - rows[n].debit, **{"holding.quantity": rows[n - 1].holding.quantity + 1}
        )

    validator = WindowValidator(Ledger, "rows")
    opening = {
        "holding.quantity": 0,
        "holding.price": 10,
        "holding.value": 0,
        "cash": 100,
        "spendable": 100,
        "debit": 0,
        "nav": 100,
    }
    current = {
        "holding.quantity": 1,
        "holding.price": 10,
        "holding.value": 10,
        "cash": 90,
        "spendable": 90,
        "debit": 10,
        "nav": 100,
    }
    result = validator.check_window((opening, current), position=1)
    assert result.valid is True
    assert {check.kind for check in result.checks} == {"domain", "definition", "alias", "recurrence"}
    current["spendable"] = 89
    bad = validator.check_window((opening, current), position=1)
    assert bad.valid is False and bad.first_invalid.kind == "alias"


def test_import_and_execution_need_no_optional_frame_libraries():
    program = """
import sys
from examples.validate_stream import CashLedger
from modello.contrib.validation import WindowValidator
assert not {"pandas", "polars", "numpy"} & sys.modules.keys()
result = WindowValidator(CashLedger, "rows").check_window(((100, 0), (90, 10)), position=1)
assert result.valid is True
assert not {"pandas", "polars", "numpy"} & sys.modules.keys()
"""
    process = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
