"""Portable window contracts and strict coverage for downstream consumers."""

import pickle
import subprocess
import sys
from collections import deque
from itertools import chain

import pytest
from sympy import Integer

from examples.accounting_breaks import AccountRow
from examples.portfolio.models import HoldingState
from modello import Index, InstanceDummy, Modello, model_recurrence, model_vector
from modello.contrib.validation import ValidationResult, ValidationSummary, WindowValidator
from modello.errors import ModelInputError


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_first_validation_is_at_p_and_requires_p_plus_one_rows(make_validator, order):
    validator = make_validator(lambda rows, n: {"cash": n if order == 0 else rows[n - order].cash + order}, start=order)
    assert validator.fields == ("quantity", "cash")
    assert validator.field_paths == (("quantity",), ("cash",))
    assert (validator.order, validator.window_size, validator.first_test_position) == (order, order + 1, order)
    assert validator.fully_testable_position == order
    results = list(validator.iter_results((0, n) for n in range(order + 2)))
    assert [result.valid for result in results] == [None] * order + [True, True]
    for result in results[:order]:
        assert result.assumed and not result.complete
        assert all(check.assumed and check.valid is None and not check.invalid for check in result.checks)
        assert {check.start for check in result.checks} == {order}
    first = results[order]
    assert first.position == order and first.complete and not first.assumed
    assert {check.kind for check in first.checks} == {"domain", "recurrence"}


def test_plain_mappings_and_projected_sequences_have_the_same_results(validator, rows):
    tuples = (tuple(row[field] for field in validator.fields) for row in rows)
    mappings = (dict(reversed(tuple(row.items())), metadata="ignored") for row in rows)
    expected = list(validator.iter_results(tuples))
    assert list(validator.iter_results(mappings)) == expected
    assert [result.valid for result in expected] == [None, True, False, True]
    problem = next(check for check in expected[2].checks if check.invalid)
    assert (problem.expected, problem.residual, problem.start, problem.kind) == (30, -1, 1, "recurrence")


def test_validator_is_a_callable_window_function(validator, rows):
    window = rows[: validator.window_size]
    assert isinstance(validator(window, position=1), ValidationResult)
    assert validator(window, position=1).valid
    with pytest.raises(TypeError, match="position"):
        validator(window)


def test_callers_can_select_the_first_error_lazily_or_collect_all_errors(validator, rows):
    rows[-1]["cash"] -= 1
    rows[-1]["nav"] -= 1

    def stream():
        for position, row in enumerate(rows):
            if position == 3:
                raise AssertionError("A first-error search should stop before this row")
            yield row

    first = next(result for result in validator.iter_results(stream()) if result.invalid)
    problem = next(check for check in first.checks if check.invalid)
    assert (first.position, problem.expected, problem.residual) == (2, 30, -1)
    errors = [
        (result.position, check.expected, check.residual)
        for result in validator.iter_results(rows)
        for check in result.checks
        if check.invalid
    ]
    assert errors == [(2, 30, -1), (3, 19, -1)]


def test_checks_and_rows_distinguish_valid_invalid_assumed_and_unknown(validator, rows):
    rows[-1]["cash"] = None
    results = list(validator.iter_results(rows))
    states = [(None, False, True), (True, False, False), (False, True, False), (None, False, False)]
    assert [(result.valid, result.invalid, result.assumed) for result in results] == states
    recurrence_checks = [next(check for check in result.checks if check.kind == "recurrence") for result in results]
    assert [(check.valid, check.invalid, check.assumed) for check in recurrence_checks] == states
    assert [result.complete for result in results] == [False, True, True, False]
    summary = ValidationSummary.from_results(results)
    assert (
        summary.row_count,
        summary.valid_rows,
        summary.invalid_rows,
        summary.assumed_rows,
        summary.unknown_rows,
    ) == (4, 1, 1, 1, 1)
    assert summary.first_invalid is results[2]
    assert (summary.valid, summary.invalid, summary.complete) == (False, True, False)


@pytest.mark.parametrize("cash,failed_fields", [("bad", ["cash", "cash_change"]), (None, ["cash_change"])])
def test_invalid_rows_retain_failing_and_undecidable_checks(validator, rows, cash, failed_fields):
    rows[1].update(cash=cash, cash_change="bad")
    results = list(validator.iter_results(rows[:2]))
    result = results[-1]
    assert validator(rows[:2], position=1) == result
    assert result.invalid and not result.complete
    assert [check.field for check in result.checks if check.invalid] == failed_fields
    assert any(check.valid is None for check in result.checks)
    summary = ValidationSummary.from_results(results)
    assert summary.first_invalid is result
    assert validator.audit(rows[:2]) == summary
    assert (summary.valid_rows, summary.invalid_rows, summary.assumed_rows, summary.unknown_rows) == (0, 1, 1, 0)
    assert summary.invalid and not summary.complete


def test_windows_are_stateless_reentrant_and_keep_observed_predecessors(validator, rows):
    final = validator(rows[2:], position=3)
    broken = validator(rows[1:3], position=2)
    other_account = validator(rows[:2], position=1)
    assert final.valid and broken.invalid and other_account.valid
    assert validator(rows[2:], position=3) == final
    assert next(check for check in broken.checks if check.invalid).expected == 30


def test_position_is_the_absolute_coordinate_not_the_window_length_or_label(make_validator):
    validator = make_validator(lambda rows, n: {"cash": rows[n - 1].cash - n})
    correct = validator(((0, 97), (0, 94)), position=3, index="event-at-100")
    assert correct.valid
    assert correct.index == "event-at-100"
    wrong = validator(((0, 97), (0, 94)), position=1)
    assert wrong.invalid
    assert next(check for check in wrong.checks if check.invalid).expected == 96


@pytest.mark.parametrize("position", [-1, True, 1.5, None, "1"])
def test_invalid_absolute_positions_are_rejected(validator, rows, position):
    with pytest.raises(ModelInputError, match="absolute nonnegative integer"):
        validator(rows[:1], position=position)


@pytest.mark.parametrize("size,position,required", [(0, 0, 1), (1, 10, 2), (3, 2, 2), (2, 0, 1)])
def test_wrong_window_sizes_are_configuration_errors(validator, rows, size, position, required):
    with pytest.raises(ModelInputError, match=f"requires exactly {required} window rows"):
        validator(rows[:size], position=position)


@pytest.mark.parametrize("row", ["bad", 1, (100,), (100, 0, 0)])
def test_wrong_projected_row_shapes_are_rejected(validator, row):
    with pytest.raises(ModelInputError, match="row|fields"):
        validator((row,), position=0)


def test_missing_mapping_keys_are_distinct_from_explicit_nulls(validator, rows):
    del rows[0]["cash_change"]
    with pytest.raises(ModelInputError, match="Missing row fields: cash_change"):
        validator(rows[:1], position=0)
    rows[0]["cash_change"] = 0
    rows[1]["cash"] = None
    result = validator(rows[:2], position=1)
    assert result.valid is None and not result.complete and not result.assumed
    assert not result.invalid
    assert any(check.valid is None and not check.assumed for check in result.checks)


def test_labels_are_metadata_and_explicit_none_is_preserved(validator, rows):
    assert validator(rows[:2], position=1).index == 1
    assert validator(rows[:2], position=1, index=None).index is None
    results = list(validator.iter_results(rows[:3], labels=[100, 5, 5]))
    assert [result.position for result in results] == [0, 1, 2]
    assert [result.index for result in results] == [100, 5, 5]
    assert [result.valid for result in results] == [None, True, False]


@pytest.mark.parametrize("labels", [[], ["one"], ["one", "two", "extra"]])
def test_label_stream_length_must_match_rows(validator, rows, labels):
    with pytest.raises(ValueError):
        list(validator.iter_results(rows[:2], labels=labels))


def test_stream_scan_snapshots_reused_mutable_row_containers(validator, rows):
    def stream():
        row = {}
        for observation in rows:
            row.update(observation)
            yield row

    assert [result.valid for result in validator.iter_results(stream())] == [None, True, False, True]


def test_stream_history_keeps_numeric_values_as_observed_when_read(validator, rows):
    class Cell:
        value = 0
        conversions = 0

        def _sympy_(self):
            self.conversions += 1
            return Integer(self.value)

    cash = Cell()

    def stream():
        for row in rows:
            cash.value = row["cash"]
            yield dict(row, cash=cash)

    results = list(validator.iter_results(stream()))
    assert [result.valid for result in results] == [None, True, False, True]
    assert [
        next(check for check in result.checks if check.rule == "posting" and check.field == "cash").expected
        for result in results
    ] == [None, 20, 30, 19]
    assert cash.conversions == len(results)
    cash.value = 99
    assert all(
        check.observed == row["cash"]
        for result, row in zip(results, rows, strict=True)
        for check in result.checks
        if check.field == "cash"
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("label", [[], ([],)])
def test_window_and_stream_reject_unhashable_labels(validator, rows, streaming, label):
    with pytest.raises(ModelInputError, match="label must be hashable"):
        if streaming:
            list(validator.iter_results(rows[:1], labels=[label]))
        else:
            validator(rows[:1], position=0, index=label)


def test_batch_chaining_and_external_window_management_preserve_history(validator, rows):
    batches = (rows[:2], rows[2:3], rows[3:])
    expected = list(validator.iter_results(chain.from_iterable(batches)))
    window = deque(maxlen=validator.window_size)
    actual = []
    for position, row in enumerate(chain.from_iterable(batches)):
        window.append(row)
        actual.append(validator(tuple(window), position=position))
    assert actual == expected
    assert next(validator.iter_results(batches[1])).assumed
    with pytest.raises(ModelInputError, match=f"exactly {validator.window_size} window rows"):
        validator(batches[1], position=2)


@pytest.mark.parametrize("length,valid_rows,assumed_rows", [(0, 0, 0), (1, 0, 1), (2, 1, 1)])
def test_audits_distinguish_assumed_history_from_valid_transitions(validator, rows, length, valid_rows, assumed_rows):
    summary = validator.audit(rows[:length])
    assert (summary.row_count, summary.valid_rows, summary.assumed_rows) == (length, valid_rows, assumed_rows)
    assert summary.invalid_rows == summary.unknown_rows == 0
    assert summary.first_invalid is None
    assert not summary.complete and summary.valid is None


def test_audit_reports_missing_coverage_that_can_resume_after_a_gap(validator, rows):
    rows[2]["cash"] = None
    rows.append(dict(rows[-1], cash=9, nav=39, **{"holding.quantity": 3, "holding.value": 30}))
    summary = validator.audit(rows)
    assert (summary.row_count, summary.valid_rows, summary.assumed_rows, summary.unknown_rows) == (5, 2, 1, 2)
    assert summary.first_invalid is None and summary.valid is None


def test_audit_reads_past_the_first_failure_to_measure_all_coverage(validator, rows):
    rows[1]["cash_change"] = -19
    rows[2]["cash"] = None
    summary = validator.audit(rows)
    assert summary.row_count == len(rows) and summary.unknown_rows == 2
    assert summary.invalid_rows == 1
    assert summary.first_invalid.position == 1


@pytest.mark.parametrize("positions", [(1,), (0, 2), (0, 0), (1, 0)])
def test_audit_aggregation_rejects_omitted_reordered_or_duplicated_positions(validator, rows, positions):
    results = list(validator.iter_results(rows))
    with pytest.raises(ModelInputError, match="every position from 0 in order"):
        ValidationSummary.from_results(results[position] for position in positions)


def test_periodic_stream_audits_must_name_their_interval_explicitly(validator, rows):
    results = list(validator.iter_results(rows))
    with pytest.raises(ModelInputError, match="expected 0"):
        ValidationSummary.from_results(results[-1:])
    audit = ValidationSummary.from_results(results[-1:], start_position=len(rows) - 1)
    assert (audit.start_position, audit.stop_position, audit.row_count) == (3, 4, 1)
    assert audit.valid and audit.complete
    assert audit.assumed_rows == 0


@pytest.mark.parametrize("start", [-1, True, 1.5])
def test_invalid_audit_intervals_are_rejected(start):
    with pytest.raises(ModelInputError, match="Audit start_position"):
        ValidationSummary.from_results((), start_position=start)


def test_zero_order_audits_can_prove_every_row_without_a_seed_assumption(make_validator):
    validator = make_validator(lambda rows, n: {"cash": rows[n].quantity}, start=0)
    summary = validator.audit(((0, 0), (10, 10)))
    assert summary.complete and summary.valid and summary.assumed_rows == 0
    bad = validator.audit(((0, 0), (10, 9)))
    assert bad.complete and bad.invalid


def test_mixed_law_starts_are_visible_in_each_check_not_just_the_mask():
    class History(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(HoldingState, size=N)
        n = Index("n")
        first = model_recurrence(rows[n], cash=rows[n - 1].cash - rows[n].quantity)
        second = model_recurrence(rows[n], start=2, quantity=rows[n - 2].quantity + 2)

    validator = WindowValidator(History, "rows")
    assert (validator.window_size, validator.first_test_position, validator.fully_testable_position) == (3, 1, 2)
    result = validator(((0, 100), (1, 99)), position=1)
    assert result.valid is None and not result.assumed and not result.complete
    late = next(check for check in result.checks if check.rule == "second")
    assert (late.start, late.kind, late.assumed) == (2, "recurrence", True)
    assert late.valid is None and not late.invalid
    bad = validator(((0, 100), (1, 98)), position=1)
    assert bad.invalid and not bad.complete
    assert next(check for check in bad.checks if check.invalid).rule == "first"
    summary = validator.audit(((0, 100), (1, 99), (2, 97)))
    assert (summary.valid_rows, summary.assumed_rows, summary.unknown_rows) == (1, 1, 1)


def test_prepared_validator_can_be_serialised_for_a_worker(validator, rows):
    restored = pickle.loads(pickle.dumps(validator))
    assert list(restored.iter_results(rows)) == list(validator.iter_results(rows))


def test_nested_flat_mappings_and_aliases_need_no_frame_adapter(make_validator, rows):
    class Row(AccountRow):
        spendable = AccountRow.cash

    for row in rows:
        row["spendable"] = row["cash"]
    validator = make_validator(lambda rows, n: {"cash": rows[n - 1].cash + rows[n].cash_change}, row_model=Row)
    result = validator(rows[:2], position=1)
    assert result.valid
    assert {check.kind for check in result.checks} == {"domain", "definition", "alias", "recurrence"}
    rows[1]["spendable"] -= 1
    bad = validator(rows[:2], position=1)
    assert bad.invalid
    assert next(check for check in bad.checks if check.invalid).kind == "alias"


def test_import_and_execution_need_no_optional_frame_libraries():
    program = """
import sys
from examples.accounting_breaks import Ledger, accounting_rows
from modello.contrib.validation import WindowValidator
assert not {"pandas", "polars", "numpy"} & sys.modules.keys()
results = WindowValidator(Ledger, "rows").iter_results(accounting_rows())
assert [result.valid for result in results] == [None, True, False, True]
assert not {"pandas", "polars", "numpy"} & sys.modules.keys()
"""
    process = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
