"""Portable window contracts and strict coverage for downstream consumers."""

import pickle
import subprocess
import sys
from collections import deque
from decimal import Decimal
from itertools import chain

import pytest
from sympy import S

from modello import Index, InstanceDummy, Modello, component, model_recurrence, model_vector
from modello.contrib.validation import ValidationResult, ValidationSummary, WindowValidator
from modello.errors import ModelInputError

from ._models import CashLedger, CashRow


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
        assert all(check.assumed and check.valid is None and not check.invalid for check in result.checks)
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
    problem = next(check for check in expected[2].checks if check.invalid)
    assert (problem.position, problem.expected, problem.residual, problem.start, problem.kind) == (
        2,
        88,
        -3,
        1,
        "recurrence",
    )


def test_validator_is_a_callable_window_function():
    validator = WindowValidator(CashLedger, "rows")
    window = ((100, 0), (90, 10))
    assert isinstance(validator(window, position=1), ValidationResult)
    assert validator(window, position=1).valid is True
    with pytest.raises(TypeError, match="position"):
        validator(window)


def test_callers_can_select_the_first_error_lazily_or_collect_all_errors():
    validator = WindowValidator(CashLedger, "rows")
    rows = ((100, 0), (90, 10), (85, 2), (80, 2))

    def stream():
        for position, row in enumerate(rows):
            if position == 3:
                raise AssertionError("A first-error search should stop before this row")
            yield row

    first = next(check for result in validator.iter_results(stream()) for check in result.checks if check.invalid)
    assert (first.position, first.expected, first.residual) == (2, 88, -3)
    errors = [check for result in validator.iter_results(rows) for check in result.checks if check.invalid]
    assert [(check.position, check.expected, check.residual) for check in errors] == [(2, 88, -3), (3, 83, -3)]


def test_checks_and_rows_distinguish_valid_invalid_assumed_and_unknown():
    results = list(WindowValidator(CashLedger, "rows").iter_results(((100, 0), (90, 10), (85, 2), (None, 2))))
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
    assert summary.invalid and summary.valid is False and not summary.complete


def test_first_invalid_row_retains_all_failing_and_undecidable_checks():
    results = list(WindowValidator(CashLedger, "rows").iter_results(((100, 0), ("bad", "bad"))))
    summary = ValidationSummary.from_results(results)
    assert summary.first_invalid is results[1]
    assert isinstance(summary.first_invalid, ValidationResult)
    assert summary.first_invalid.invalid and not summary.first_invalid.complete
    assert [check.field for check in summary.first_invalid.checks if check.invalid] == ["cash", "debit"]
    assert any(check.valid is None for check in summary.first_invalid.checks)
    assert (summary.valid_rows, summary.invalid_rows, summary.assumed_rows, summary.unknown_rows) == (0, 1, 1, 0)


def test_windows_are_stateless_reentrant_and_keep_observed_predecessors():
    validator = WindowValidator(CashLedger, "rows")
    final = validator(((85, 2), (83, 2)), position=3)
    broken = validator(((90, 10), (85, 2)), position=2)
    other_account = validator(((50, 0), (45, 5)), position=1)
    assert final.valid is True and broken.valid is False and other_account.valid is True
    assert validator(((85, 2), (83, 2)), position=3) == final
    assert next(check for check in broken.checks if check.invalid).expected == 88


def test_position_is_the_absolute_coordinate_not_the_window_length_or_label():
    class TimeDriven(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        rows = model_vector(CashRow, size=N)
        n = Index("n")
        law = model_recurrence(rows[n], cash=rows[n - 1].cash - n)

    validator = WindowValidator(TimeDriven, "rows")
    correct = validator(((97, 0), (94, 0)), position=3, index="event-at-100")
    assert correct.valid is True
    assert correct.index == "event-at-100"
    wrong = validator(((97, 0), (94, 0)), position=1)
    assert wrong.invalid
    assert next(check for check in wrong.checks if check.invalid).expected == 96


@pytest.mark.parametrize("position", [-1, True, 1.5, None, "1"])
def test_invalid_absolute_positions_are_rejected(position):
    with pytest.raises(ModelInputError, match="absolute nonnegative integer"):
        WindowValidator(CashLedger, "rows")(((100, 0),), position=position)


@pytest.mark.parametrize(
    "window,position,required", [([], 0, 1), ([(1, 0)], 10, 2), ([(1, 0)] * 3, 2, 2), ([(1, 0)] * 2, 0, 1)]
)
def test_wrong_window_sizes_are_configuration_errors(window, position, required):
    with pytest.raises(ModelInputError, match=f"requires exactly {required} window rows"):
        WindowValidator(CashLedger, "rows")(window, position=position)


@pytest.mark.parametrize("row", ["bad", 1, (100,), (100, 0, 0)])
def test_wrong_projected_row_shapes_are_rejected(row):
    with pytest.raises(ModelInputError, match="row|fields"):
        WindowValidator(CashLedger, "rows")((row,), position=0)


def test_missing_mapping_keys_are_distinct_from_explicit_nulls():
    validator = WindowValidator(CashLedger, "rows")
    with pytest.raises(ModelInputError, match="Missing row fields: debit"):
        validator(({"cash": 100},), position=0)
    result = validator(({"cash": 100, "debit": 0}, {"cash": None, "debit": 10}), position=1)
    assert result.valid is None and not result.complete and not result.assumed
    assert not result.invalid
    assert any(check.valid is None and not check.assumed for check in result.checks)


def test_labels_are_metadata_and_explicit_none_is_preserved():
    validator = WindowValidator(CashLedger, "rows")
    assert validator(((100, 0), (90, 10)), position=1).index == 1
    assert validator(((100, 0), (90, 10)), position=1, index=None).index is None
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
            actual.append(validator(tuple(window), position=position))
            position += 1
    assert actual == expected
    # A new scan resets position and history; this must be a deliberate partition boundary.
    assert next(validator.iter_results(batches[1])).assumed
    with pytest.raises(ModelInputError, match="exactly 2 window rows"):
        validator(batches[1], position=2)


def test_audit_makes_successful_transitions_and_assumed_history_distinct():
    summary = WindowValidator(CashLedger, "rows").audit(((100, 0), (90, 10), (88, 2)))
    assert (
        summary.row_count,
        summary.valid_rows,
        summary.assumed_rows,
        summary.unknown_rows,
        summary.invalid_rows,
    ) == (3, 2, 1, 0, 0)
    assert summary.first_invalid is None
    assert not summary.complete and summary.valid is None


@pytest.mark.parametrize("rows", [(), ((100, 0),)])
def test_empty_and_history_only_audits_never_claim_complete_success(rows):
    summary = WindowValidator(CashLedger, "rows").audit(rows)
    assert summary.row_count == summary.assumed_rows == len(rows)
    assert summary.valid_rows == summary.unknown_rows == summary.invalid_rows == 0
    assert summary.first_invalid is None
    assert not summary.complete and summary.valid is None


def test_audit_reports_missing_coverage_that_can_resume_after_a_gap():
    rows = ((100, 0), (90, 10), (None, 2), (86, 2), (84, 2))
    summary = WindowValidator(CashLedger, "rows").audit(rows)
    assert (summary.row_count, summary.valid_rows, summary.assumed_rows, summary.unknown_rows) == (5, 2, 1, 2)
    assert summary.first_invalid is None and summary.valid is None


def test_proven_violation_does_not_hide_incomplete_checks_on_the_same_row():
    validator = WindowValidator(CashLedger, "rows")
    result = validator(((100, 0), (None, "bad")), position=1)
    assert result.valid is False and not result.complete
    assert next(check for check in result.checks if check.invalid).field == "debit"
    summary = validator.audit(((100, 0), (None, "bad")))
    assert (summary.valid_rows, summary.assumed_rows, summary.unknown_rows, summary.invalid_rows) == (0, 1, 0, 1)
    assert summary.valid is False and not summary.complete


def test_audit_reads_past_the_first_failure_to_measure_all_coverage():
    summary = WindowValidator(CashLedger, "rows").audit(((100, 0), (85, 10), (None, 2), (70, 2)))
    assert summary.row_count == 4 and summary.unknown_rows == 2
    assert summary.invalid_rows == 1
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
    result = validator(((100, 0), (99, 1)), position=1)
    assert result.valid is None and not result.assumed and not result.complete
    late = next(check for check in result.checks if check.rule == "second")
    assert (late.start, late.kind, late.assumed) == (2, "recurrence", True)
    assert late.valid is None and not late.invalid
    bad = validator(((100, 0), (98, 1)), position=1)
    assert bad.valid is False and not bad.complete
    assert next(check for check in bad.checks if check.invalid).rule == "first"
    summary = validator.audit(((100, 0), (99, 1), (97, 2)))
    assert (summary.valid_rows, summary.assumed_rows, summary.unknown_rows) == (1, 1, 1)


@pytest.mark.parametrize("missing", [None, float("nan"), S.NaN, Decimal("NaN")])
def test_portable_nulls_produce_unknown_checks(missing):
    result = WindowValidator(CashLedger, "rows")(((100, 0), (missing, 0)), position=1)
    assert result.valid is None and not result.invalid


def test_prepared_validator_can_be_serialised_for_a_worker():
    original = WindowValidator(CashLedger, "rows")
    restored = pickle.loads(pickle.dumps(original))
    window = ((Decimal("0.30"), 0), (Decimal("0.20"), Decimal("0.10")))
    assert restored(window, position=1) == original(window, position=1)


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
    result = validator((opening, current), position=1)
    assert result.valid is True
    assert {check.kind for check in result.checks} == {"domain", "definition", "alias", "recurrence"}
    current["spendable"] = 89
    bad = validator((opening, current), position=1)
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
