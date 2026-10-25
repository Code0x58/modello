"""The nested ledger example identifies a locally isolated accounting break."""

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from examples.accounting_breaks import Ledger, accounting_frame, find_breaks
from modello.contrib.pandas import FrameValidator


def test_accounting_break_and_input_preservation():
    frame = accounting_frame()
    original = frame.copy(deep=True)
    breaks = find_breaks(frame)
    assert breaks[["position", "index", "field", "rule", "observed", "expected", "residual"]].values.tolist() == [
        [2, "sale", "cash", "posting", 90_990, 90_998, -8]
    ]
    assert_frame_equal(frame, original)
    validator = FrameValidator(Ledger, "rows")
    report = validator.report(frame)
    assert not report["verdict"].eq("unknown").any()
    final = report.loc[report["position"].eq(3)]
    assert final["verdict"].eq("satisfied").all()
    assert validator.window_size == 2
    assert_series_equal(
        validator.validate(frame),
        pd.Series([pd.NA, True, False, True], index=frame.index, dtype="boolean", name="valid"),
    )
    first = validator.first_invalid(frame)
    assert first is not None
    assert (first.position, first.index, first.field, first.residual) == (2, "sale", "cash", -8)


def test_correcting_the_break_and_subsequent_balances_validates_the_ledger():
    frame = accounting_frame()
    for position in (2, 3):
        for field in ("cash", "nav"):
            frame.loc[frame.index[position], (field, "")] += 8
    assert find_breaks(frame).empty
    assert FrameValidator(Ledger, "rows").first_invalid(frame) is None
