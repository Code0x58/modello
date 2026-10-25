"""The nested ledger example identifies a locally isolated accounting break."""

from pandas.testing import assert_frame_equal

from examples.accounting_breaks import Ledger, accounting_frame, find_breaks
from modello.contrib.pandas import FrameValidator


def test_accounting_break_and_input_preservation():
    frame = accounting_frame()
    original = frame.copy(deep=True)
    report = find_breaks(frame)
    assert not report.valid
    assert report.unknown.empty
    assert report.breaks[
        ["position", "index", "field", "rule", "observed", "expected", "residual"]
    ].values.tolist() == [[2, "sale", "cash", "posting", 90_990, 90_998, -8]]
    assert_frame_equal(frame, original)
    final = report.checks.loc[report.checks["position"].eq(3)]
    assert final["verdict"].eq("satisfied").all()
    assert FrameValidator(Ledger, "rows").window_size == 2


def test_correcting_the_break_and_subsequent_balances_validates_the_ledger():
    frame = accounting_frame()
    for position in (2, 3):
        for field in ("cash", "nav"):
            frame.loc[frame.index[position], (field, "")] += 8
    assert find_breaks(frame).valid
