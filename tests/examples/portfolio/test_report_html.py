"""HTML/JSON reporting, escaping, and presentation alignment."""

import json
from dataclasses import replace
from datetime import timedelta
from html.parser import HTMLParser

import pytest

from examples.compute.named_map import NamedMap
from examples.compute.series import Series
from examples.portfolio.account import Account, AccountEngine
from examples.portfolio.basket import basket_model
from examples.portfolio.demo import performance_comparison
from examples.portfolio.models import USDCash
from examples.portfolio.named import portfolio_model
from examples.portfolio.performance import PerformanceIndex, compare
from examples.portfolio.report_html import render_html, report_data
from examples.portfolio.reporting import report
from examples.portfolio.sources import Income, TransactionLog, timestamp
from modello import Modello


def test_report_html_and_json_preserve_values_and_escape_user_labels(reporting):
    comparison = performance_comparison(reporting.path)
    document = render_html(reporting, comparison)
    parser = HTMLParser()
    parser.feed(document)
    assert "55.00" in document and "11.90" in document
    assert "90.90" in document and "-24.00" in document
    assert "3 returns over 3 calendar days" in document
    assert "Absolute (1/USD)" in document
    assert 'data-instrument="1" hidden' in document
    assert "unavailable below one full calendar year" in document
    assert "cdn" not in document and "<script src=" not in document
    bad = replace(reporting.utilities["cara"].definition, name='<script>alert("x")</script>')
    changed = replace(reporting, utilities=NamedMap({"bad": replace(reporting.utilities["cara"], definition=bad)}))
    escaped = render_html(changed, comparison)
    assert '<script>alert("x")</script>' not in escaped
    assert "&lt;script&gt;" in escaped
    payload = json.loads(json.dumps(report_data(reporting)))
    assert payload["sources"]["signal:momentum"]["net"]["values"] == pytest.approx([0, 11.4, 0.5])
    assert payload["orders"]["rebalance-1:aapl"]["signal_weights"] == {"momentum": 1.0}
    wrong = replace(
        comparison,
        normalized=NamedMap({"portfolio": reporting.path.nav, "benchmark": comparison.normalized["benchmark"]}),
    )
    with pytest.raises(ValueError, match="unit-NAV path"):
        render_html(reporting, wrong)


def test_utility_table_aligns_values_by_name(reporting):
    reordered = replace(
        reporting,
        utilities=reporting.utilities.map(
            lambda utility: replace(utility, values=NamedMap(reversed(tuple(utility.values.items()))))
        ),
    )
    comparison = performance_comparison(reporting.path)
    assert render_html(reordered, comparison) == render_html(reporting, comparison)


def test_report_supports_longer_histories_and_supplied_benchmark_context():
    class CashMembers(Modello):
        pass

    times = tuple(timestamp(f"2024-{month:02d}-01T16:00:00Z") for month in (1, 4, 7, 10)) + (
        timestamp("2025-01-01T16:00:00Z"),
    )
    name = "Long <account> & research"
    opening = portfolio_model(basket_model(CashMembers, NamedMap({})))("opening", cash=USDCash("cash", amount=100))
    engine = AccountEngine(Account(name, times[0], opening), NamedMap({}), max_price_age=timedelta(0))
    ledger = TransactionLog(tuple(Income(str(i), name, at, at, 10) for i, at in enumerate(times[1:])))
    path = engine.evaluate(ledger, times)
    reporting = report(path, NamedMap({}), signals=NamedMap({}), utilities=NamedMap({}))
    comparison = compare(
        PerformanceIndex("Account after costs", path.unit_nav),
        PerformanceIndex("Reference <index> & income", Series(times, (100, 103, 106, 109, 112))),
        risk_free=PerformanceIndex("Cash <reference>", Series(times, (100,) * 5)),
        periods_per_year=4,
    )
    document = render_html(reporting, comparison, context="External data <reviewed> & fees")
    assert "4 returns over 366 calendar days" in document
    assert "2024-01-01 16:00" in document and "2025-01-01 16:00" in document
    assert "Long &lt;account&gt; &amp; research" in document
    assert "Account index: Account after costs" in document
    assert "Benchmark: Reference &lt;index&gt; &amp; income" in document
    assert "Sharpe reference: Cash &lt;reference&gt;" in document
    assert "External data &lt;reviewed&gt; &amp; fees" in document
    assert "Synthetic" not in document and "unavailable below" not in document
    assert "Annualized performance" in document
    assert f"{1.4 ** (365.25 / 366) - 1:.4%}" in document
    assert comparison.portfolio.sharpe_ratio is not None
    assert f"{comparison.portfolio.sharpe_ratio:.4f}" in document
    assert "External data" not in render_html(reporting, comparison)
