"""HTML and JSON reports and the research command-line interface."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta

import pytest

from examples.data import NamedMap, Series
from examples.portfolio.account import Account, AccountEngine
from examples.portfolio.analytics import PerformanceIndex
from examples.portfolio.demo import ResearchScenario, main
from examples.portfolio.models import USDCash
from examples.portfolio.report_html import render_html, report_data
from examples.portfolio.reporting import AccountReport
from examples.portfolio.sources import Income, TransactionLog, timestamp
from examples.portfolio.valuation import EquityBasket, EquityHoldings, NamedPortfolio


def test_report_html_and_json_preserve_values_and_escape_user_labels(reporting, scenario):
    comparison = scenario.comparison
    document = render_html(reporting, comparison)
    assert "3 returns over 3 calendar days" in document
    assert "Absolute (1/USD)" in document
    assert "unavailable below one full calendar year" in document
    assert "<script src=" not in document
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


def test_utility_table_aligns_values_by_name(reporting, scenario):
    reordered = replace(
        reporting,
        utilities=reporting.utilities.map(
            lambda utility: replace(utility, values=NamedMap(reversed(tuple(utility.values.items()))))
        ),
    )
    comparison = scenario.comparison
    assert render_html(reordered, comparison) == render_html(reporting, comparison)


def test_report_supports_longer_histories_and_supplied_benchmark_context():
    class CashMembers(EquityHoldings):
        pass

    times = tuple(timestamp(f"2024-{month:02d}-01T16:00:00Z") for month in (1, 4, 7, 10)) + (
        timestamp("2025-01-01T16:00:00Z"),
    )
    name = "Long <account> & research"
    opening = NamedPortfolio.define(EquityBasket.define(CashMembers))("opening", cash=USDCash("cash", amount=100))
    engine = AccountEngine(Account(name, times[0], opening), NamedMap({}), max_price_age=timedelta(0))
    ledger = TransactionLog(tuple(Income(str(i), name, at, at, 10) for i, at in enumerate(times[1:])))
    path = engine.evaluate(ledger, times)
    reporting = AccountReport.from_account(path, (), signals=NamedMap({}), utilities=NamedMap({}))
    comparison = PerformanceIndex("Account after costs", path.unit_nav).compare(
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


def test_demo_reports_short_history_and_independent_synthetic_benchmark(tmp_path, monkeypatch, capsys):
    report = tmp_path / "report.json"
    html = tmp_path / "report.html"
    monkeypatch.setattr("sys.argv", ["account-demo", "--json", str(report), "--html", str(html)])
    main()
    output = capsys.readouterr().out
    assert "3 calendar days, 3 daily returns" in output
    assert "Synthetic broad-market total-return index: 3.0000%" in output
    assert "CAGR and annualized Sharpe: N/A" in output
    payload = json.loads(report.read_text())["performance"]
    assert payload["portfolio"]["cagr"] is payload["portfolio"]["sharpe_ratio"] is None
    assert payload["periods_per_year"] == 252
    assert payload["normalized"]["benchmark"]["values"] == [100, 100.5, 102, 103]
    assert "Synthetic" in payload["context"]
    assert payload["context"] in html.read_text()


def test_a_scenario_declares_policy_and_labels_without_inventing_new_fills():
    from dataclasses import replace

    scenario = replace(ResearchScenario.example(), exposure=0.5, batch="alternative", performance_name="Other account")
    result = scenario.run()
    assert result.targets.weights == {"aapl": 0.5, "msft": 0}
    assert tuple(order.id for order in result.orders) == ("alternative:aapl", "alternative:msft")
    assert result.comparison.portfolio_name == "Other account"
    assert result.account.nav.values[-1] == pytest.approx(3066.9)
    assert all(benchmark.filled_quantity == 0 for benchmark in result.execution)
    assert sum(result.report().sources["unattributed"]["net"].values) == pytest.approx(11.9)
