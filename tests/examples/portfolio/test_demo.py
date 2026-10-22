"""The portfolio demo applies declared policies and reports its own benchmark context."""

from __future__ import annotations

import json

import pytest

from examples.portfolio.demo import ResearchScenario, main


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
