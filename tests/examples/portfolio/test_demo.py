"""Runnable account research and its CLI reports."""

import json

import pytest

from examples.portfolio.demo import ResearchScenario, main


def test_demo_reports_short_history_and_independent_synthetic_benchmark(tmp_path, monkeypatch, capsys):
    comparison = ResearchScenario.example().run().comparison
    assert comparison.portfolio.return_count == 3
    assert comparison.portfolio.elapsed_days == 3
    assert comparison.portfolio.total_return == pytest.approx((2532.9 / 2500) * (3066.9 / 3032.9) - 1)
    assert comparison.benchmark.total_return == pytest.approx(0.03)
    assert comparison.return_difference == pytest.approx((2532.9 / 2500) * (3066.9 / 3032.9) - 1 - 0.03)
    assert comparison.portfolio.cagr is comparison.portfolio.sharpe_ratio is None
    assert comparison.benchmark.cagr is comparison.benchmark.sharpe_ratio is None
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
    assert tuple(result.orders) == ("alternative:aapl", "alternative:msft")
    assert result.comparison.portfolio_name == "Other account"
    assert result.account.nav.values[-1] == pytest.approx(3066.9)
    assert all(benchmark.filled_quantity == 0 for benchmark in result.execution.values())
    assert sum(result.report().sources["unattributed"]["net"].values) == pytest.approx(11.9)
