"""Run an account, signal, and execution-analysis system over external CSV fixtures.

Run ``python -m examples.portfolio.demo --json /tmp/account-report.json``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from examples.compute.named_map import NamedMap
from examples.compute.series import Series
from examples.portfolio.account import Account, AccountEngine, AccountPath, AccountSnapshot
from examples.portfolio.execution import ExecutionBenchmark, benchmark_executions
from examples.portfolio.models import EquityPosition, USDCash
from examples.portfolio.named import Basket, EquityMembers, portfolio_model
from examples.portfolio.performance import Comparison, PerformanceIndex, compare
from examples.portfolio.report_html import render_html, report_data
from examples.portfolio.reporting import AccountReport, OrderAttribution
from examples.portfolio.reporting import report as build_report
from examples.portfolio.signals import (
    MomentumSignals,
    OrderIntent,
    TargetWeights,
    positive_signal_targets,
    rebalance_orders,
)
from examples.portfolio.sources import (
    CSVIndexSource,
    CSVMarketSource,
    CSVTransactionSource,
    MarketSource,
    TransactionSource,
    timestamp,
)
from examples.portfolio.utility import cara, crra

CONTEXT = (
    "Synthetic observations demonstrate accounting and reporting mechanics; "
    "they do not establish investment performance."
)


@dataclass(frozen=True)
class ResearchRun:
    """Inspectable intermediate layers rather than a monolithic backtest result."""

    account: AccountPath
    signals: NamedMap[Series[datetime]]
    targets: TargetWeights
    orders: NamedMap[OrderIntent]
    execution: NamedMap[ExecutionBenchmark]


def run(market_source: MarketSource, transaction_source: TransactionSource) -> ResearchRun:
    """Bind replaceable sources to one deterministic research scenario."""
    opening = portfolio_model(Basket)(
        "opening",
        basket=Basket(
            "basket",
            members=EquityMembers(
                "equities", aapl=EquityPosition("apple", quantity=10), msft=EquityPosition("microsoft", quantity=10)
            ),
        ),
        cash=USDCash("cash", amount=1000),
    )
    times = tuple(timestamp(f"2026-01-{day:02d}T16:01:00Z") for day in range(5, 9))
    account = Account("research", times[0], opening)
    market = market_source.load(opening.instruments)
    ledger = transaction_source.load()
    age = timedelta(days=4)
    engine = AccountEngine(account, market, max_price_age=age)
    decision_at = timestamp("2026-01-07T09:00:00Z")
    decision = engine.evaluate(ledger, (times[0], decision_at)).snapshots[-1]
    feature_prices = market.map(lambda history: history.sample((*times[:2], decision_at), max_age=age))
    signals = MomentumSignals(feature_prices, lookback=2).load()
    targets = positive_signal_targets(signals, decision_at)
    orders = rebalance_orders(decision, targets, batch="rebalance-1", arrival_at=timestamp("2026-01-07T09:05:00Z"))
    execution = benchmark_executions(orders, ledger, market, at=timestamp("2026-01-07T09:15:00Z"), max_price_age=age)
    return ResearchRun(engine.evaluate(ledger, times), signals, targets, orders, execution)


def example_run() -> ResearchRun:
    """Use synthetic, independent market and broker files shipped with the example."""
    directory = Path(__file__).with_name("data")
    return run(
        CSVMarketSource(directory / "market.csv"),
        CSVTransactionSource(directory / "transactions.csv"),
    )


def performance_comparison(path: AccountPath) -> Comparison[datetime]:
    """Compare the toy account with a separate synthetic total-return index."""
    directory = Path(__file__).with_name("data")
    return compare(
        PerformanceIndex("Synthetic account, net of booked fees, including income", path.unit_nav),
        PerformanceIndex(
            "Synthetic broad-market total-return index", CSVIndexSource(directory / "benchmark.csv").load()
        ),
        risk_free=PerformanceIndex(
            "Zero-return USD cash (illustrative assumption)", path.unit_nav.map(lambda _: 100.0)
        ),
        periods_per_year=252,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Write the account and benchmark layers to a JSON report")
    parser.add_argument("--html", type=Path, help="Write account/instrument tables and interactive timeline charts")
    arguments = parser.parse_args()
    result = example_run()
    path = result.account
    comparison = performance_comparison(path)
    reporting = research_report(result)
    print("Synthetic USD account; external market prints and transaction log")
    print("Time                             AAPL shares  MSFT shares      Cash        NAV   Unit NAV")
    for snapshot in path.snapshots:
        print(
            f"{snapshot.at.isoformat()}  {snapshot.quantities['aapl']:11.4f}  {snapshot.quantities['msft']:11.4f}"
            f"  {snapshot.cash:8.2f}  {snapshot.nav:9.2f}  {snapshot.unit_nav:9.4f}"
        )
    print("Targets:", dict(result.targets.weights))
    performance = comparison.portfolio
    print(
        f"Measurement: {performance.start.isoformat()} to {performance.end.isoformat()}; "
        f"{performance.elapsed_days:g} calendar days, {performance.return_count} daily returns"
    )
    print("Time-weighted price-and-income return (net of booked fees):", f"{performance.total_return:.4%}")
    print(f"{comparison.benchmark_name}: {comparison.benchmark.total_return:.4%}")
    print(f"Account minus index: {comparison.return_difference * 100:+.4f} percentage points")
    print(f"Relative wealth return: {comparison.relative_return:+.4%}")
    print(
        f"Observed maximum drawdown: account {performance.maximum_drawdown:.4%}, "
        f"index {comparison.benchmark.maximum_drawdown:.4%}"
    )
    print("CAGR and annualized Sharpe: N/A — history shorter than one year")
    print("Sharpe reference:", comparison.risk_free_name)
    print(CONTEXT)
    print("P&L:", path.pnl.total.values, "External capital:", path.pnl.external_flows.values)
    print("NAV reconciliation residuals:", path.reconciliation.values)
    for order_id, report in result.execution.items():
        print(
            f"{order_id}: filled {report.filled_quantity:.4f}, remaining {report.remaining_quantity:.4f}, "
            f"model slippage USD {report.model_slippage_cost:.4f}, "
            f"implementation shortfall USD {report.implementation_shortfall:.4f}"
        )
    if arguments.json:
        payload = {
            "reporting": report_data(reporting),
            "performance": {
                "portfolio_name": comparison.portfolio_name,
                "benchmark_name": comparison.benchmark_name,
                "risk_free_name": comparison.risk_free_name,
                "currency": comparison.currency,
                "periods_per_year": comparison.periods_per_year,
                "portfolio": asdict(comparison.portfolio),
                "benchmark": asdict(comparison.benchmark),
                "return_difference": comparison.return_difference,
                "relative_return": comparison.relative_return,
                "normalized": {name: asdict(series) for name, series in comparison.normalized.items()},
                "active_returns": asdict(comparison.active_returns),
                "context": CONTEXT,
                "annualization_policy": "CAGR and annualized Sharpe require at least one full calendar year.",
                "cagr_basis": "Elapsed calendar time, ACT/365.25",
                "sharpe_basis": "Mean excess simple return / sample excess-return deviation * sqrt(periods_per_year)",
            },
            "snapshots": [asdict_snapshot(snapshot) for snapshot in path.snapshots],
            "targets": dict(result.targets.weights),
            "signals": {
                name: {at.isoformat(): value for at, value in zip(series.times, series.values, strict=True)}
                for name, series in result.signals.items()
            },
            "execution": {name: asdict(report) for name, report in result.execution.items()},
            "pnl": {
                "profit": path.pnl.total.values,
                "external_flows": path.pnl.external_flows.values,
                "residuals": path.reconciliation.values,
            },
        }
        arguments.json.write_text(json.dumps(payload, default=encode_value, indent=2) + "\n")
        print("Report:", arguments.json)
    if arguments.html:
        arguments.html.write_text(render_html(reporting, comparison, context=CONTEXT), encoding="utf-8")
        print("Dashboard:", arguments.html)


def research_report(result: ResearchRun) -> AccountReport:
    """Declare the scenario's attribution policy and illustrative wealth preferences."""
    return build_report(
        result.account,
        result.orders.map(lambda order: OrderAttribution(order, NamedMap({"momentum": 1.0}))),
        signals=NamedMap({"momentum": result.signals}),
        utilities=NamedMap({"cara": cara(0.001), "crra": crra(2)}),
    )


def asdict_snapshot(snapshot: AccountSnapshot) -> dict[str, object]:
    """Serialize named maps without copying mapping proxies or model instances."""
    return {
        "at": snapshot.at.isoformat(),
        "known_at": snapshot.known_at.isoformat(),
        "account": snapshot.account,
        "quantities": dict(snapshot.quantities),
        "prices": dict(snapshot.prices),
        "values": dict(snapshot.values),
        "cash": snapshot.cash,
        "nav": snapshot.nav,
        "unit_nav": snapshot.unit_nav,
    }


def encode_value(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Cannot encode {type(value).__name__}")


if __name__ == "__main__":
    main()
