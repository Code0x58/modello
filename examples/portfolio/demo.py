"""Run an account, signal, and execution-analysis system over external CSV fixtures.

Run ``python -m examples.portfolio.demo --json /tmp/account-report.json``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generic, TypeVar

from examples.compute.named_map import NamedMap
from examples.compute.providers import Provider
from examples.compute.series import Series
from examples.portfolio.account import Account, AccountPath, AccountSnapshot
from examples.portfolio.basket import EquityHoldings, equity
from examples.portfolio.execution import ExecutionBenchmark
from examples.portfolio.explicit import Basket, EquityMembers, Portfolio
from examples.portfolio.models import USDCash
from examples.portfolio.performance import Comparison, PerformanceIndex
from examples.portfolio.reporting import AccountReport, OrderAttribution
from examples.portfolio.signals import (
    MomentumSignals,
    OrderIntent,
    TargetWeights,
)
from examples.portfolio.sources import (
    CSVIndexSource,
    CSVTransactionSource,
    TransactionLog,
    timestamp,
)
from examples.portfolio.utility import CashUtility

_Members = TypeVar("_Members", bound=EquityHoldings)

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

    attributions: NamedMap[OrderAttribution]
    utilities: NamedMap[CashUtility]
    comparison: Comparison[datetime]

    def report(self) -> AccountReport:
        """Evaluate the reporting layer using the decision-time attribution declaration."""
        return self.account.report(
            self.attributions, signals=NamedMap({"momentum": self.signals}), utilities=self.utilities
        )


@dataclass(frozen=True)
class ResearchScenario(Generic[_Members]):
    """Declare data, timing, and policy separately from their evaluation."""

    account: Account[_Members]
    transactions: Provider[TransactionLog]
    valuation_times: tuple[datetime, ...]
    feature_times: tuple[datetime, ...]
    decision_at: datetime
    arrival_at: datetime
    assessment_at: datetime
    max_price_age: timedelta
    benchmark: PerformanceIndex[datetime]
    cash_reference: PerformanceIndex[datetime]
    utilities: NamedMap[CashUtility]
    lookback: int = 2
    exposure: float = 0.8
    periods_per_year: int = 252
    batch: str = "rebalance-1"
    performance_name: str = "Account, net of booked fees, including income"

    def run(self) -> ResearchRun:
        market = self.account.opening.load_market()
        ledger = self.transactions.load()
        decision = self.account.replay(
            ledger,
            (self.account.opened_at, self.decision_at),
            market=market,
            max_price_age=self.max_price_age,
        ).snapshots[-1]
        prices = market.map(lambda history: history.sample(self.feature_times, max_age=self.max_price_age))
        signals = MomentumSignals(prices, lookback=self.lookback).load()
        targets = TargetWeights.from_signals(signals, self.decision_at, exposure=self.exposure)
        orders = decision.rebalance(targets, batch=self.batch, arrival_at=self.arrival_at)
        attributions = orders.map(lambda order: OrderAttribution(order, NamedMap({"momentum": 1.0})))
        execution = ExecutionBenchmark.assess(
            orders, ledger, market, at=self.assessment_at, max_price_age=self.max_price_age
        )
        path = self.account.replay(ledger, self.valuation_times, market=market, max_price_age=self.max_price_age)
        comparison = PerformanceIndex(self.performance_name, path.unit_nav).compare(
            self.benchmark, risk_free=self.cash_reference, periods_per_year=self.periods_per_year
        )
        return ResearchRun(path, signals, targets, orders, execution, attributions, self.utilities, comparison)

    @staticmethod
    def example() -> ResearchScenario[EquityMembers]:
        """A complete, offline declaration of the synthetic research case."""
        directory = Path(__file__).with_name("data")
        times = tuple(timestamp(f"2026-01-{day:02d}T16:01:00Z") for day in range(5, 9))
        decision_at = timestamp("2026-01-07T09:00:00Z")
        opening = Portfolio(
            "opening",
            basket=Basket(
                "basket",
                members=EquityMembers(
                    "equities", aapl=equity("apple", quantity=10), msft=equity("microsoft", quantity=10)
                ),
            ),
            cash=USDCash("cash", amount=1000),
        )
        return ResearchScenario(
            account=Account("research", times[0], opening),
            transactions=CSVTransactionSource(directory / "transactions.csv"),
            valuation_times=times,
            feature_times=(*times[:2], decision_at),
            decision_at=decision_at,
            arrival_at=timestamp("2026-01-07T09:05:00Z"),
            assessment_at=timestamp("2026-01-07T09:15:00Z"),
            max_price_age=timedelta(days=4),
            benchmark=PerformanceIndex(
                "Synthetic broad-market total-return index", CSVIndexSource(directory / "benchmark.csv").load()
            ),
            cash_reference=PerformanceIndex(
                "Zero-return USD cash (illustrative assumption)", Series(times, (100.0,) * len(times))
            ),
            utilities=NamedMap({"cara": CashUtility.cara(0.001), "crra": CashUtility.crra(2)}),
            performance_name="Synthetic account, net of booked fees, including income",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Write the account and benchmark layers to a JSON report")
    parser.add_argument("--html", type=Path, help="Write account/instrument tables and interactive timeline charts")
    arguments = parser.parse_args()
    result = ResearchScenario.example().run()
    path = result.account
    comparison = result.comparison
    reporting = result.report()
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
            "reporting": reporting.to_dict(),
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
        arguments.html.write_text(reporting.to_html(comparison, context=CONTEXT), encoding="utf-8")
        print("Dashboard:", arguments.html)


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
