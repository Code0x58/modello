# mypy: warn-unused-ignores
"""Portfolio integration preserves concrete models, members, and provider results."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date, datetime

    from examples.data import NamedMap, Provider
    from examples.named_members import MemberValues
    from examples.portfolio.account import AccountStep
    from examples.portfolio.demo import EquityMembers, Portfolio, ResearchRun
    from examples.portfolio.execution import ExecutionBenchmark
    from examples.portfolio.models import (
        AccountFill,
        AccountPosting,
        AccountState,
        Equity,
        EquityPosition,
        FillPosting,
        USDCash,
    )
    from examples.portfolio.optimal_execution import (
        Calibration,
        ExecutionDensity,
        ExecutionGrid,
        ExecutionSolution,
        ImpactObservation,
    )
    from examples.portfolio.reporting import AccountReport
    from examples.portfolio.signals import OrderIntent
    from examples.portfolio.sources import EquityProvider, PriceHistory
    from examples.portfolio.valuation import EquityHolding, MarketSnapshot, NamedMarketData, equity
    from examples.symbolic import ModelPath, ModelPlan

    def valuation(portfolio: Portfolio, quote: MarketSnapshot, observations: NamedMarketData[date]) -> None:
        view: ModelPlan[Portfolio] = portfolio.pricer()
        point: Portfolio = view.evaluate(quote)
        path: ModelPath[Portfolio, date] = view.over(observations)
        point = path.at(0)
        member: EquityPosition = point.basket.members.aapl
        holding: EquityHolding = point.basket.holding(member)
        provider: EquityProvider = holding.provider
        generic: Provider[PriceHistory] = provider
        _history: PriceHistory = generic.load()
        _identity: Equity = provider.instrument
        _declared: EquityPosition = equity.component(provider)
        point.basket.members.aapl = USDCash("cash")  # type: ignore[assignment]
        equity.component(equity)  # type: ignore[arg-type]
        _wrong_holding: USDCash = equity.component(provider)  # type: ignore[assignment]
        _wrong_provider: Provider[Equity] = provider  # type: ignore[assignment]
        _wrong_result: EquityPosition = view.evaluate(quote)  # type: ignore[assignment]
        view.evaluate(NamedMap({}))  # type: ignore[arg-type]
        point.basket.holding(USDCash("cash"))  # type: ignore[arg-type]

    def research(result: ResearchRun[EquityMembers]) -> None:
        step: AccountStep[EquityMembers] = result.account.transitions[0]
        posting: AccountPosting = step.posting
        _state: AccountState = posting
        _holding: EquityPosition = step.after.portfolio.basket.members.aapl
        if isinstance(posting, AccountFill):
            _settlement: FillPosting = posting.settlement
        members: EquityMembers = result.decision.portfolio.basket.members
        orders: MemberValues[EquityPosition, OrderIntent] = result.orders
        _order: OrderIntent = orders.for_member(members.aapl)
        _benchmark: ExecutionBenchmark = result.execution.for_member(members.aapl)
        report: AccountReport[EquityMembers] = result.report()
        _reported: EquityPosition = report.path.snapshots[0].portfolio.basket.members.aapl
        result.execution.for_member(USDCash("cash"))  # type: ignore[arg-type]
        _wrong: OrderIntent = result.execution.for_member(members.aapl)  # type: ignore[assignment]
        result.execution["rebalance-1:aapl"]  # type: ignore[index]

    def timestamp_prices(portfolio: Portfolio, observations: NamedMarketData[datetime]) -> None:
        path: ModelPath[Portfolio, datetime] = portfolio.pricer().over(observations)
        _wrong_clock: ModelPath[Portfolio, date] = path  # type: ignore[assignment]

    def execution(
        model: ExecutionDensity, observations: tuple[ImpactObservation, ...]
    ) -> tuple[ExecutionGrid, ExecutionSolution, Calibration]:
        grid = model.discretize(10, (1, 1))
        solution = grid.stationary(at=NamedMap({"gamma": 0.001}))
        return grid, solution, model.fit_impact(observations)
