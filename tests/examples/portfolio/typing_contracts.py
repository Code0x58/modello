# mypy: warn-unused-ignores
"""Portfolio integration preserves concrete models, members, and provider results."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    from examples.compute.named_map import NamedMap
    from examples.compute.providers import Provider
    from examples.model_plan import ModelPath, ModelPlan
    from examples.named_members import MemberValues
    from examples.portfolio.account import AccountStep
    from examples.portfolio.basket import EquityHolding, equity
    from examples.portfolio.demo import ResearchRun
    from examples.portfolio.execution import ExecutionBenchmark
    from examples.portfolio.explicit import EquityMembers, Portfolio
    from examples.portfolio.models import Equity, EquityPosition, USDCash
    from examples.portfolio.optimal_execution import (
        Calibration,
        ExecutionDensity,
        ExecutionGrid,
        ExecutionSolution,
        ImpactObservation,
    )
    from examples.portfolio.postings import AccountFill, AccountPosting, AccountState, FillPosting
    from examples.portfolio.providers import EquityProvider
    from examples.portfolio.reporting import AccountReport
    from examples.portfolio.signals import OrderIntent
    from examples.portfolio.sources import PriceHistory
    from examples.portfolio.valuation import MarketSnapshot, NamedMarketData

    def valuation(portfolio: Portfolio, quote: MarketSnapshot, observations: NamedMarketData) -> None:
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

    def execution(
        model: ExecutionDensity, observations: tuple[ImpactObservation, ...]
    ) -> tuple[ExecutionGrid, ExecutionSolution, Calibration]:
        grid = model.discretize(10, (1, 1))
        solution = grid.stationary(at=NamedMap({"gamma": 0.001}))
        return grid, solution, model.fit_impact(observations)
