# mypy: warn-unused-ignores
"""Native account cases and member-bound order, execution, and attribution views."""

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from examples.compute.named_map import NamedMap
from examples.named_members import MemberValues
from examples.portfolio.demo import ResearchRun, ResearchScenario
from examples.portfolio.execution import ExecutionBenchmark, ExecutionCosts
from examples.portfolio.explicit import EquityMembers
from examples.portfolio.models import EquityPosition, USDCash
from examples.portfolio.reporting import AccountReport
from examples.portfolio.signals import OrderIntent, TargetWeights


def test_order_and_execution_views_keep_the_decision_case_and_evaluated_models(scenario, market, monkeypatch):
    decision = scenario.decision
    members = decision.portfolio.basket.members
    order = scenario.orders.for_member(members.aapl)
    benchmark = scenario.execution.for_member(members.aapl)
    attribution = scenario.attributions.for_member(members.aapl)
    assert benchmark.order is attribution.order is order
    assert float(members.aapl.quantity) == decision.quantities[order.member] == 10
    assert type(benchmark.costs) is ExecutionCosts
    assert float(benchmark.costs.model_slippage.cost) == pytest.approx(8.6)
    example = ResearchScenario.example()
    monkeypatch.setattr("modello.solve", lambda *args, **kwargs: pytest.fail("Repeated symbolic solve"))
    assessed = order.assess(
        scenario.account.ledger,
        market,
        at=benchmark.assessed_at,
        max_price_age=example.max_price_age,
    )
    assert assessed.implementation_shortfall == benchmark.implementation_shortfall
    assert float(assessed.costs.shortfall.implementation_shortfall) == assessed.implementation_shortfall
    foreign = example.account.opening.basket.members.aapl
    for view in (scenario.orders, scenario.execution, scenario.attributions):
        with pytest.raises(ValueError, match="this model case"):
            view.for_member(foreign)


def test_an_unchanged_allocation_has_an_empty_order_projection(scenario):
    decision = scenario.decision
    targets = TargetWeights(decision.at, decision.weights)
    orders = decision.rebalance(targets, batch="unchanged", arrival_at=decision.at)
    assert len(orders) == 0
    assert len(orders.members) == 2
    with pytest.raises(KeyError):
        orders.for_member(decision.portfolio.basket.members.aapl)


def test_reporting_derives_ledger_ids_and_supports_multiple_orders_per_member(scenario):
    first, second = scenario.attributions
    another = replace(first.order, id="another-decision").attribute(first.weights)
    report = AccountReport.from_account(
        scenario.account,
        (first, second, another),
        signals=NamedMap({"momentum": scenario.signals}),
        utilities=NamedMap({}),
    )
    assert tuple(record.order for record in report.attributions) == (first.order, second.order, another.order)
    assert sum(report.account["net"].values) == pytest.approx(66.9)
    with pytest.raises(ValueError, match="Duplicate member name"):
        AccountReport.from_account(scenario.account, (first, first), signals=NamedMap({}), utilities=NamedMap({}))


if TYPE_CHECKING:

    def typed_research_views() -> None:
        result: ResearchRun[EquityMembers] = ResearchScenario.example().run()
        members: EquityMembers = result.decision.portfolio.basket.members
        position: EquityPosition = members.aapl
        orders: MemberValues[EquityPosition, OrderIntent] = result.orders
        order: OrderIntent = orders.for_member(position)
        benchmark: ExecutionBenchmark = result.execution.for_member(position)
        costs: ExecutionCosts = benchmark.costs
        report: AccountReport[EquityMembers] = result.report()
        reporting_members: EquityMembers = report.path.snapshots[0].portfolio.basket.members
        assert order and costs and reporting_members.aapl
        result.execution.for_member(USDCash("cash"))  # type: ignore[arg-type]
        _wrong: OrderIntent = result.execution.for_member(position)  # type: ignore[assignment]
        result.execution["rebalance-1:aapl"]  # type: ignore[index]
