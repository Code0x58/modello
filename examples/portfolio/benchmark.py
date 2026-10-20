"""Run ``python -m examples.portfolio.benchmark`` for exploratory local timings."""

from collections.abc import Callable
from datetime import date, timedelta
from math import isclose
from statistics import median
from time import perf_counter
from typing import TypeVar

from examples.compute.series import Series
from examples.portfolio.analytics import NamedPortfolioPath
from examples.portfolio.explicit import MarketData, Portfolio
from examples.portfolio.models import AAPL, MSFT, EquityPrices

_Result = TypeVar("_Result")


class QuotedPortfolio(Portfolio):
    """Expose quote aliases so the baseline needs only one joint solve per row."""

    aapl_price = Portfolio.basket.members.aapl.instrument.price
    msft_price = Portfolio.basket.members.msft.instrument.price


def measure(call: Callable[[], _Result], samples: int = 5) -> tuple[float, _Result]:
    """Report median milliseconds after one untimed warmup."""
    result = call()
    elapsed = []
    for _ in range(samples):
        start = perf_counter()
        result = call()
        elapsed.append((perf_counter() - start) * 1000)
    return median(elapsed), result


def observations(count: int) -> MarketData:
    """A synthetic grid for timing valuation, without annualizing its returns."""
    dates = tuple(date(2000, 1, 1) + timedelta(days=index) for index in range(count))
    return MarketData(
        EquityPrices(AAPL, Series(dates, tuple(100 + index / 1000 for index in range(count)))),
        EquityPrices(MSFT, Series(dates, tuple(50 + index / 2000 for index in range(count)))),
    )


def main() -> None:
    """Separate model construction, compilation, and path evaluation costs."""
    model_ms, model = measure(Portfolio.example)
    compile_ms, pricer = measure(lambda: model.pricer())
    print(f"Symbolic model construction: {model_ms:.3f} ms")
    print(f"Numerical compilation: {compile_ms:.3f} ms")
    for count in (10, 100, 10000):
        data = observations(count)
        elapsed, path = measure(lambda: NamedPortfolioPath.from_model(pricer.over(data.named())))
        assert isclose(path.nav.values[-1], 2100 + 0.02 * (count - 1))
        print(f"{count} observations, compiled complete paths: {elapsed:.3f} ms")

    sample = observations(10)

    def solve_points() -> tuple[float, ...]:
        return tuple(
            float(
                QuotedPortfolio(
                    "point",
                    basket=model.basket,
                    cash=model.cash,
                    aapl_price=apple,
                    msft_price=microsoft,
                ).nav
            )
            for apple, microsoft in zip(sample.aapl.close.values, sample.msft.close.values, strict=True)
        )

    baseline_ms, baseline = measure(solve_points, samples=3)
    path = NamedPortfolioPath.from_model(pricer.over(sample.named()))
    assert all(
        isclose(actual, expected, abs_tol=1e-9) for actual, expected in zip(path.nav.values, baseline, strict=True)
    )
    print(f"10 observations, one joint model solve per row: {baseline_ms:.3f} ms")


if __name__ == "__main__":
    main()
