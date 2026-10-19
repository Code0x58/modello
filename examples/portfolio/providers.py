"""Configured equity-data requests with typed market-history results."""

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from examples.compute.providers import Provider
from examples.compute.series import Series
from examples.portfolio.models import Equity
from examples.portfolio.sources import PriceHistory, csv_rows, timestamp


class EquityProvider(Provider[PriceHistory]):
    """Request price history for one instrument with an identity known before I/O."""

    @property
    @abstractmethod
    def instrument(self) -> Equity:
        """Identify the instrument whose history ``load`` returns."""


@dataclass(frozen=True)
class CSVEquityProvider(EquityProvider):
    """Read one provider symbol from an external CSV of independent market prints."""

    symbol: str
    path: Path
    namespace: str
    currency: str = "USD"

    def __post_init__(self) -> None:
        # Validate identity at definition time, leaving acquisition explicit.
        _ = self.instrument
        object.__setattr__(self, "path", Path(self.path))

    @property
    def instrument(self) -> Equity:
        return Equity(self.symbol, provider=self.namespace, currency=self.currency)

    def load(self) -> PriceHistory:
        """Select a symbol and check its currency before constructing its history."""
        rows = csv_rows(self.path, {"provider", "symbol", "currency", "at", "available_at", "price", "volume"})
        points = sorted(
            (row for row in rows if (row["provider"], row["symbol"]) == (self.namespace, self.symbol)),
            key=lambda row: timestamp(row["at"]),
        )
        if not points:
            raise ValueError(f"No market data for {self.namespace}:{self.symbol}")
        if any(row["currency"] != self.currency for row in points):
            raise ValueError(f"Market currency for {self.namespace}:{self.symbol} must be {self.currency}")
        return PriceHistory(
            self.instrument,
            Series(tuple(timestamp(row["at"]) for row in points), tuple(float(row["price"]) for row in points)),
            tuple(timestamp(row["available_at"]) for row in points),
            tuple(float(row["volume"]) for row in points),
        )


@dataclass(frozen=True)
class DummyEquityProvider(CSVEquityProvider):
    """Serve the bundled synthetic equity prints through the same provider contract."""

    path: Path = field(init=False, default=Path(__file__).with_name("data") / "market.csv")
    namespace: str = field(init=False, default="synthetic")
