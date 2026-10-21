"""External market histories, transaction logs, and configured CSV requests."""

from __future__ import annotations

import csv
from abc import abstractmethod
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import isfinite
from pathlib import Path

from examples.data import Provider, Series, utc_time
from examples.portfolio.models import Equity


def timestamp(value: str) -> datetime:
    """Read an ISO timestamp with an explicit offset, including Z on Python 3.10."""
    return utc_time(datetime.fromisoformat(value.replace("Z", "+00:00")))


def finite(*values: float) -> None:
    """Validate numerical observations at the boundary."""
    if not all(isfinite(value) for value in values):
        raise ValueError("Amounts, prices, quantities, and volumes must be finite")


@dataclass(frozen=True)
class PriceHistory:
    """Unadjusted trade prints with separate economic and publication times."""

    equity: Equity
    prices: Series[datetime]
    available_at: tuple[datetime, ...]
    volumes: tuple[float, ...]

    def __post_init__(self) -> None:
        available = tuple(utc_time(at) for at in self.available_at)
        volumes = tuple(float(value) for value in self.volumes)
        if len(available) != len(self.prices.times) or len(volumes) != len(available):
            raise ValueError("Every market print needs a publication time and volume")
        if any(known < observed for known, observed in zip(available, self.prices.times, strict=True)):
            raise ValueError("A market print cannot be available before it occurred")
        if any(left >= right for left, right in zip(available, available[1:])):
            raise ValueError("Normalize prints to strictly increasing publication times")
        finite(*volumes)
        if any(price <= 0 for price in self.prices.values) or any(volume <= 0 for volume in volumes):
            raise ValueError("Trade prices and volumes must be strictly positive")
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "volumes", volumes)

    @property
    def known_prices(self) -> Series[datetime]:
        """Expose the price vector on the axis when observations became available."""
        return Series(self.available_at, self.prices.values)

    def mark(self, at: datetime, *, max_age: timedelta) -> float:
        """Use only published observations and measure age from their economic time."""
        at = utc_time(at)
        index = bisect_right(self.available_at, at) - 1
        if index < 0:
            raise ValueError(f"No {self.equity.symbol} price is available at {at}")
        if max_age < timedelta(0) or at - self.prices.times[index] > max_age:
            raise ValueError(f"{self.equity.symbol} price at {at} is stale")
        return self.prices.values[index]

    def sample(self, times: tuple[datetime, ...], *, max_age: timedelta) -> Series[datetime]:
        """Bind an explicit valuation or feature schedule to causal observations."""
        return Series(times, tuple(self.mark(at, max_age=max_age) for at in times))

    def vwap(self, start: datetime, end: datetime) -> float | None:
        """Volume-weight independent prints in [start, end], known by end."""
        rows = tuple(
            (price, volume)
            for at, available, price, volume in zip(
                self.prices.times, self.available_at, self.prices.values, self.volumes, strict=True
            )
            if start <= at <= end and available <= end
        )
        volume = sum(volume for _, volume in rows)
        return sum(price * size for price, size in rows) / volume if volume else None


@dataclass(frozen=True)
class Posting:
    """A unique external ledger event and its reporting timestamp."""

    id: str
    account: str
    at: datetime
    recorded_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", utc_time(self.at))
        object.__setattr__(self, "recorded_at", utc_time(self.recorded_at))
        if not self.id or not self.account or self.recorded_at < self.at:
            raise ValueError("Postings need identities and a reporting time at or after execution")


@dataclass(frozen=True)
class Fill(Posting):
    """Signed shares (positive buy), a price, and a nonnegative execution fee."""

    member: str
    quantity: float
    price: float
    fee: float = 0.0
    order_id: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        finite(self.quantity, self.price, self.fee)
        if not self.member or not self.quantity or self.price <= 0 or self.fee < 0:
            raise ValueError("Fills need a member, nonzero shares, a positive price, and nonnegative fees")


@dataclass(frozen=True)
class CashFlow(Posting):
    """An external deposit or withdrawal, separated from investment income."""

    amount: float

    def __post_init__(self) -> None:
        super().__post_init__()
        finite(self.amount)


@dataclass(frozen=True)
class Income(Posting):
    """Booked dividend, interest, or expense cash; amounts may include withholding."""

    amount: float
    member: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        finite(self.amount)


LedgerEvent = Fill | CashFlow | Income


@dataclass(frozen=True)
class ExecutionSeries:
    """Executions aggregated by timestamp, retaining signed shares and gross volume."""

    quantity: Series[datetime]
    volume: Series[datetime]
    notional: Series[datetime]
    fees: Series[datetime]

    @property
    def vwap(self) -> Series[datetime]:
        return self.notional.combine(self.volume, lambda value, volume: value / volume)


@dataclass(frozen=True)
class TransactionLog:
    """Immutable postings; simultaneous events retain source order."""

    events: tuple[LedgerEvent, ...]

    def __post_init__(self) -> None:
        events = tuple(sorted(self.events, key=lambda event: event.at))
        if len({(event.account, event.id) for event in events}) != len(events):
            raise ValueError("Duplicate transaction ID within an account")
        object.__setattr__(self, "events", events)

    def known(self, account: str, at: datetime) -> TransactionLog:
        """Take a reporting snapshot for one account, excluding later publications."""
        at = utc_time(at)
        return TransactionLog(
            tuple(event for event in self.events if event.account == account and event.recorded_at <= at)
        )

    def series(self, project: Callable[[LedgerEvent], float]) -> Series[datetime]:
        """Aggregate an event projection at each economic timestamp."""
        values: dict[datetime, float] = {}
        for event in self.events:
            values[event.at] = values.get(event.at, 0.0) + project(event)
        return Series(tuple(values), tuple(values.values()))

    def executions(self, member: str) -> ExecutionSeries:
        """View a member's fills as vectors without losing the underlying event log."""
        fills = TransactionLog(
            tuple(event for event in self.events if isinstance(event, Fill) and event.member == member)
        )

        def projected(operation: Callable[[Fill], float]) -> Series[datetime]:
            return fills.series(lambda event: operation(event) if isinstance(event, Fill) else 0.0)

        return ExecutionSeries(
            projected(lambda fill: fill.quantity),
            projected(lambda fill: abs(fill.quantity)),
            projected(lambda fill: abs(fill.quantity) * fill.price),
            projected(lambda fill: fill.fee),
        )


def csv_rows(path: Path, fields: set[str]) -> list[dict[str, str]]:
    """Read a declared feed schema and report malformed rows at the boundary."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if not fields <= set(header) or len(set(header)) != len(header):
            raise ValueError(f"{path.name} needs unique columns {sorted(fields)}")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"{path.name} has a malformed row")
        return rows


@dataclass(frozen=True)
class CSVIndexSource(Provider[Series[datetime]]):
    """Load a total-return index already aligned to a reporting schedule."""

    path: Path

    def load(self) -> Series[datetime]:
        rows = csv_rows(self.path, {"at", "level"})
        return Series(tuple(timestamp(row["at"]) for row in rows), tuple(float(row["level"]) for row in rows))


@dataclass(frozen=True)
class CSVTransactionSource(Provider[TransactionLog]):
    """Adapt a separate account ledger without coupling it to price acquisition."""

    path: Path

    def load(self) -> TransactionLog:
        fields = {
            "id",
            "account",
            "at",
            "recorded_at",
            "kind",
            "member",
            "quantity",
            "price",
            "fee",
            "amount",
            "order_id",
        }
        events: list[LedgerEvent] = []
        for row in csv_rows(self.path, fields):
            common = (row["id"], row["account"], timestamp(row["at"]), timestamp(row["recorded_at"]))
            if row["kind"] == "fill":
                events.append(
                    Fill(
                        *common,
                        row["member"],
                        float(row["quantity"]),
                        float(row["price"]),
                        float(row["fee"]),
                        row["order_id"] or None,
                    )
                )
            elif row["kind"] == "flow":
                events.append(CashFlow(*common, float(row["amount"])))
            elif row["kind"] == "income":
                events.append(Income(*common, float(row["amount"]), row["member"] or None))
            else:
                raise ValueError(f"Unknown transaction kind {row['kind']!r}")
        return TransactionLog(tuple(events))


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
