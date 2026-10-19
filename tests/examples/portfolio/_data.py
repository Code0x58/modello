"""Packaged synthetic data and observation times shared by portfolio tests."""

from datetime import timedelta
from pathlib import Path

import examples.portfolio
from examples.portfolio.sources import timestamp

DATA = Path(examples.portfolio.__file__).with_name("data")
AGE = timedelta(days=4)
TIMES = tuple(timestamp(f"2026-01-{day:02d}T16:01:00Z") for day in range(5, 9))
DECISION = timestamp("2026-01-07T09:00:00Z")
ARRIVAL = timestamp("2026-01-07T09:05:00Z")
ASSESSMENT = timestamp("2026-01-07T09:15:00Z")
