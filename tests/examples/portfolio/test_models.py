"""Instrument identity and market-observation constraints."""

from datetime import date

import pytest

from examples.compute.series import Series
from examples.portfolio.models import (
    AAPL,
    EquityPrices,
)


@pytest.mark.parametrize("price", [0, -1])
def test_nonpositive_prices(price):
    with pytest.raises(ValueError, match="strictly positive"):
        EquityPrices(AAPL, Series((date(2026, 1, 1),), (price,)))
