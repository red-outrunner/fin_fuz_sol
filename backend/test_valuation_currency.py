"""DCF cash flows and share prices must use the same major currency units."""
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from market_data import get_financials


@pytest.mark.parametrize("reporting,quote,expected,price", [
    ("USD", "ZAc", 18000, 100),
    ("USD", "GBp", 750, 100),
    ("ZAR", "ZAc", 1000, 100),
    ("USD", "USD", 1000, 10000),
])
def test_statement_currency_is_converted_to_quote_major_units(reporting, quote, expected, price):
    stock = SimpleNamespace(
        info={"financialCurrency": reporting, "currency": quote, "currentPrice": 10000,
              "sharesOutstanding": 100, "beta": 1},
        cashflow=pd.DataFrame({"2026": [1000]}, index=["Free Cash Flow"]),
    )
    rates = {"ZAR=X": 18, "GBPZAR=X": 24}

    def provider(symbol):
        if symbol == "TEST":
            return stock
        return SimpleNamespace(history=lambda **kw: pd.DataFrame({"Close": [rates[symbol]]}))

    with patch("market_data.yf.Ticker", side_effect=provider), \
            patch("market_data.core_math.calculate_wacc", return_value=.1):
        result = get_financials("TEST")
    assert result["fcf"] == expected
    assert result["fcf_reporting"] == 1000
    assert result["reporting_currency"] == reporting
    assert result["fcf_currency"] == result["currency"]
    assert result["price"] == price
    assert result["valuation_status"] == "available"
    assert result["warning"] is None


@pytest.mark.parametrize("reporting,quote", [(None, "ZAc"), ("USD", None), ("XYZ", "ZAc"), ("USD", "ZAc")])
def test_missing_metadata_or_exchange_rate_cannot_produce_a_comparison(reporting, quote):
    stock = SimpleNamespace(
        info={"financialCurrency": reporting, "currency": quote, "currentPrice": 10000,
              "freeCashflow": 1000, "sharesOutstanding": 100},
        cashflow=pd.DataFrame(),
    )
    with patch("market_data.yf.Ticker", return_value=stock), \
            patch("market_data.fx_rate_to_zar", return_value=None), \
            patch("market_data.core_math.calculate_wacc", return_value=.1):
        result = get_financials("TEST.JO")
    assert result["fcf"] is None
    assert result["fcf_reporting"] == 1000
    assert result["fcf_currency"] is None
    assert result["valuation_status"] == "unavailable"
    assert result["warning"]


def test_same_currency_needs_no_exchange_rate_even_when_provider_is_down():
    stock = SimpleNamespace(
        info={"financialCurrency": "ZAR", "currency": "ZAc", "freeCashflow": -100,
              "currentPrice": 10000, "sharesOutstanding": 100},
        cashflow=pd.DataFrame(),
    )
    with patch("market_data.yf.Ticker", return_value=stock), \
            patch("market_data.fx_rate_to_zar") as fx, \
            patch("market_data.core_math.calculate_wacc", return_value=.1):
        result = get_financials("TEST.JO")
    fx.assert_not_called()
    assert result["fcf"] == -100
    assert result["fx_rate"] == 1
