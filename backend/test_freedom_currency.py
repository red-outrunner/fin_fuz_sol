"""Foreign price and dividend amounts must use the same real ZAR exchange rate."""
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from analytics import calculate_financial_freedom
from main import app


@pytest.mark.parametrize("ticker, quote_currency, rate, pair, unit", [
    ("TEST.DE", "EUR", 20, "EURZAR=X", 1),
    ("TEST.L", "GBP", 25, "GBPZAR=X", 1),
    ("TEST.L", "GBp", 25, "GBPZAR=X", 100),
    ("TEST", "USD", 18, "ZAR=X", 1),
    ("TEST.AX", "AUD", 12, "AUDZAR=X", 1),
    ("TEST.TO", "CAD", 15, "CADZAR=X", 1),
    ("TEST.JO", "ZAR", 1, None, 1),
    ("TEST.JO", "ZAc", 1, None, 100),
])
def test_prices_and_dividends_are_converted_together(ticker, quote_currency, rate, pair, unit):
    dividends = pd.Series([2. * unit], index=pd.to_datetime(["2026-10-01"]))
    history = pd.DataFrame({"Close": [100. * unit]})
    def provider(symbol):
        if symbol == ticker:
            return SimpleNamespace(info={"currency": quote_currency}, dividends=dividends)
        assert symbol == pair
        return SimpleNamespace(history=lambda **kw: pd.DataFrame({"Close": [rate]}))
    with patch("yfinance.Ticker", side_effect=provider) as fetch:
        result = calculate_financial_freedom(history, 100, ticker)
    assert result["current_price"] == 100 * rate
    assert result["dividends_12m"] == 2 * rate
    assert result["annual_yield"] == pytest.approx(.02)
    assert result["investment_needed"] == pytest.approx(60000)
    assert result["shares_needed"] == int(60000 / (100 * rate))
    assert result["currency"] == "ZAR"
    assert result["fx_rate_to_zar"] == rate
    # One rate lookup, reused for both monetary amounts. ZAR needs none.
    assert fetch.call_count == (1 if pair is None else 2)


@pytest.mark.parametrize("currency", ["EUR", "USD", "XYZ"])
def test_missing_fx_is_explicit_and_does_not_relabel_foreign_amounts(currency):
    stock = SimpleNamespace(info={"currency": currency})
    def provider(symbol):
        if symbol == "TEST":
            return stock
        raise RuntimeError("FX unavailable")
    with patch("yfinance.Ticker", side_effect=provider):
        result = calculate_financial_freedom(pd.DataFrame({"Close": [100]}), 100, "TEST")
    assert result["valuation_status"] == "unavailable"
    assert result["source_currency"] == currency
    for field in ("current_price", "dividends_12m", "shares_needed", "investment_needed", "annual_yield"):
        assert result[field] is None
    assert "Exchange rate" in result["warning"]


def test_freedom_endpoint_reports_fx_outage_without_server_error():
    with patch("main.download_data", return_value=pd.DataFrame({"Close": [100]})), \
         patch("yfinance.Ticker") as provider:
        provider.return_value.info = {"currency": "EUR"}
        provider.return_value.history.side_effect = RuntimeError("FX unavailable")
        response = TestClient(app).post("/api/freedom", json={"ticker": "TEST.DE", "monthly_income_goal": 100})
    assert response.status_code == 200
    assert response.json()["valuation_status"] == "unavailable"
    assert response.json()["investment_needed"] is None
