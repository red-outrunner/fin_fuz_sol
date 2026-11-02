"""Missing provider data must not be turned into fabricated portfolio values."""
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from main import _holding_quote, _fx_rate_to_zar
from test_portfolio_access import client  # disposable database fixture


HEADERS = {"X-Portfolio-Key": "a" * 64}


def add(api, ticker="SBK.JO"):
    response = api.post("/api/portfolio/holdings", headers=HEADERS,
                        json={"ticker": ticker, "quantity": 2, "avg_cost": 100})
    assert response.status_code == 200


@pytest.mark.parametrize("price", [None, 0, -1, "bad", float("nan"), float("inf")])
def test_invalid_quote_stays_unavailable(price):
    with patch("main.yf.Ticker") as provider:
        provider.return_value.info = {"currentPrice": price, "currency": "ZAR"}
        assert _holding_quote("SBK.JO")["current_price"] is None


def test_quote_failure_keeps_holdings_and_cost_without_false_loss(client):
    api, _ = client
    add(api)
    with patch("main.yf.Ticker", side_effect=RuntimeError("provider unavailable")):
        result = api.get("/api/portfolio/holdings", headers=HEADERS).json()
        performance = api.get("/api/portfolio/performance", headers=HEADERS).json()
    row = result["holdings"][0]
    assert result["count"] == 1
    assert row["cost_basis"] == row["cost_zar"] == 200
    for field in ("current_price", "current_value", "pnl", "pnl_pct", "value_zar", "pnl_zar"):
        assert row[field] is None
    assert row["missing_data"] == ["price"]
    assert performance["total_value"] is None
    assert performance["total_cost"] == 200
    assert performance["total_pnl"] is None
    assert performance["total_pnl_pct"] is None
    assert performance["sector_allocation"] == {}
    assert performance["valuation_status"] == "incomplete"


@pytest.mark.parametrize("currency", ["USD", "EUR", "XYZ", ""])
def test_failed_or_unknown_fx_never_uses_fixed_rates(currency):
    with patch("main.yf.Ticker", side_effect=RuntimeError("provider unavailable")) as provider:
        cache = {}
        assert _fx_rate_to_zar(currency, cache) is None
        calls = provider.call_count
        assert _fx_rate_to_zar(currency, cache) is None
        assert provider.call_count == calls


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_fx_is_unavailable(rate):
    with patch("main.yf.Ticker") as provider:
        provider.return_value.history.return_value = pd.DataFrame({"Close": [rate]})
        assert _fx_rate_to_zar("USD", {}) is None


def test_provider_backed_cross_rate():
    rates = {"EURZAR=X": None, "ZAR=X": 18, "EURUSD=X": 1.1}
    def provider(pair):
        return SimpleNamespace(history=lambda **kw: pd.DataFrame({"Close": [rates[pair]]}))
    with patch("main.yf.Ticker", side_effect=provider):
        assert _fx_rate_to_zar("EUR", {}) == pytest.approx(19.8)


def test_unknown_fx_preserves_native_values_but_suppresses_zar_totals(client):
    api, _ = client
    add(api, "TEST")
    with patch("main.yf.Ticker") as provider:
        provider.return_value.info = {"currentPrice": 150, "currency": "XYZ"}
        row = api.get("/api/portfolio/holdings", headers=HEADERS).json()["holdings"][0]
        performance = api.get("/api/portfolio/performance", headers=HEADERS).json()
    assert row["current_value"] == 300
    assert row["pnl"] == 100
    assert row["pnl_pct"] == 50
    assert row["missing_data"] == ["fx"]
    for field in ("value_zar", "cost_zar", "pnl_zar"):
        assert row[field] is None
    assert performance["total_value"] is performance["total_cost"] is None


def test_partial_portfolio_does_not_present_subset_as_total(client):
    api, _ = client
    add(api)
    add(api, "NED.JO")
    def provider(ticker):
        return SimpleNamespace(info={"currency": "ZAR", "currentPrice": 150 if ticker == "SBK.JO" else None})
    with patch("main.yf.Ticker", side_effect=provider):
        rows = api.get("/api/portfolio/holdings", headers=HEADERS).json()["holdings"]
        performance = api.get("/api/portfolio/performance", headers=HEADERS).json()
    assert sorted([row["valuation_status"] for row in rows]) == ["available", "unavailable"]
    assert performance["total_value"] is None
    assert performance["unavailable_holdings"] == [{"ticker": "NED.JO", "missing_data": ["price"]}]


def test_complete_portfolio_still_normalizes_cents_and_converts_fx(client):
    api, _ = client
    add(api)
    add(api, "AAPL")
    def provider(ticker):
        if ticker == "ZAR=X":
            return SimpleNamespace(history=lambda **kw: pd.DataFrame({"Close": [18]}))
        return SimpleNamespace(info={"currency": "ZAc" if ticker == "SBK.JO" else "USD",
                                     "currentPrice": 15000 if ticker == "SBK.JO" else 150})
    with patch("main.yf.Ticker", side_effect=provider):
        result = api.get("/api/portfolio/performance", headers=HEADERS).json()
    assert result["valuation_status"] == "available"
    assert result["total_value"] == 5700
    assert result["total_cost"] == 3800
    assert result["total_pnl"] == 1900
    assert result["total_pnl_pct"] == 50


def test_missing_quote_units_are_not_assumed():
    with patch("main.yf.Ticker") as provider:
        provider.return_value.info = {"currentPrice": 15000}
        assert _holding_quote("SBK.JO")["current_price"] is None


def test_regular_market_price_fallback():
    with patch("main.yf.Ticker") as provider:
        provider.return_value.info = {"currentPrice": None, "regularMarketPrice": 15000, "currency": "ZAc"}
        assert _holding_quote("SBK.JO")["current_price"] == 150
