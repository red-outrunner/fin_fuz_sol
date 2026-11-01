"""Daily-feature regressions use provider fixtures, never live market calls."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import screener
from main import app


@pytest.fixture(autouse=True)
def daily_data(monkeypatch):
    monkeypatch.setattr(screener, "_SOTD_CACHE", {"date": None, "payload": None})
    monkeypatch.setattr(screener, "get_jse_top_40", lambda: ["AAA.JO", "BBB.JO", "CCC.JO"])
    monkeypatch.setattr(screener, "_date_index", lambda date, n: 0)
    monkeypatch.setattr(screener, "_sparkline_and_change", lambda ticker: ([], None))
    monkeypatch.setattr(screener, "_day_change_since", lambda ticker, date: None)
    monkeypatch.setattr(screener, "get_sector_for_ticker", lambda *a, **kw: "Banks")


@pytest.mark.parametrize("details, expected", [
    ({}, "featured"),
    ({"pe_ratio": None, "dividend_yield": None}, "featured"),
    ({"pe_ratio": 25, "pct_from_low": 50, "pct_from_high": -20}, "featured"),
    ({"pe_ratio": 8}, "value"),
    ({"dividend_yield": .06}, "dividend"),
    ({"pct_from_low": 5}, "bargain"),
    ({"pct_from_high": -2}, "momentum"),
    ({"revenue_growth": .2}, "growth"),
])
def test_theme_fallback_and_scored_themes(details, expected):
    theme, bullets = screener._build_why_bullets(details)
    assert theme == expected
    assert 1 <= len(bullets) <= 3


def test_featured_endpoint_without_scoring_signals():
    with patch("screener.get_ticker_details", return_value={"current_price": 100, "name": "Alpha"}):
        response = TestClient(app).get("/api/stock-of-the-day")
    assert response.status_code == 200
    assert response.json()["theme"] == "featured"
    assert response.json()["ticker"] == "AAA.JO"


def test_failed_candidates_do_not_mix_company_and_ticker():
    with patch("screener.get_ticker_details", side_effect=lambda t: {"name": t, "current_price": None}):
        result = screener.get_stock_of_the_day()
    assert result["ticker"] == result["name"] == "AAA.JO"
    assert result["current_price"] is None


def test_usable_fallback_is_cached_with_matching_company():
    def details(ticker):
        return {"name": "Beta", "current_price": 120} if ticker == "BBB.JO" else None
    with patch("screener.get_ticker_details", side_effect=details):
        result = screener.get_stock_of_the_day()
    assert result["ticker"] == "BBB.JO"
    assert result["name"] == "Beta"
    with patch("screener.get_ticker_details", side_effect=AssertionError("cache miss")):
        assert screener.get_stock_of_the_day() == result


def test_empty_universe_returns_service_unavailable(monkeypatch):
    monkeypatch.setattr(screener, "get_jse_top_40", lambda: [])
    assert TestClient(app).get("/api/stock-of-the-day").status_code == 503
