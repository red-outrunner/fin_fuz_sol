"""Daily-feature regressions use provider fixtures, never live market calls."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import screener
import daily_picks
from database import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from main import app


@pytest.fixture(autouse=True)
def daily_data(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(daily_picks, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setattr(screener, "_SOTD_CACHE", {"date": None, "payload": None})
    monkeypatch.setattr(screener, "get_jse_top_40", lambda: ["AAA.JO", "BBB.JO", "CCC.JO"])
    monkeypatch.setattr(screener, "_date_index", lambda date, n: 0)
    monkeypatch.setattr(screener, "_sparkline_and_change", lambda ticker: ([], None))
    monkeypatch.setattr(screener, "_day_change_since", lambda ticker, date: None)
    monkeypatch.setattr(screener, "get_sector_for_ticker", lambda *a, **kw: "Banks")
    yield
    engine.dispose()


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


def test_yesterday_preserves_fallback_after_restart_and_provider_recovery(monkeypatch):
    monkeypatch.setattr(screener, "_jhb_today", lambda: "2026-10-31")
    with patch("screener.get_ticker_details", side_effect=lambda t: {"name": "Beta", "current_price": 120} if t == "BBB.JO" else None):
        published = screener.get_stock_of_the_day()
    assert published["ticker"] == "BBB.JO"
    # New process/day: no in-memory cache, recovered first candidate, changed universe.
    monkeypatch.setattr(screener, "_SOTD_CACHE", {"date": None, "payload": None})
    monkeypatch.setattr(screener, "_jhb_today", lambda: "2026-11-01")
    monkeypatch.setattr(screener, "get_jse_top_40", lambda: ["CCC.JO", "AAA.JO"])
    with patch("screener.get_ticker_details", return_value={"name": "Recovered", "current_price": 100}):
        current = screener.get_stock_of_the_day()
    assert current["yesterday"]["ticker"] == "BBB.JO"
    assert current["yesterday"]["name"] == "Beta"
    assert current["yesterday"]["date"] == "2026-10-31"


def test_no_published_history_is_not_fabricated():
    with patch("screener.get_ticker_details", return_value={"name": "Alpha", "current_price": 100}):
        assert screener.get_stock_of_the_day()["yesterday"] is None


def test_first_publication_wins_between_workers():
    daily_picks.publish_pick("2026-11-01", "BBB.JO", "Beta")
    assert daily_picks.publish_pick("2026-11-01", "AAA.JO", "Alpha") == {"ticker": "BBB.JO", "name": "Beta"}
