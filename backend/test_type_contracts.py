"""Compatibility coverage for the contracts corrected after the Pylance audit."""
import io

import pandas as pd
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from analytics import calculate_summary_stats, get_jse_peers
from database import Base
from main import AnalysisRequest, _make_excel_safe, rate_limit_exceeded_handler
from models import Alert, PortfolioHolding


def test_rate_limit_adapter_keeps_429_response_and_retry_headers():
    app = FastAPI()
    limiter = Limiter(key_func=get_remote_address, headers_enabled=True)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    @app.get("/limited")
    @limiter.limit("1/minute")
    def limited(request: Request):
        return JSONResponse({"ok": True})

    client = TestClient(app)
    assert client.get("/limited").status_code == 200
    rejected = client.get("/limited")
    assert rejected.status_code == 429
    assert "Rate limit exceeded" in rejected.json()["error"]
    assert int(rejected.headers["Retry-After"]) >= 0


def test_rate_limit_adapter_does_not_mask_unrelated_exceptions():
    error = ValueError("unrelated")
    with pytest.raises(ValueError, match="unrelated"):
        rate_limit_exceeded_handler(Request({"type": "http"}), error)


def test_null_inflation_retains_the_nominal_summary_contract():
    request = AnalysisRequest(ticker="TEST", start_year=2025,
                              end_date="2026-01-01", inflation_rate=None)
    assert request.inflation_rate is None
    returns = pd.Series([.01] * 12, index=pd.date_range("2025-01-31", periods=12, freq="ME"))
    pivot = pd.DataFrame([returns.values], columns=range(1, 13))
    nominal = calculate_summary_stats(returns, pivot, inflation_rate=0.)
    assert nominal is not None
    assert calculate_summary_stats(returns, pivot, inflation_rate=request.inflation_rate) == nominal


def test_excel_timezone_conversion_keeps_values_and_does_not_mutate_input():
    dates = pd.date_range("2026-01-01", periods=2, tz="Africa/Johannesburg")
    original = pd.DataFrame({"date": dates, "price": [100., 120.]}, index=dates)
    safe = _make_excel_safe(original)
    assert original.index.tz is not None
    assert original["date"].dt.tz is not None
    assert safe.index.tz is None
    assert safe["date"].dt.tz is None
    assert safe["price"].tolist() == [100., 120.]
    buffer = io.BytesIO()
    safe.to_excel(buffer)
    restored = pd.read_excel(io.BytesIO(buffer.getvalue()), index_col=0)
    assert restored["date"].tolist() == dates.tz_localize(None).tolist()
    assert restored["price"].tolist() == [100., 120.]


def test_peer_list_and_metadata_preserve_their_distinct_return_shapes():
    for ticker in ("SBK.JO", "STX40.JO", "^J203.JO"):
        peers = get_jse_peers(ticker)
        metadata = get_jse_peers(ticker, return_meta=True)
        assert isinstance(peers, list)
        assert metadata["peers"] == peers
        assert isinstance(metadata["peer_group"], str)
        assert metadata["asset_class"] in ("equity", "etf", "index")


def test_typed_models_read_nullable_legacy_rows_and_persist_scalar_updates():
    engine = create_engine("sqlite://")
    try:
        Base.metadata.create_all(engine)
        # SQL inserts bypass Python defaults, as legacy nullable records may do.
        with engine.begin() as connection:
            connection.execute(text("INSERT INTO alerts (client_key, ticker) VALUES ('owner', 'TEST')"))
        with Session(engine) as db:
            alert = db.query(Alert).one()
            assert alert.active is None
            assert alert.triggered is None
            assert alert.last_triggered_at is None
            assert alert.created_at is None
            alert.active = True
            alert.triggered = True
            holding = PortfolioHolding(client_key="owner", ticker="TEST", quantity=2., avg_cost=100.)
            db.add(holding)
            db.commit()
            holding.quantity += 1
            holding.avg_cost = 120.
            db.commit()
            db.expire_all()
            assert db.query(Alert).one().triggered is True
            stored = db.query(PortfolioHolding).one()
            assert stored.quantity == 3.
            assert stored.avg_cost == 120.
    finally:
        engine.dispose()
