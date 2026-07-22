"""Lightweight tests for technical analysis helpers (no network beyond yfinance optional)."""
from unittest.mock import patch
import pandas as pd
import numpy as np
from technical import (
    compute_rsi,
    compute_macd,
    compute_bollinger,
    compute_fibonacci,
    detect_patterns,
    backtest_sma_crossover,
)


def _synthetic_ohlcv(n=120, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.2, 1.5, n)
    low = close - rng.uniform(0.2, 1.5, n)
    open_ = close + rng.normal(0, 0.3, n)
    vol = rng.integers(1000, 5000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def test_indicators_finite():
    df = _synthetic_ohlcv()
    rsi = compute_rsi(df["Close"])
    assert rsi.dropna().between(0, 100).all()
    macd, signal, hist = compute_macd(df["Close"])
    assert macd.dropna().shape[0] > 10
    u, m, l = compute_bollinger(df["Close"])
    assert (u.dropna() >= m.dropna()).all()
    assert (m.dropna() >= l.dropna()).all()


def test_fibonacci_and_patterns():
    df = _synthetic_ohlcv(200)
    fib = compute_fibonacci(df)
    assert fib["available"] is True
    assert "61.8%" in fib["levels"]
    patterns = detect_patterns(df)
    assert isinstance(patterns, list)


def test_backtest_runs():
    df = _synthetic_ohlcv(200)
    result = backtest_sma_crossover(df, fast=5, slow=20)
    assert result["available"] is True
    assert "total_return_pct" in result
    assert result["num_trades"] >= 0
