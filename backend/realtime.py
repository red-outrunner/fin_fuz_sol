"""Live quotes, optional vendor APIs, and alert evaluation."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import yfinance as yf

from serialization import clean_data
from technical import volume_zscore_alert

logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
IEX_API_KEY = os.getenv("IEX_API_KEY", "").strip()


def quote_provider() -> str:
    if POLYGON_API_KEY:
        return "polygon"
    if IEX_API_KEY:
        return "iex"
    if ALPHA_VANTAGE_API_KEY:
        return "alpha_vantage"
    return "yfinance"


def _yf_quote(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            fi = t.fast_info
            info = {
                "price": getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None),
                "prev_close": getattr(fi, "previous_close", None) or getattr(fi, "previousClose", None),
                "volume": getattr(fi, "last_volume", None) or getattr(fi, "lastVolume", None),
                "currency": getattr(fi, "currency", None),
            }
        except Exception:
            pass
        if not info.get("price"):
            hist = t.history(period="5d", interval="1d")
            if hist is not None and not hist.empty:
                info["price"] = float(hist["Close"].iloc[-1])
                if len(hist) >= 2:
                    info["prev_close"] = float(hist["Close"].iloc[-2])
                info["volume"] = float(hist["Volume"].iloc[-1]) if "Volume" in hist else None
        price = info.get("price")
        if price is None:
            return None
        prev = info.get("prev_close")
        change = None
        change_pct = None
        if prev:
            change = float(price) - float(prev)
            change_pct = (change / float(prev)) * 100
        return {
            "ticker": ticker.upper(),
            "price": round(float(price), 4),
            "prev_close": round(float(prev), 4) if prev else None,
            "change": round(change, 4) if change is not None else None,
            "change_pct": round(change_pct, 3) if change_pct is not None else None,
            "volume": info.get("volume"),
            "currency": info.get("currency"),
            "provider": "yfinance",
            "delayed": True,
            "as_of": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error(f"yfinance quote failed for {ticker}: {e}")
        return None


def _polygon_quote(ticker: str) -> Optional[Dict[str, Any]]:
    # Polygon uses tickers without .JO — map SA equities poorly; fall back if fails
    sym = ticker.replace(".JO", "").replace("^", "I:")
    try:
        url = f"https://api.polygon.io/v2/last/trade/{sym}"
        r = requests.get(url, params={"apiKey": POLYGON_API_KEY}, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json().get("results") or r.json().get("result") or {}
        price = data.get("p") or data.get("price")
        if price is None:
            return None
        return {
            "ticker": ticker.upper(),
            "price": round(float(price), 4),
            "prev_close": None,
            "change": None,
            "change_pct": None,
            "volume": data.get("s"),
            "currency": None,
            "provider": "polygon",
            "delayed": False,
            "as_of": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.warning(f"Polygon quote failed for {ticker}: {e}")
        return None


def get_live_quote(ticker: str) -> Optional[Dict[str, Any]]:
    ticker = ticker.upper().strip()
    if POLYGON_API_KEY:
        q = _polygon_quote(ticker)
        if q:
            return q
    return _yf_quote(ticker)


def get_live_quotes(tickers: List[str]) -> Dict[str, Any]:
    out = {}
    for t in tickers[:30]:
        q = get_live_quote(t)
        if q:
            out[t.upper()] = q
    return clean_data({
        "provider": quote_provider(),
        "quotes": out,
        "note": (
            "Set POLYGON_API_KEY, IEX_API_KEY, or ALPHA_VANTAGE_API_KEY for lower-latency quotes. "
            "Default provider is yfinance (delayed)."
            if quote_provider() == "yfinance"
            else f"Streaming via {quote_provider()}."
        ),
    })


def evaluate_price_alert(ticker: str, condition: str, threshold: float) -> Optional[Dict[str, Any]]:
    q = get_live_quote(ticker)
    if not q or q.get("price") is None:
        return None
    price = float(q["price"])
    cond = condition.lower().strip()
    triggered = False
    if cond in ("above", "crosses_above", ">"):
        triggered = price >= threshold
    elif cond in ("below", "crosses_below", "<"):
        triggered = price <= threshold
    return {
        "ticker": ticker.upper(),
        "type": "price",
        "condition": cond,
        "threshold": threshold,
        "price": price,
        "triggered": triggered,
        "message": (
            f"{ticker.upper()} is {price} ({cond} {threshold})"
            if triggered else None
        ),
    }


def evaluate_earnings_surprise(ticker: str) -> Optional[Dict[str, Any]]:
    """Compare latest reported EPS vs estimate when yfinance exposes it."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        actual = info.get("trailingEps")
        estimate = info.get("forwardEps") or info.get("epsForward")
        # Prefer earnings history if present
        surprise_pct = None
        try:
            eh = t.earnings_history
            if eh is not None and not eh.empty:
                row = eh.iloc[0]
                # columns vary by yfinance version
                for a_col, e_col in (
                    ("epsActual", "epsEstimate"),
                    ("Actual", "Estimate"),
                ):
                    if a_col in eh.columns and e_col in eh.columns:
                        actual = row[a_col]
                        estimate = row[e_col]
                        break
        except Exception:
            pass

        if actual is None or estimate is None or float(estimate) == 0:
            return {
                "ticker": ticker.upper(),
                "type": "earnings_surprise",
                "triggered": False,
                "available": False,
                "message": "No earnings estimate/actual pair available for this ticker.",
            }

        surprise_pct = (float(actual) - float(estimate)) / abs(float(estimate)) * 100
        triggered = abs(surprise_pct) >= 5.0
        return {
            "ticker": ticker.upper(),
            "type": "earnings_surprise",
            "available": True,
            "actual_eps": round(float(actual), 4),
            "estimated_eps": round(float(estimate), 4),
            "surprise_pct": round(surprise_pct, 2),
            "triggered": triggered,
            "message": (
                f"{ticker.upper()} earnings surprise {surprise_pct:+.1f}% "
                f"(actual {actual} vs est {estimate})"
                if triggered else None
            ),
        }
    except Exception as e:
        logger.error(f"Earnings surprise check failed for {ticker}: {e}")
        return None


def evaluate_alerts(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate a batch of alert rules.
    Rule shapes:
      { id, type: "price"|"volume"|"earnings", ticker, condition?, threshold? }
    """
    fired = []
    checked = []
    for rule in rules[:50]:
        rtype = (rule.get("type") or "price").lower()
        ticker = (rule.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        result = None
        if rtype == "price":
            result = evaluate_price_alert(
                ticker,
                rule.get("condition", "above"),
                float(rule.get("threshold", 0)),
            )
        elif rtype == "volume":
            result = volume_zscore_alert(ticker, float(rule.get("threshold", 2.5)))
            if result:
                result["type"] = "volume"
                result["message"] = (
                    f"{ticker} unusual volume (z={result['z_score']})"
                    if result.get("triggered") else None
                )
        elif rtype == "earnings":
            result = evaluate_earnings_surprise(ticker)

        if result:
            result["id"] = rule.get("id")
            checked.append(result)
            if result.get("triggered"):
                fired.append(result)

    return clean_data({
        "checked_at": datetime.utcnow().isoformat() + "Z",
        "provider": quote_provider(),
        "fired": fired,
        "results": checked,
    })
