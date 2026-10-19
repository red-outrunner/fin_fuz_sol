"""Shared Yahoo Finance currency normalization.

Contract: APIs expose major currency units (ZAR, GBP, USD, …), never ZAc/GBp
as numeric units. Trust Yahoo's `currency` field — never magnitude heuristics
(Capitec/Naspers trade well above R500 in real rands).
"""
from __future__ import annotations

from typing import Optional, Tuple

# Yahoo minor-unit codes → major ISO display codes
_MINOR_TO_MAJOR = {
    "ZAc": "ZAR",
    "GBp": "GBP",
}

_SUFFIX_CURRENCY = {
    "JO": "ZAR",
    "L": "GBP",
    "IL": "GBP",
    "DE": "EUR",
    "F": "EUR",
    "PA": "EUR",
    "AS": "EUR",
    "BR": "EUR",
    "AX": "AUD",
    "TO": "CAD",
    "T": "JPY",
    "HK": "HKD",
}


def display_currency(yahoo_currency: str = "", ticker: str = "") -> str:
    """Map Yahoo currency (incl. ZAc/GBp) to a major ISO code for display/API."""
    raw = (yahoo_currency or "").strip()
    if raw in _MINOR_TO_MAJOR:
        return _MINOR_TO_MAJOR[raw]
    if raw:
        return raw
    sym = (ticker or "").upper().strip()
    if "." in sym:
        suffix = sym.rsplit(".", 1)[-1]
        if suffix in _SUFFIX_CURRENCY:
            return _SUFFIX_CURRENCY[suffix]
    return "USD"


def normalize_price(price, yahoo_currency: str = "") -> Optional[float]:
    """Convert Yahoo quote units to major currency units (ZAc→ZAR, GBp→GBP)."""
    if price is None:
        return None
    try:
        value = float(price)
    except (TypeError, ValueError):
        return None
    if value == 0:
        return 0.0
    cur = (yahoo_currency or "").strip()
    if cur in ("ZAc", "GBp"):
        return value / 100.0
    return value


def normalize_money_pair(
    price,
    yahoo_currency: str = "",
    ticker: str = "",
) -> Tuple[Optional[float], str]:
    """Return (major-unit price, display currency code)."""
    return normalize_price(price, yahoo_currency), display_currency(yahoo_currency, ticker)


# Yahoo history-series minor-unit listings. Unlike Fast Info, quote history
# (Close/Adj Close/dividends) carries no per-bar currency field, so the unit is
# inferred from the ticker suffix: JSE & London/IRL close in ZAc/GBp.
_HISTORY_MINOR_SUFFIX = {
    "JO": "ZAc",
    "L": "GBp",
    "IL": "GBp",
}


def history_quote_mode(ticker: str = "") -> Optional[str]:
    """Yahoo minor unit (ZAc/GBp) for a ticker's history series, else None."""
    sym = (ticker or "").upper().strip()
    if "." in sym:
        suffix = sym.rsplit(".", 1)[-1]
        return _HISTORY_MINOR_SUFFIX.get(suffix)
    return None


def normalize_history_value(value, ticker: str = "") -> Optional[float]:
    """Convert a Yahoo history-series value to major units via ticker suffix."""
    mode = history_quote_mode(ticker)
    if mode is None:
        return normalize_price(value)
    return normalize_price(value, mode)


BRAND_ATTRIBUTION = "Analyzed with Ubomvu (ubomvu.com)"
