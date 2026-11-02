"""Shared provider-backed FX rates. Unavailable rates stay None."""
import math
from typing import Optional

import yfinance as yf


FX_TO_ZAR = {
    "ZAR": None,
    "USD": "ZAR=X",
    "GBP": "GBPZAR=X",
    "EUR": "EURZAR=X",
    "AUD": "AUDZAR=X",
    "CAD": "CADZAR=X",
    "CHF": "CHFZAR=X",
    "JPY": "JPYZAR=X",
    "HKD": "HKDZAR=X",
    "CNY": "CNYZAR=X",
    "SGD": "SGDZAR=X",
    "NZD": "NZDZAR=X",
    "INR": "INRZAR=X",
    "BRL": "BRLZAR=X",
    "MXN": "MXNZAR=X",
    "KRW": "KRWZAR=X",
}

def positive_number(value) -> Optional[float]:
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def fx_rate_to_zar(currency: str, cache: dict) -> Optional[float]:
    """Return a provider-backed rate, or None; never invent an exchange rate."""
    code = (currency or "").upper()
    if code == "ZAR":
        return 1.0
    if code in cache:
        return cache[code]
    if code not in FX_TO_ZAR:
        cache[code] = None
        return None

    def latest(pair):
        try:
            hist = yf.Ticker(pair).history(period="5d")
            return positive_number(hist["Close"].dropna().iloc[-1])
        except Exception:
            return None

    rate = latest(FX_TO_ZAR[code])
    if rate is None and code != "USD":
        usd_zar = fx_rate_to_zar("USD", cache)
        per_unit_usd = latest(f"{code}USD=X") if usd_zar is not None else None
        if per_unit_usd is not None:
            rate = positive_number(per_unit_usd * usd_zar)
    cache[code] = rate
    return rate


