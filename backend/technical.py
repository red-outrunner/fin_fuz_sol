"""Technical analysis: OHLCV multi-timeframe, indicators, patterns, backtests."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from serialization import clean_data

logger = logging.getLogger(__name__)

TIMEFRAME_MAP = {
    "daily": {"interval": "1d", "period": "2y"},
    "weekly": {"interval": "1wk", "period": "5y"},
    "monthly": {"interval": "1mo", "period": "10y"},
}


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # yfinance sometimes returns MultiIndex columns
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    rename = {c: c.title() for c in out.columns if isinstance(c, str)}
    out = out.rename(columns=rename)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols:
        if c not in out.columns:
            if c == "Close" and "Adj Close" in out.columns:
                out["Close"] = out["Adj Close"]
            else:
                out[c] = np.nan
    out = out[cols].dropna(subset=["Close"])
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out


def fetch_ohlcv(ticker: str, timeframe: str = "daily") -> pd.DataFrame:
    """Fetch OHLCV for daily / weekly / monthly."""
    tf = TIMEFRAME_MAP.get(timeframe, TIMEFRAME_MAP["daily"])
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=tf["period"], interval=tf["interval"], auto_adjust=True)
        return _normalize_ohlcv(df)
    except Exception as e:
        logger.error(f"OHLCV fetch failed for {ticker}/{timeframe}: {e}")
        return pd.DataFrame()


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = compute_sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def compute_fibonacci(df: pd.DataFrame, lookback: int = 120) -> Dict[str, Any]:
    """Fibonacci retracement levels from swing high/low over lookback bars."""
    if df.empty or len(df) < 10:
        return {"available": False}
    window = df.tail(lookback)
    swing_low = float(window["Low"].min())
    swing_high = float(window["High"].max())
    diff = swing_high - swing_low
    if diff <= 0:
        return {"available": False}
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {
        f"{int(r * 1000) / 10 if r not in (0.0, 0.5, 1.0) else int(r * 100)}%": round(
            swing_high - diff * r, 4
        )
        for r in ratios
    }
    # Cleaner keys
    levels = {
        "0%": round(swing_high, 4),
        "23.6%": round(swing_high - diff * 0.236, 4),
        "38.2%": round(swing_high - diff * 0.382, 4),
        "50%": round(swing_high - diff * 0.5, 4),
        "61.8%": round(swing_high - diff * 0.618, 4),
        "78.6%": round(swing_high - diff * 0.786, 4),
        "100%": round(swing_low, 4),
    }
    return {
        "available": True,
        "swing_high": round(swing_high, 4),
        "swing_low": round(swing_low, 4),
        "lookback_bars": len(window),
        "levels": levels,
    }


def _local_extrema(series: pd.Series, order: int = 5) -> Tuple[List[int], List[int]]:
    """Return indices of local peaks and troughs."""
    vals = series.values
    peaks, troughs = [], []
    for i in range(order, len(vals) - order):
        window = vals[i - order : i + order + 1]
        if np.isnan(window).any():
            continue
        if vals[i] == np.max(window):
            peaks.append(i)
        if vals[i] == np.min(window):
            troughs.append(i)
    return peaks, troughs


def detect_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Heuristic detection of double top/bottom and head & shoulders."""
    patterns: List[Dict[str, Any]] = []
    if df.empty or len(df) < 40:
        return patterns

    close = df["Close"]
    peaks, troughs = _local_extrema(close, order=5)
    dates = df.index

    # Double top: two peaks within 2% price, separated by ≥10 bars
    for i in range(len(peaks) - 1):
        for j in range(i + 1, len(peaks)):
            a, b = peaks[i], peaks[j]
            if b - a < 10:
                continue
            pa, pb = float(close.iloc[a]), float(close.iloc[b])
            if pa == 0:
                continue
            if abs(pa - pb) / pa <= 0.02:
                mid_low = float(close.iloc[a:b].min())
                if mid_low < min(pa, pb) * 0.97:
                    patterns.append({
                        "type": "double_top",
                        "label": "Double Top",
                        "bias": "bearish",
                        "confidence": round(1 - abs(pa - pb) / pa, 3),
                        "points": [
                            {"date": str(dates[a].date()), "price": round(pa, 4)},
                            {"date": str(dates[b].date()), "price": round(pb, 4)},
                        ],
                        "neckline": round(mid_low, 4),
                    })
                    break
        if any(p["type"] == "double_top" for p in patterns[-1:]):
            break

    # Double bottom
    for i in range(len(troughs) - 1):
        for j in range(i + 1, len(troughs)):
            a, b = troughs[i], troughs[j]
            if b - a < 10:
                continue
            pa, pb = float(close.iloc[a]), float(close.iloc[b])
            if pa == 0:
                continue
            if abs(pa - pb) / pa <= 0.02:
                mid_high = float(close.iloc[a:b].max())
                if mid_high > max(pa, pb) * 1.03:
                    patterns.append({
                        "type": "double_bottom",
                        "label": "Double Bottom",
                        "bias": "bullish",
                        "confidence": round(1 - abs(pa - pb) / pa, 3),
                        "points": [
                            {"date": str(dates[a].date()), "price": round(pa, 4)},
                            {"date": str(dates[b].date()), "price": round(pb, 4)},
                        ],
                        "neckline": round(mid_high, 4),
                    })
                    break
        if any(p["type"] == "double_bottom" for p in patterns[-1:]):
            break

    # Head & shoulders: three peaks, middle highest
    if len(peaks) >= 3:
        for i in range(len(peaks) - 2):
            l, h, r = peaks[i], peaks[i + 1], peaks[i + 2]
            pl, ph, pr = float(close.iloc[l]), float(close.iloc[h]), float(close.iloc[r])
            if ph > pl and ph > pr and abs(pl - pr) / max(pl, pr) <= 0.05:
                if h - l >= 5 and r - h >= 5:
                    neck = min(float(close.iloc[l:h].min()), float(close.iloc[h:r].min()))
                    patterns.append({
                        "type": "head_and_shoulders",
                        "label": "Head & Shoulders",
                        "bias": "bearish",
                        "confidence": round(min(ph / max(pl, pr) - 1, 0.15) / 0.15, 3),
                        "points": [
                            {"date": str(dates[l].date()), "price": round(pl, 4), "role": "left_shoulder"},
                            {"date": str(dates[h].date()), "price": round(ph, 4), "role": "head"},
                            {"date": str(dates[r].date()), "price": round(pr, 4), "role": "right_shoulder"},
                        ],
                        "neckline": round(neck, 4),
                    })
                    break

    # Inverse H&S
    if len(troughs) >= 3:
        for i in range(len(troughs) - 2):
            l, h, r = troughs[i], troughs[i + 1], troughs[i + 2]
            pl, ph, pr = float(close.iloc[l]), float(close.iloc[h]), float(close.iloc[r])
            if ph < pl and ph < pr and abs(pl - pr) / max(pl, pr) <= 0.05:
                if h - l >= 5 and r - h >= 5:
                    neck = max(float(close.iloc[l:h].max()), float(close.iloc[h:r].max()))
                    patterns.append({
                        "type": "inverse_head_and_shoulders",
                        "label": "Inverse Head & Shoulders",
                        "bias": "bullish",
                        "confidence": round(min(max(pl, pr) / ph - 1, 0.15) / 0.15, 3),
                        "points": [
                            {"date": str(dates[l].date()), "price": round(pl, 4), "role": "left_shoulder"},
                            {"date": str(dates[h].date()), "price": round(ph, 4), "role": "head"},
                            {"date": str(dates[r].date()), "price": round(pr, 4), "role": "right_shoulder"},
                        ],
                        "neckline": round(neck, 4),
                    })
                    break

    # Keep most recent / highest confidence
    patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
    return patterns[:6]


def backtest_sma_crossover(
    df: pd.DataFrame, fast: int = 20, slow: int = 50, initial_cash: float = 100_000.0
) -> Dict[str, Any]:
    """Long-only SMA crossover: enter when fast > slow, exit when fast < slow."""
    if df.empty or len(df) < slow + 5:
        return {"available": False, "reason": "Insufficient history"}

    close = df["Close"].astype(float)
    sma_fast = compute_sma(close, fast)
    sma_slow = compute_sma(close, slow)
    signal = (sma_fast > sma_slow).astype(int)
    position = signal.shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    strat_ret = position * ret
    equity = (1 + strat_ret).cumprod() * initial_cash
    buy_hold = (1 + ret).cumprod() * initial_cash

    trades = []
    prev = 0
    entry_price = None
    entry_date = None
    for dt, pos in position.items():
        pos = int(pos)
        if pos == 1 and prev == 0:
            entry_price = float(close.loc[dt])
            entry_date = str(pd.Timestamp(dt).date())
        elif pos == 0 and prev == 1 and entry_price is not None:
            exit_price = float(close.loc[dt])
            trades.append({
                "entry_date": entry_date,
                "exit_date": str(pd.Timestamp(dt).date()),
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "return_pct": round((exit_price / entry_price - 1) * 100, 2),
            })
            entry_price = None
        prev = pos

    total_ret = float(equity.iloc[-1] / initial_cash - 1)
    bh_ret = float(buy_hold.iloc[-1] / initial_cash - 1)
    vol = float(strat_ret.std() * np.sqrt(252)) if strat_ret.std() > 0 else 0.0
    sharpe = float((strat_ret.mean() * 252) / (strat_ret.std() * np.sqrt(252))) if strat_ret.std() > 0 else 0.0
    dd = float(((equity / equity.cummax()) - 1).min())

    equity_curve = [
        {"date": str(pd.Timestamp(i).date()), "strategy": round(float(v), 2), "buy_hold": round(float(buy_hold.loc[i]), 2)}
        for i, v in equity.iloc[:: max(1, len(equity) // 120)].items()
    ]

    return {
        "available": True,
        "strategy": f"SMA({fast}) / SMA({slow}) crossover",
        "fast": fast,
        "slow": slow,
        "initial_cash": initial_cash,
        "final_value": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(total_ret * 100, 2),
        "buy_hold_return_pct": round(bh_ret * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(dd * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "num_trades": len(trades),
        "win_rate_pct": round(
            100 * sum(1 for t in trades if t["return_pct"] > 0) / len(trades), 1
        ) if trades else 0.0,
        "trades": trades[-20:],
        "equity_curve": equity_curve,
    }


def _series_to_points(series: pd.Series, every: int = 1) -> List[Dict[str, Any]]:
    pts = []
    for i, (dt, val) in enumerate(series.items()):
        if every > 1 and i % every != 0 and i != len(series) - 1:
            continue
        if pd.isna(val):
            continue
        pts.append({"time": str(pd.Timestamp(dt).date()), "value": round(float(val), 4)})
    return pts


def build_technical_snapshot(ticker: str, timeframe: str = "daily") -> Optional[Dict[str, Any]]:
    """Full TA payload: candles, indicators, fib, patterns, multi-TF summary."""
    ticker = ticker.upper().strip()
    df = fetch_ohlcv(ticker, timeframe)
    if df.empty:
        return None

    close = df["Close"]
    rsi = compute_rsi(close)
    macd_line, signal_line, hist = compute_macd(close)
    bb_u, bb_m, bb_l = compute_bollinger(close)
    sma20 = compute_sma(close, 20)
    sma50 = compute_sma(close, 50)
    sma200 = compute_sma(close, 200)

    # Candles for lightweight charts (unix time)
    candles = []
    volumes = []
    for dt, row in df.iterrows():
        ts = int(pd.Timestamp(dt).timestamp())
        candles.append({
            "time": ts,
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
        })
        volumes.append({
            "time": ts,
            "value": float(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
            "color": "rgba(74,124,89,0.5)" if row["Close"] >= row["Open"] else "rgba(140,74,74,0.5)",
        })

    last = {
        "price": round(float(close.iloc[-1]), 4),
        "rsi": round(float(rsi.iloc[-1]), 2) if not pd.isna(rsi.iloc[-1]) else None,
        "macd": round(float(macd_line.iloc[-1]), 4) if not pd.isna(macd_line.iloc[-1]) else None,
        "macd_signal": round(float(signal_line.iloc[-1]), 4) if not pd.isna(signal_line.iloc[-1]) else None,
        "macd_hist": round(float(hist.iloc[-1]), 4) if not pd.isna(hist.iloc[-1]) else None,
        "bb_upper": round(float(bb_u.iloc[-1]), 4) if not pd.isna(bb_u.iloc[-1]) else None,
        "bb_mid": round(float(bb_m.iloc[-1]), 4) if not pd.isna(bb_m.iloc[-1]) else None,
        "bb_lower": round(float(bb_l.iloc[-1]), 4) if not pd.isna(bb_l.iloc[-1]) else None,
        "sma20": round(float(sma20.iloc[-1]), 4) if not pd.isna(sma20.iloc[-1]) else None,
        "sma50": round(float(sma50.iloc[-1]), 4) if not pd.isna(sma50.iloc[-1]) else None,
        "sma200": round(float(sma200.iloc[-1]), 4) if not pd.isna(sma200.iloc[-1]) else None,
    }

    # RSI regime
    rsi_val = last["rsi"]
    if rsi_val is not None:
        if rsi_val >= 70:
            last["rsi_regime"] = "overbought"
        elif rsi_val <= 30:
            last["rsi_regime"] = "oversold"
        else:
            last["rsi_regime"] = "neutral"

    # Multi-timeframe RSI/trend snapshot
    mtf = {}
    for tf in ("daily", "weekly", "monthly"):
        if tf == timeframe:
            mtf[tf] = {
                "rsi": last["rsi"],
                "sma20": last["sma20"],
                "sma50": last["sma50"],
                "trend": (
                    "bullish" if last["sma20"] and last["sma50"] and last["sma20"] > last["sma50"]
                    else "bearish" if last["sma20"] and last["sma50"]
                    else "n/a"
                ),
            }
        else:
            tdf = fetch_ohlcv(ticker, tf)
            if tdf.empty:
                mtf[tf] = None
                continue
            c = tdf["Close"]
            r = compute_rsi(c)
            s20 = compute_sma(c, 20)
            s50 = compute_sma(c, min(50, max(10, len(c) // 3)))
            mtf[tf] = {
                "rsi": round(float(r.iloc[-1]), 2) if not pd.isna(r.iloc[-1]) else None,
                "sma20": round(float(s20.iloc[-1]), 4) if not pd.isna(s20.iloc[-1]) else None,
                "sma50": round(float(s50.iloc[-1]), 4) if not pd.isna(s50.iloc[-1]) else None,
                "trend": (
                    "bullish" if not pd.isna(s20.iloc[-1]) and not pd.isna(s50.iloc[-1]) and s20.iloc[-1] > s50.iloc[-1]
                    else "bearish" if not pd.isna(s20.iloc[-1]) and not pd.isna(s50.iloc[-1])
                    else "n/a"
                ),
            }

    # Indicator series (downsampled for payload size)
    step = max(1, len(df) // 250)
    indicator_series = {
        "rsi": _series_to_points(rsi, step),
        "macd": _series_to_points(macd_line, step),
        "macd_signal": _series_to_points(signal_line, step),
        "bb_upper": _series_to_points(bb_u, step),
        "bb_mid": _series_to_points(bb_m, step),
        "bb_lower": _series_to_points(bb_l, step),
        "sma20": _series_to_points(sma20, step),
        "sma50": _series_to_points(sma50, step),
    }

    return clean_data({
        "ticker": ticker,
        "timeframe": timeframe,
        "candles": candles,
        "volumes": volumes,
        "latest": last,
        "fibonacci": compute_fibonacci(df),
        "patterns": detect_patterns(df),
        "multi_timeframe": mtf,
        "indicator_series": indicator_series,
        "as_of": datetime.utcnow().isoformat() + "Z",
    })


def run_backtest(ticker: str, timeframe: str = "daily", fast: int = 20, slow: int = 50) -> Dict[str, Any]:
    df = fetch_ohlcv(ticker, timeframe)
    result = backtest_sma_crossover(df, fast=fast, slow=slow)
    result["ticker"] = ticker.upper()
    result["timeframe"] = timeframe
    return clean_data(result)


def volume_zscore_alert(ticker: str, z_threshold: float = 2.5) -> Optional[Dict[str, Any]]:
    """Flag unusual volume vs 20-day average."""
    df = fetch_ohlcv(ticker, "daily")
    if df.empty or len(df) < 25:
        return None
    vol = df["Volume"].astype(float)
    mean = vol.iloc[-21:-1].mean()
    std = vol.iloc[-21:-1].std()
    last = float(vol.iloc[-1])
    if std == 0 or pd.isna(std):
        return None
    z = (last - mean) / std
    return {
        "ticker": ticker.upper(),
        "triggered": bool(z >= z_threshold),
        "z_score": round(float(z), 2),
        "last_volume": last,
        "avg_volume_20": round(float(mean), 0),
        "threshold": z_threshold,
    }
