from __future__ import annotations
import yfinance as yf
from dataclasses import dataclass, field, asdict
from typing import List, Dict
import json, pathlib, appdirs

CACHE_PATH = pathlib.Path(appdirs.user_data_dir("jse_analyzer")) / "instruments.json"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

@dataclass(slots=True)
class Instrument:
    symbol: str                      # "^GSPC"
    name: str                        # "S&P 500"
    currency: str = ""
    exchange: str = ""
    asset_type: str = "EQ"           # EQ, IDX, ETF, CRYPTO …
    colour: str = "#3498db"          # hex for chart line

    @staticmethod
    def fetch(symbol: str) -> "Instrument":
        """Return cached or live metadata."""
        cache: Dict = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
        if symbol in cache:
            return Instrument(**cache[symbol])
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        obj = Instrument(
            symbol=symbol.upper(),
            name=info.get("longName") or info.get("shortName") or symbol,
            currency=info.get("currency", ""),
            exchange=info.get("exchange", ""),
            asset_type=("IDX" if symbol.startswith("^") else
                        "CRYPTO" if info.get("quoteType") == "CRYPTOCURRENCY" else
                        info.get("quoteType", "EQ")),
        )
        cache[symbol] = asdict(obj)
        CACHE_PATH.write_text(json.dumps(cache, indent=2))
        return obj
