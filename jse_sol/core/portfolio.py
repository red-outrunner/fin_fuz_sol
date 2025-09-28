import json, pathlib
from typing import List
from .instruments import Instrument, CACHE_PATH

LIST_FILE = CACHE_PATH.with_name("watchlist.json")

class WatchList:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.items: List[Instrument] = []
            cls._instance.load()
        return cls._instance

    # ---------- CRUD ----------
    def add(self, symbol: str) -> bool:
        symbol = symbol.upper()
        if any(i.symbol == symbol for i in self.items):
            return False
        self.items.append(Instrument.fetch(symbol))
        self.save()
        return True

    def remove(self, symbol: str):
        self.items = [i for i in self.items if i.symbol != symbol.upper()]
        self.save()

    def reorder(self, new_order: List[str]):   # ["AAPL","^GSPC"]
        symbols = {i.symbol: i for i in self.items}
        self.items = [symbols[s] for s in new_order]
        self.save()

    # ---------- persistence ----------
    def load(self):
        if LIST_FILE.exists():
            data = json.loads(LIST_FILE.read_text())
            self.items = [Instrument(**d) for d in data]

    def save(self):
        LIST_FILE.write_text(json.dumps([asdict(i) for i in self.items], indent=2))
