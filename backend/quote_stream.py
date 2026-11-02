"""Bounded, process-local admission and shared caching for public quote streams."""
from collections import Counter, OrderedDict, deque
from threading import Lock
from time import monotonic


class QuoteConnectionLimiter:
    def __init__(self, max_connections=100, max_per_ip=3,
                 starts_per_minute=10, global_starts_per_minute=120, clock=monotonic):
        self.max_connections = max_connections
        self.max_per_ip = max_per_ip
        self.starts_per_minute = starts_per_minute
        self.global_starts_per_minute = global_starts_per_minute
        self.clock = clock
        self.active = Counter()
        self.starts = deque()
        self.lock = Lock()

    def acquire(self, ip):
        with self.lock:
            now = self.clock()
            while self.starts and self.starts[0][0] <= now - 60:
                self.starts.popleft()
            if (sum(self.active.values()) >= self.max_connections
                    or self.active[ip] >= self.max_per_ip
                    or len(self.starts) >= self.global_starts_per_minute
                    or sum(host == ip for _, host in self.starts) >= self.starts_per_minute):
                return False
            self.active[ip] += 1
            self.starts.append((now, ip))
            return True

    def release(self, ip):
        with self.lock:
            self.active[ip] -= 1
            if self.active[ip] <= 0:
                del self.active[ip]


class SharedQuoteCache:
    def __init__(self, ttl_seconds: float = 5.0, max_entries=512, clock=monotonic):
        if ttl_seconds < 3 or max_entries < 30:
            raise ValueError("Quote cache requires TTL >= 3s and capacity >= 30")
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.clock = clock
        self.entries = OrderedDict()
        self.lock = Lock()

    def get_quotes(self, tickers, fetch):
        tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))[:30]
        # Serialize cache misses so simultaneous subscribers (including overlapping
        # ticker lists) cannot issue duplicate upstream requests. Called in workers.
        with self.lock:
            now = self.clock()
            missing = [t for t in tickers
                       if t not in self.entries or self.entries[t][0] <= now]
            # Protect requested cache hits from eviction while inserting misses.
            for ticker in tickers:
                if ticker in self.entries:
                    self.entries.move_to_end(ticker)
            if missing:
                payload = fetch(missing)
                expires = self.clock() + self.ttl_seconds
                for ticker in missing:
                    # Negative-cache unavailable quotes too, avoiding retry storms.
                    self.entries[ticker] = (expires, payload.get("quotes", {}).get(ticker),
                                            {k: v for k, v in payload.items() if k != "quotes"})
                    self.entries.move_to_end(ticker)
                while len(self.entries) > self.max_entries:
                    self.entries.popitem(last=False)
            quotes = {}
            metadata = {}
            for ticker in tickers:
                _, quote, metadata = self.entries[ticker]
                self.entries.move_to_end(ticker)
                if quote is not None:
                    quotes[ticker] = quote
            return {**metadata, "quotes": quotes}
