from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

from quote_stream import QuoteConnectionLimiter, SharedQuoteCache


def test_connection_limits_and_reconnect_throttle_expire():
    now = [0]
    limiter = QuoteConnectionLimiter(max_connections=2, max_per_ip=1,
                                    starts_per_minute=2, global_starts_per_minute=3,
                                    clock=lambda: now[0])
    assert limiter.acquire("a")
    assert not limiter.acquire("a")
    assert limiter.acquire("b")
    assert not limiter.acquire("c")
    limiter.release("a")
    assert limiter.acquire("a")
    limiter.release("a")
    assert not limiter.acquire("a")
    assert not limiter.acquire("c")
    now[0] = 60
    assert limiter.acquire("a")
    assert len(limiter.starts) == 1
    limiter.release("a")
    limiter.release("b")
    assert not limiter.active


def test_overlapping_requests_fetch_only_missing_symbols_and_expire():
    now = [0]
    cache = SharedQuoteCache(clock=lambda: now[0])
    fetch = Mock(side_effect=lambda tickers: {
        "provider": "test", "quotes": {t: {"price": 100} for t in tickers},
    })
    cache.get_quotes(["a", "A", "B"], fetch)
    result = cache.get_quotes(["B", "C"], fetch)
    assert set(result["quotes"]) == {"B", "C"}
    assert [call.args[0] for call in fetch.call_args_list] == [["A", "B"], ["C"]]
    now[0] = 5
    cache.get_quotes(["A", "B"], fetch)
    assert fetch.call_args.args[0] == ["A", "B"]


def test_concurrent_subscribers_do_not_duplicate_provider_calls():
    cache = SharedQuoteCache()
    entered, release = Event(), Event()

    def provider(tickers):
        entered.set()
        assert release.wait(5)
        return {"quotes": {"A": {"price": 100}}}

    fetch = Mock(side_effect=provider)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.get_quotes, ["A"], fetch)
        try:
            assert entered.wait(5)
            second = pool.submit(cache.get_quotes, ["A"], fetch)
        finally:
            release.set()
        assert first.result(timeout=5) == second.result(timeout=5)
    fetch.assert_called_once_with(["A"])


def test_missing_quotes_are_cached_and_entries_remain_bounded():
    cache = SharedQuoteCache(max_entries=30)
    fetch = Mock(return_value={"provider": "test", "quotes": {}})
    assert cache.get_quotes(["A"], fetch)["quotes"] == {}
    cache.get_quotes(["A"], fetch)
    fetch.assert_called_once()
    cache.get_quotes([str(i) for i in range(30)], fetch)
    # Preserve a hit at the oldest end while new symbols force eviction.
    cache.get_quotes(["0", "new"], fetch)
    assert "0" in cache.entries and "NEW" in cache.entries
    assert len(cache.entries) == 30
