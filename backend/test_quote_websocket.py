import asyncio
import threading
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
from contextlib import ExitStack

import pytest
import main
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from quote_stream import QuoteConnectionLimiter, SharedQuoteCache

from main import ws_quotes


@pytest.fixture(autouse=True)
def fresh_quote_budgets(monkeypatch):
    monkeypatch.setattr(main, "quote_cache", SharedQuoteCache())
    monkeypatch.setattr(main, "quote_connections", QuoteConnectionLimiter())


def mock_socket(tickers="AAA.JO", ip="203.0.113.10"):
    socket = AsyncMock()
    socket.client = SimpleNamespace(host=ip)
    socket.query_params = {"tickers": tickers}
    socket.receive.return_value = {"type": "websocket.disconnect", "code": 1000}
    return socket


def test_quotes_run_off_event_loop_and_disconnect_stops_polling():
    loop_thread = threading.get_ident()
    socket = mock_socket()
    socket.query_params = {"tickers": "AAA.JO,AAA.JO,BBB.JO"}
    socket.receive.return_value = {"type": "websocket.disconnect", "code": 1000}
    def provider(tickers):
        assert threading.get_ident() != loop_thread
        assert tickers == ["AAA.JO", "BBB.JO"]
        return {"quotes": {}}
    with patch("main.get_live_quotes", side_effect=provider) as fetch:
        asyncio.run(ws_quotes(socket))
    fetch.assert_called_once()
    socket.send_json.assert_awaited_once_with({"quotes": {}})


def test_transport_disconnect_is_not_logged_as_server_error():
    socket = mock_socket()
    socket.query_params = {"tickers": "AAA.JO"}
    socket.send_json.side_effect = OSError()
    with patch("main.get_live_quotes", return_value={}), patch("main.logger") as logger:
        asyncio.run(ws_quotes(socket))
    logger.exception.assert_not_called()


def test_provider_failure_keeps_traceback_for_diagnosis():
    socket = mock_socket()
    socket.query_params = {"tickers": "AAA.JO"}
    with patch("main.get_live_quotes", side_effect=ValueError("provider failed")), patch("main.logger") as logger:
        asyncio.run(ws_quotes(socket))
    logger.exception.assert_called_once_with("Quote WebSocket failed")
    socket.close.assert_awaited_once()


def test_excess_connection_is_rejected_without_accept_or_provider_work():
    limiter = QuoteConnectionLimiter(max_per_ip=1)
    assert limiter.acquire("203.0.113.10")
    socket = mock_socket()
    with patch("main.quote_connections", limiter), patch("main.get_live_quotes") as fetch:
        asyncio.run(ws_quotes(socket))
    socket.accept.assert_not_awaited()
    socket.close.assert_awaited_once_with(code=1013, reason="Quote connection limit exceeded")
    fetch.assert_not_called()
    assert limiter.active["203.0.113.10"] == 1


@pytest.mark.parametrize("failure", [None, OSError("transport"), ValueError("provider")])
def test_connection_slot_is_released_on_disconnect_or_failure(failure):
    socket = mock_socket()
    with patch("main.get_live_quotes", return_value={"quotes": {}}, side_effect=failure):
        asyncio.run(ws_quotes(socket))
    assert not main.quote_connections.active


def test_rest_and_websocket_subscribers_share_quotes():
    socket = mock_socket()
    payload = {"provider": "test", "quotes": {"AAA.JO": {"price": 100}}}
    with patch("main.get_live_quotes", return_value=payload) as fetch:
        assert main.live_quotes.__wrapped__(None, main.QuotesRequest(tickers=["AAA.JO"])) == payload
        asyncio.run(ws_quotes(socket))
        asyncio.run(ws_quotes(mock_socket()))
    fetch.assert_called_once_with(["AAA.JO"])
    socket.send_json.assert_awaited_once_with(payload)


def test_real_websocket_handshakes_enforce_ip_limit_and_share_payloads():
    client = TestClient(main.app)
    payload = {"provider": "test", "quotes": {"AAA.JO": {"price": 100}}}
    with patch("main.get_live_quotes", return_value=payload) as fetch:
        with ExitStack() as stack:
            for _ in range(3):
                channel = stack.enter_context(client.websocket_connect("/ws/quotes?tickers=AAA.JO"))
                assert channel.receive_json() == payload
            with pytest.raises(WebSocketDisconnect) as rejection:
                with client.websocket_connect("/ws/quotes?tickers=AAA.JO"):
                    pytest.fail("Excess connection was accepted")
            assert rejection.value.code == 1013
        fetch.assert_called_once_with(["AAA.JO"])
    assert not main.quote_connections.active
