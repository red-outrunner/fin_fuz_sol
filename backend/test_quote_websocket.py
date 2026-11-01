import asyncio
import threading
from unittest.mock import AsyncMock, patch

from main import ws_quotes


def test_quotes_run_off_event_loop_and_disconnect_stops_polling():
    loop_thread = threading.get_ident()
    socket = AsyncMock()
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
    socket = AsyncMock()
    socket.query_params = {"tickers": "AAA.JO"}
    socket.send_json.side_effect = OSError()
    with patch("main.get_live_quotes", return_value={}), patch("main.logger") as logger:
        asyncio.run(ws_quotes(socket))
    logger.exception.assert_not_called()


def test_provider_failure_keeps_traceback_for_diagnosis():
    socket = AsyncMock()
    socket.query_params = {"tickers": "AAA.JO"}
    with patch("main.get_live_quotes", side_effect=ValueError("provider failed")), patch("main.logger") as logger:
        asyncio.run(ws_quotes(socket))
    logger.exception.assert_called_once_with("Quote WebSocket failed")
    socket.close.assert_awaited_once()
