"""Exercise the real HTTP connection path with mocked DNS and TCP sockets."""
import io
import socket
import ssl
from unittest.mock import MagicMock, patch

import pytest

import safe_http
from market_data import get_article_content


def dns_answer(ip):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return (family, socket.SOCK_STREAM, 6, "", (ip, 443))


def response_socket(body=b"article", status="200 OK", extra_headers=""):
    transport = MagicMock()
    wire = (f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\n"
            f"{extra_headers}\r\n").encode() + body
    transport.makefile.return_value = io.BytesIO(wire)
    return transport


def test_rebinding_cannot_change_tcp_destination():
    transport = response_socket()
    # A subsequent DNS lookup would return loopback. No such lookup is allowed.
    with patch("safe_http.socket.getaddrinfo", side_effect=[
        [dns_answer("8.8.8.8")], [dns_answer("127.0.0.1")],
    ]) as dns, patch("safe_http.socket.create_connection", return_value=transport) as connect:
        status, _, body = safe_http.fetch_public_url("http://news.example/story?q=1", {})
    assert (status, body) == (200, b"article")
    dns.assert_called_once_with("news.example", 80, type=socket.SOCK_STREAM)
    connect.assert_called_once_with(("8.8.8.8", 80), 10, None)
    sent = b"".join(call.args[0] for call in transport.sendall.call_args_list)
    assert b"GET /story?q=1 HTTP/1.1" in sent
    assert b"Host: news.example\r\n" in sent


def test_https_retains_hostname_for_sni_and_certificate_verification():
    transport = response_socket()
    context = ssl.create_default_context()
    assert context.check_hostname
    assert context.verify_mode == ssl.CERT_REQUIRED
    with patch("safe_http.socket.getaddrinfo", return_value=[dns_answer("8.8.8.8")]), \
            patch("safe_http.socket.create_connection", return_value=transport) as connect, \
            patch("safe_http.ssl.create_default_context", return_value=context), \
            patch.object(context, "wrap_socket", return_value=transport) as tls:
        safe_http.fetch_public_url("https://news.example/story", {})
    connect.assert_called_once_with(("8.8.8.8", 443), 10, None)
    tls.assert_called_once_with(transport, server_hostname="news.example")


@pytest.mark.parametrize("addresses", [
    [], ["127.0.0.1"], ["::1"], ["10.0.0.1"], ["169.254.169.254"],
    ["8.8.8.8", "192.168.1.1"], ["8.8.8.8", "fc00::1"], ["224.0.0.1"],
])
def test_entire_dns_answer_must_be_public(addresses):
    with patch("safe_http.socket.getaddrinfo", return_value=[dns_answer(ip) for ip in addresses]), \
            patch("safe_http.socket.create_connection") as connect:
        with pytest.raises(ValueError):
            safe_http.fetch_public_url("https://news.example/story", {})
    connect.assert_not_called()


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://news.example/story", "https://user:pass@news.example/",
    "http://[fe80::1%25eth0]/", "https://news.example:bad/",
])
def test_invalid_urls_are_rejected_before_dns(url):
    with patch("safe_http.socket.getaddrinfo") as dns:
        with pytest.raises(ValueError):
            safe_http.fetch_public_url(url, {})
    dns.assert_not_called()


def test_redirect_to_private_host_is_rejected_before_second_connection():
    redirect = response_socket(status="302 Found", extra_headers="Location: http://127.0.0.1/private\r\n")
    with patch("safe_http.socket.getaddrinfo", side_effect=[
        [dns_answer("8.8.8.8")], [dns_answer("127.0.0.1")],
    ]), patch("safe_http.socket.create_connection", return_value=redirect) as connect:
        result = get_article_content("http://news.example/story")
    assert "security reasons" in result["content"]
    connect.assert_called_once()


def test_relative_redirect_is_validated_and_pinned_again():
    text = "A public news paragraph. " * 12
    redirect = response_socket(status="302 Found", extra_headers="Location: /next\r\n")
    article = response_socket(body=f"<article><p>{text}</p></article>".encode())
    with patch("safe_http.socket.getaddrinfo", return_value=[dns_answer("8.8.8.8")]) as dns, \
            patch("safe_http.socket.create_connection", side_effect=[redirect, article]) as connect, \
            patch("safe_http.ssl.create_default_context") as context:
        context.return_value.wrap_socket.side_effect = lambda sock, **kwargs: sock
        result = get_article_content("https://news.example/story")
    assert text.strip() in result["content"]
    assert dns.call_count == connect.call_count == 2
    assert all(call.args[0] == ("8.8.8.8", 443) for call in connect.call_args_list)


def test_article_body_is_bounded():
    with patch("safe_http.MAX_ARTICLE_BYTES", 10), \
            patch("safe_http.socket.getaddrinfo", return_value=[dns_answer("8.8.8.8")]), \
            patch("safe_http.socket.create_connection", return_value=response_socket(b"x" * 11)):
        with pytest.raises(ValueError, match="size limit"):
            safe_http.fetch_public_url("http://news.example/", {})


def test_failed_address_retries_only_prevalidated_addresses():
    with patch("safe_http.socket.getaddrinfo", return_value=[dns_answer("8.8.8.8"), dns_answer("1.1.1.1")]) as dns, \
            patch("safe_http.socket.create_connection", side_effect=[OSError(), response_socket()]) as connect:
        assert safe_http.fetch_public_url("http://news.example/", {})[0] == 200
    dns.assert_called_once()
    assert [call.args[0] for call in connect.call_args_list] == [("8.8.8.8", 80), ("1.1.1.1", 80)]
