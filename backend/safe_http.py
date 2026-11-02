"""Direct, DNS-pinned HTTP transport for untrusted article URLs."""
import http.client
import ipaddress
import socket
import ssl
from functools import partial
from urllib.parse import urlsplit


MAX_ARTICLE_BYTES = 2 * 1024 * 1024


def resolve_public_url(url):
    """Validate the entire DNS answer and return the URL and approved addresses."""
    parsed = urlsplit(url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or "%" in parsed.hostname):
        raise ValueError("Invalid article URL")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    host = parsed.hostname.encode("idna").decode("ascii")
    answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addresses = list(dict.fromkeys(answer[4][0] for answer in answers))
    if not addresses:
        raise ValueError("Article host has no addresses")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global or ip.is_multicast:
            raise ValueError("Article host must resolve only to public addresses")
    return parsed, host, port, addresses


def _connect_pinned(ip, address, timeout=10, source_address=None):
    # Only the numeric address reaches the socket layer. DNS cannot change the
    # destination between validation and connection, even when a retry occurs.
    return socket.create_connection((ip, address[1]), timeout, source_address)


def fetch_public_url(url, headers, timeout=10):
    """Fetch one hop, keeping the hostname for Host, TLS SNI and verification.

    Uses direct sockets (never environment proxies). Redirects are deliberately
    returned to the caller so each destination gets a fresh validation and pin.
    """
    parsed, host, port, addresses = resolve_public_url(url)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    last_error = None
    for address in addresses:
        if parsed.scheme == "https":
            connection = http.client.HTTPSConnection(
                host, port, timeout=timeout, context=ssl.create_default_context(),
            )
        else:
            connection = http.client.HTTPConnection(host, port, timeout=timeout)
        # HTTPConnection.connect uses this hook; HTTPSConnection then wraps the
        # same socket with the original hostname and normal certificate checks.
        connection._create_connection = partial(_connect_pinned, address)
        try:
            connection.request("GET", path, headers=headers)
            response = connection.getresponse()
            status = response.status
            location = response.getheader("Location")
            if status in (301, 302, 303, 307, 308):
                return status, location, b""
            if status >= 400:
                raise ValueError(f"Article returned HTTP {status}")
            body = response.read(MAX_ARTICLE_BYTES + 1)
            if len(body) > MAX_ARTICLE_BYTES:
                raise ValueError("Article exceeds size limit")
            return status, location, body
        except (OSError, http.client.HTTPException) as exc:
            last_error = exc
        finally:
            connection.close()
    raise OSError("Could not connect to article host") from last_error
