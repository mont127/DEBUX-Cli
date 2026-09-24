"""Web search, URL fetching and file ingest.

The fetcher is deliberately careful. Debux runs on a machine that usually sits inside the network
someone is debugging, and the model chooses the URLs. A model persuaded by a search result to fetch
http://169.254.169.254/ would be reading cloud credentials, so the guard here is not decoration:
scheme, embedded credentials, resolved address and every redirect hop are all checked, and the
connection is pinned to the address that was validated so a DNS rebind between check and connect
cannot redirect the fetch inward.
"""
import ipaddress
import os
import re
import http.client
import socket
import ssl
import urllib.parse
from urllib.parse import urlparse, urlunparse

MAX_BYTES = 512 * 1024
TIMEOUT = 20
MAX_REDIRECTS = 4
UA = "debux-cli/1.0"


class FetchError(Exception):
    pass


def _addresses(host):
    try:
        info = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise FetchError(f"cannot resolve {host}: {e}")
    return [ipaddress.ip_address(i[4][0]) for i in info]


def _check_public(host):
    """Every address the host resolves to must be globally routable. Returns one validated IP."""
    addrs = _addresses(host)
    for a in addrs:
        if a.is_private or a.is_loopback or a.is_link_local or a.is_reserved or a.is_multicast:
            raise FetchError(f"refusing {host}: resolves to non-public address {a}")
        # IPv4-mapped IPv6 hides a private v4 address behind a v6 literal
        if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped:
            v4 = a.ipv4_mapped
            if v4.is_private or v4.is_loopback or v4.is_link_local:
                raise FetchError(f"refusing {host}: IPv4-mapped private address {v4}")
    if not addrs:
        raise FetchError(f"no addresses for {host}")
    return addrs[0]


def _validate(url):
    u = urlparse(url)
    if u.scheme not in ("http", "https"):
        raise FetchError(f"refusing {u.scheme or 'empty'} URL; only http and https are fetched")
    if u.username or u.password:
        raise FetchError("refusing URL with embedded credentials")
    if not u.hostname:
        raise FetchError("URL has no host")
    ip = _check_public(u.hostname)
    return u, ip


class _Pinned(http.client.HTTPSConnection):
    """Connect to a pre-validated address, but keep the hostname for SNI and cert verification.

    Pinning the socket to the address that passed the check is what stops a DNS rebind between
    validation and connect. Naively putting the IP in the URL would also send the IP as SNI, which
    most servers reject outright.
    """

    def __init__(self, host, ip, port, context, timeout):
        super().__init__(host, port=port, context=context, timeout=timeout)
        self._ip = ip

    def connect(self):
        self.sock = socket.create_connection((self._ip, self.port), self.timeout)
        if self._tunnel_host:
            self._tunnel()
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


class _PinnedPlain(http.client.HTTPConnection):
    def __init__(self, host, ip, port, timeout):
        super().__init__(host, port=port, timeout=timeout)
        self._ip = ip

    def connect(self):
        self.sock = socket.create_connection((self._ip, self.port), self.timeout)


def fetch(url, max_bytes=MAX_BYTES):
    """Fetch a URL as text. Redirects are followed manually so every hop is re-validated."""
    hops = 0
    while True:
        u, ip = _validate(url)
        port = u.port or (443 if u.scheme == "https" else 80)
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        ipstr = str(ip)
        try:
            if u.scheme == "https":
                conn = _Pinned(u.hostname, ipstr, port, ssl.create_default_context(), TIMEOUT)
            else:
                conn = _PinnedPlain(u.hostname, ipstr, port, TIMEOUT)
            conn.request("GET", path, headers={
                "Host": u.hostname, "User-Agent": UA, "Connection": "close",
                "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.5",
            })
            resp = conn.getresponse()
            if resp.status in (301, 302, 303, 307, 308):
                loc = resp.headers.get("Location")
                conn.close()
                hops += 1
                if not loc or hops > MAX_REDIRECTS:
                    raise FetchError("too many redirects")
                url = urllib.parse.urljoin(url, loc)
                continue
            if resp.status >= 400:
                conn.close()
                raise FetchError(f"HTTP {resp.status} {resp.reason}")
            ctype = resp.headers.get("Content-Type", "")
            body = resp.read(max_bytes + 1)
            conn.close()
        except ssl.SSLCertVerificationError as e:
            raise FetchError(f"TLS verification failed: {e}")
        except ssl.SSLError as e:
            raise FetchError(f"TLS error: {e}")
        except socket.timeout:
            raise FetchError(f"timed out after {TIMEOUT}s")
        except OSError as e:
            raise FetchError(f"{e}")
        break

    truncated = len(body) > max_bytes
    text = body[:max_bytes].decode("utf-8", "replace")
    if "html" in ctype.lower():
        text = _strip_html(text)
    if truncated:
        text += "\n[... truncated by debux ...]"
    return text


_TAG = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_ANY = re.compile(r"<[^>]+>")


def _strip_html(html):
    t = _TAG.sub(" ", html)
    t = _ANY.sub(" ", t)
    t = (t.replace("&nbsp;", " ").replace("&amp;", "&")
          .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"'))
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", t)).strip()


def search(q, n=5):
    """DuckDuckGo. A failure is reported as a failure.

    Returning an empty string on error would let the model read 'no results' as a negative answer,
    which is how a search tool turns into a source of confident wrong statements.
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        return "SEARCH UNAVAILABLE: no DuckDuckGo client installed (pip install ddgs)."
    try:
        with DDGS() as d:
            hits = list(d.text(q, max_results=n))
    except Exception as e:
        return f"SEARCH FAILED: {type(e).__name__}: {e}"
    if not hits:
        return "SEARCH RETURNED NO RESULTS. Treat this as no evidence, not as a negative answer."
    return "\n".join(
        f"[{i}] {h.get('title','')}\n    {h.get('href','')}\n    {h.get('body','')}"
        for i, h in enumerate(hits, 1))


# Null bytes disqualify a file whatever it is called. Extension-based allow-listing let
# /bin/ls through, because most binaries have no extension at all.


def read_file(path, max_bytes=256 * 1024, tail_lines=400):
    """Read a file for the model. Long logs are tailed, since the end is where the failure is."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FetchError(f"no such file: {path}")
    if os.path.isdir(path):
        raise FetchError(f"{path} is a directory")
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        raw = fh.read()
    if b"\x00" in raw[:4096]:
        raise FetchError(f"{path} looks binary; paste the relevant text instead")
    text = raw.decode("utf-8", "replace")
    lines = text.splitlines()
    note = ""
    if len(lines) > tail_lines:
        lines = lines[-tail_lines:]
        note = f"[last {tail_lines} lines of {path}, {size} bytes total]\n"
    elif size > max_bytes:
        note = f"[tail of {path}, {size} bytes total]\n"
    else:
        note = f"[{path}, {size} bytes]\n"
    return note + "\n".join(lines)
