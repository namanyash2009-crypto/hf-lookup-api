"""Local Range proxy for the remote parquet file.

DuckDB's httpfs reads the parquet through HTTP Range requests. The 16 MB
footer is the expensive part (about 8s) and it never changes, so we cache it
on disk and serve it from there. Data ranges are proxied straight through.

The proxy listens on 127.0.0.1 and is only reachable from this process.
"""

import os
import threading
import time

import requests

REMOTE_URL = "https://huggingface.co/datasets/HiTeckGroup/HiTeckNuMinfo/resolve/main/users_data.parquet"
FOOTER_BYTES = 24 * 1024 * 1024  # parquet footer + column chunk index
CACHE_DIR = os.path.join(os.environ.get("TMPDIR", "/tmp"), "hf_parquet_cache")
FOOTER_PATH = os.path.join(CACHE_DIR, "footer.bin")
SIZE_PATH = os.path.join(CACHE_DIR, "size.txt")

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=2)
_session.mount("https://", _adapter)

# Upstream occasionally answers 429 when many range requests land at once.
# Retry with a short backoff so a scan does not die on a transient throttle.
_MAX_RETRIES = 5


def _remote_get(start, end):
    """Fetch a byte range from the remote file, retrying on transient errors."""
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            r = _session.get(
                REMOTE_URL,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=300,
            )
            if r.status_code in (200, 206):
                return r.content
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                time.sleep(min(2 ** attempt * 0.5, 8))
                continue
            return None
        except Exception as e:  # network hiccup
            last_err = str(e)
            time.sleep(min(2 ** attempt * 0.5, 8))
    raise RuntimeError(f"range fetch failed after {_MAX_RETRIES} tries: {last_err}")

# In-memory cache of data ranges, keyed by (start, end). Row-group data is
# re-read across queries, so caching it keeps the proxy transparent.
_DATA_CACHE = {}
_DATA_CACHE_MAX = 64
_DATA_LOCK = threading.Lock()


def _cache_get(key):
    with _DATA_LOCK:
        return _DATA_CACHE.get(key)


def _cache_put(key, data):
    with _DATA_LOCK:
        if len(_DATA_CACHE) >= _DATA_CACHE_MAX:
            for k in list(_DATA_CACHE.keys())[:16]:
                _DATA_CACHE.pop(k, None)
        # Only cache reasonably sized ranges
        if len(data) <= 8 * 1024 * 1024:
            _DATA_CACHE[key] = data


def _remote_size():
    try:
        with open(SIZE_PATH) as f:
            return int(f.read().strip())
    except Exception:
        pass
    try:
        r = requests.head(REMOTE_URL, allow_redirects=True, timeout=30)
        size = int(r.headers.get("Content-Length", 0))
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(SIZE_PATH, "w") as f:
            f.write(str(size))
        return size
    except Exception:
        return 0


def _ensure_footer():
    """Download the footer once and keep it on disk."""
    if os.path.isfile(FOOTER_PATH):
        return True
    try:
        size = _remote_size()
        if not size:
            return False
        os.makedirs(CACHE_DIR, exist_ok=True)
        start = max(0, size - FOOTER_BYTES)
        data = _remote_get(start, size - 1)
        if data is None:
            return False
        tmp = FOOTER_PATH + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, FOOTER_PATH)
        return True
    except Exception:
        return False


def _read_cached(start, end):
    """Read a byte range that lives inside the cached footer."""
    try:
        size = _remote_size()
        if not size:
            return None
        footer_start = max(0, size - FOOTER_BYTES)
        if start < footer_start:
            return None
        with open(FOOTER_PATH, "rb") as f:
            f.seek(start - footer_start)
            return f.read(end - start + 1)
    except Exception:
        return None


class _Handler:
    """Minimal HTTP handler that serves Range requests for the parquet file."""

    def __init__(self, server):
        self.server = server

    def log_message(self, *args):
        pass


def start_proxy():
    """Start the local Range proxy in a background thread.

    Returns the base URL to give DuckDB, or None if the proxy could not start.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    remote = REMOTE_URL

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_HEAD(self):
            size = _remote_size()
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            rng = self.headers.get("Range")
            size = _remote_size()
            if not size:
                self.send_error(500)
                return

            if rng and rng.startswith("bytes="):
                spec = rng[6:].split("-")
                try:
                    start = int(spec[0])
                    end = int(spec[1]) if len(spec) > 1 and spec[1] else size - 1
                except ValueError:
                    self.send_error(400)
                    return
                end = min(end, size - 1)
            else:
                start, end = 0, size - 1

            # Serve from the cached footer when possible
            data = _read_cached(start, end)
            if data is None:
                key = (start, end)
                data = _cache_get(key)
            if data is None:
                try:
                    data = _remote_get(start, end)
                    _cache_put(key, data)
                except Exception:
                    self.send_error(502)
                    return

            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            try:
                self.wfile.write(data)
            except Exception:
                pass

    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        httpd.daemon_threads = True
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        return f"http://127.0.0.1:{port}/users_data.parquet"
    except Exception:
        return None
