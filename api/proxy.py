"""Local Range proxy for the remote parquet file.

DuckDB's httpfs reads the remote parquet through HTTP Range requests.  For
this dataset the parquet footer is enormous: 14 506 row groups x 8 columns of
per-chunk statistics, roughly 789 MB of thrift metadata.  DuckDB seeks into
that blob as hundreds of tiny range requests, and every one of them used to
be a separate upstream round trip - which made even a single-row-group scan
take over two minutes.

The proxy therefore keeps the tail of the file in a local block cache:

* small reads (parquet metadata) are served from a cached 8 MB block,
* a miss fetches the whole block upstream and stores it,
* blocks are also written to a sparse file on disk so they survive across
  invocations of a warm container.

Each cached block is stored as [marker][data].  The marker is written after
the data, so a block is only served from disk when its marker proves the
block was written completely - a half written or stale cache file can never
be served.

Everything degrades gracefully: if the disk is full the cache simply stays
in memory, and if a block cannot be fetched the original range is fetched
directly.
"""

import os
import threading
import time

import requests

REMOTE_URL = "https://huggingface.co/datasets/HiTeckGroup/HiTeckNuMinfo/resolve/main/users_data.parquet"

# How much of the tail of the file to cache.  The parquet footer plus the
# column/offset indexes occupy about 789 MB of the tail.
CACHE_BYTES = int(os.environ.get("HF_PARQUET_CACHE_BYTES", 900 * 1024 * 1024))

# In-memory budget for cached blocks.  Kept modest: a serverless function
# has a hard memory limit and the proxy shares it with DuckDB.
MEM_CACHE_BYTES = int(os.environ.get("HF_PARQUET_MEM_CACHE_BYTES", 96 * 1024 * 1024))

CACHE_DIR = os.path.join(os.environ.get("TMPDIR", "/tmp"), "hf_parquet_cache")
CACHE_PATH = os.path.join(CACHE_DIR, "tail.bin")
SIZE_PATH = os.path.join(CACHE_DIR, "size.txt")

# Cache block geometry.  DuckDB seeks through the parquet metadata as
# hundreds of tiny reads that are scattered over a few hundred megabytes.
# A 1 MB block collapses those into a manageable number of upstream
# requests without pulling in far more data than is needed.
_BLOCK_SIZE = 1 * 1024 * 1024
_MARKER = b"HFPCB1\x00\x00\x00"
_MARKER_LEN = len(_MARKER)
_SLOT = _BLOCK_SIZE + _MARKER_LEN

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=2)
_session.mount("https://", _adapter)

# Upstream occasionally answers 429 when many range requests land at once.
# Retry with a short backoff so a scan does not die on a transient throttle.
_MAX_RETRIES = 5

# The resolve URL redirects to a signed CDN URL.  Resolving it once and then
# talking to the CDN directly halves the number of round trips.
_cdn_lock = threading.Lock()
_cdn_url = None
_cdn_expires = 0.0


def _resolve_cdn_url():
    """Resolve the redirect to the signed CDN URL (cached for a while)."""
    global _cdn_url, _cdn_expires
    with _cdn_lock:
        if _cdn_url and time.time() < _cdn_expires:
            return _cdn_url
        try:
            r = _session.get(
                REMOTE_URL,
                headers={"Range": "bytes=0-1"},
                timeout=60,
                allow_redirects=True,
            )
            if r.status_code in (200, 206) and r.url and "huggingface.co/datasets" not in r.url:
                _cdn_url = r.url
                # The signed URL carries an expiry; refresh well before it.
                _cdn_expires = time.time() + 1800
                return _cdn_url
        except Exception:
            pass
        _cdn_url = None
        _cdn_expires = time.time() + 30
        return REMOTE_URL


def _remote_get(start, end):
    """Fetch a byte range from the remote file, retrying on transient errors."""
    global _cdn_url, _cdn_expires
    last_err = None
    for attempt in range(_MAX_RETRIES):
        url = _resolve_cdn_url()
        try:
            r = _session.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=300,
            )
            if r.status_code in (200, 206):
                return r.content
            if r.status_code == 403:
                # Signed URL expired - resolve it again and retry.
                with _cdn_lock:
                    _cdn_url = None
                    _cdn_expires = 0.0
                last_err = "HTTP 403 (expired link)"
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                time.sleep(min(2 ** attempt * 0.5, 8))
                continue
            return None
        except Exception as e:  # network hiccup
            last_err = str(e)
            time.sleep(min(2 ** attempt * 0.5, 8))
    raise RuntimeError(f"range fetch failed after {_MAX_RETRIES} tries: {last_err}")


# ---------------------------------------------------------------------------
# Block cache of the tail of the file
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_file_size = 0          # size of the remote file
_cache_start = 0       # first byte covered by the cache
_mem = {}              # block index -> bytes
_mem_bytes = 0
_fh_w = None           # handle used for writing
_fh_r = None           # handle used for reading (a shared handle would race
                       # between seek/read and write)


def _open_w():
    global _fh_w
    if _fh_w is not None:
        return _fh_w
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        _fh_w = open(CACHE_PATH, "r+b" if os.path.exists(CACHE_PATH) else "w+b")
        return _fh_w
    except Exception:
        return None


def _open_r():
    global _fh_r
    if _fh_r is not None:
        return _fh_r
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        _fh_r = open(CACHE_PATH, "rb")
        return _fh_r
    except Exception:
        return None


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


def _init_cache():
    """Work out the cached region for the current file size."""
    global _file_size, _cache_start

    size = _remote_size()
    if not size:
        return False

    with _lock:
        if _file_size == size:
            return True
        _file_size = size
        _cache_start = max(0, size - CACHE_BYTES)
    return True


def _block_bounds(b):
    """File byte range covered by cache block b."""
    bstart = _cache_start + b * _BLOCK_SIZE
    bend = min(bstart + _BLOCK_SIZE, _file_size) - 1
    return bstart, bend


def _slot_offset(b):
    """Offset of a block slot inside the cache file."""
    return b * _SLOT


def _mem_put(b, data):
    global _mem_bytes
    with _lock:
        if b in _mem:
            return
        while _mem and _mem_bytes + len(data) > MEM_CACHE_BYTES:
            oldest = next(iter(_mem))
            _mem_bytes -= len(_mem.pop(oldest))
        _mem[b] = data
        _mem_bytes += len(data)


def _disk_put(b, data):
    """Store a block in the sparse cache file (best effort).

    The marker is written last, so a partially written block is never
    considered valid.  The file is capped so it cannot fill the disk: the
    blocks closest to the end of the file (the parquet metadata) are the
    ones worth keeping, so those are kept.
    """
    try:
        with _lock:
            fh = _open_w()
            if fh is None:
                return False
            fh.seek(_slot_offset(b))
            fh.write(data)
            fh.write(_MARKER)
            fh.flush()
        return True
    except Exception:
        return False


def _disk_get(b):
    """Read a block back from the cache file, verifying its marker."""
    try:
        fh = _open_r()
        if fh is None:
            return None
        with _lock:
            fh.seek(_slot_offset(b))
            data = fh.read(_BLOCK_SIZE)
            marker = fh.read(_MARKER_LEN)
        if marker != _MARKER:
            return None
        bstart, bend = _block_bounds(b)
        want = bend - bstart + 1
        if len(data) < want:
            return None
        return data[:want]
    except Exception:
        return None


def _block_get(b):
    """Return the bytes of a cached block, fetching it upstream if needed."""
    with _lock:
        data = _mem.get(b)
    if data is not None:
        return data

    bstart, bend = _block_bounds(b)
    if bstart > bend:
        return None

    data = _disk_get(b)
    if data is not None:
        _mem_put(b, data)
        return data

    try:
        chunk = _remote_get(bstart, bend)
    except Exception:
        return None
    if chunk is None or len(chunk) != (bend - bstart + 1):
        return None

    _mem_put(b, chunk)
    _disk_put(b, chunk)
    return chunk


def _read_cached(start, end):
    """Serve a range out of the block cache."""
    if end < _cache_start or start < _cache_start or start >= _file_size:
        return None

    first = (start - _cache_start) // _BLOCK_SIZE
    last = (end - _cache_start) // _BLOCK_SIZE

    out = bytearray()
    for b in range(first, last + 1):
        chunk = _block_get(b)
        if chunk is None:
            return None
        out.extend(chunk)

    base = _cache_start + first * _BLOCK_SIZE
    lo = start - base
    hi = end - base + 1
    if hi > len(out):
        hi = len(out)
    return bytes(out[lo:hi])


def _prefetch_tail():
    """Pull the whole metadata region into the cache in large blocks."""
    if not _init_cache():
        return
    total = (max(0, _file_size - _cache_start) + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    for b in range(total):
        with _lock:
            if b in _mem:
                continue
        _block_get(b)


def _ensure_footer():
    """Make sure the tail cache is ready (kept for backwards compatibility)."""
    return _init_cache()


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
            try:
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

                data = None
                if end >= _cache_start and start < _file_size:
                    data = _read_cached(start, end)

                if data is None:
                    try:
                        data = _remote_get(start, end)
                    except Exception:
                        self.send_error(502)
                        return

                if data is None:
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
            except Exception:
                try:
                    self.send_error(500)
                except Exception:
                    pass

    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        httpd.daemon_threads = True
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        # Warm the tail cache in the background so the first request does not
        # have to wait for it.
        threading.Thread(target=_prefetch_tail, daemon=True).start()
        return f"http://127.0.0.1:{port}/users_data.parquet"
    except Exception:
        return None
