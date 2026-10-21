from flask import Flask, request, jsonify
import os
import json
import threading
import time
import requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

CREDITS = "https://t.me/NamanYadavBots"


def _build_meta(cached=False, cache_hits=0, latency_ms=0):
    """Standard meta block attached to every response."""
    return {
        "credits": CREDITS,
        "cached": bool(cached),
        "cache_hits": int(cache_hits),
        "latency_ms": int(latency_ms),
        "timestamp": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
    }

HF_DATASET = "HiTeckGroup/HiTeckNuMinfo"
HF_PARQUET_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/users_data.parquet"
HF_ROWS_URL = f"https://datasets-server.huggingface.co/rows?dataset={HF_DATASET}&config=default&split=train"


def _start_local_proxy():
    """Serve the remote parquet through a local Range proxy with a cached footer.

    The parquet footer is ~16 MB and costs about 7s to fetch, but it never
    changes. Caching it on disk and serving it locally removes that cost
    from every request on a warm container.
    """
    try:
        from api.proxy import start_proxy

        return start_proxy()
    except Exception:
        return None


# Prefer the local proxy URL when it is available.
PROXY_URL = _start_local_proxy()
if PROXY_URL:
    HF_PARQUET_URL = PROXY_URL

# Load pre-computed row group index at cold start
REMOTE_RGS = []
for _p in (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "rg_index.json"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rg_index.json"),
    "/var/task/rg_index.json",
    os.path.join(os.getcwd(), "rg_index.json"),
):
    try:
        if os.path.isfile(_p):
            with open(_p, "r", encoding="utf-8") as f:
                REMOTE_RGS = json.load(f)
            break
    except Exception:
        pass

# Optional accurate index: adds the true first-row (highest) value of every
# row group, which lets us rank candidates far more precisely.
ACC_RGS = []
for _p in (
    os.path.join(os.path.dirname(__file__), "acc_index.json"),
    os.path.join(os.path.dirname(__file__), "..", "acc_index.json"),
    "/var/task/acc_index.json",
    os.path.join(os.getcwd(), "acc_index.json"),
):
    try:
        if os.path.isfile(_p):
            with open(_p, "r", encoding="utf-8") as f:
                ACC_RGS = json.load(f)
            break
    except Exception:
        pass

# Optional fine index: 16 sample points inside every row group, which allows
# pinpointing the exact segment a number falls into.
FINE_RGS = []
for _p in (
    os.path.join(os.path.dirname(__file__), "fine_index.json"),
    os.path.join(os.path.dirname(__file__), "..", "fine_index.json"),
    "/var/task/fine_index.json",
    os.path.join(os.getcwd(), "fine_index.json"),
):
    try:
        if os.path.isfile(_p):
            with open(_p, "r", encoding="utf-8") as f:
                FINE_RGS = json.load(f)
            break
    except Exception:
        pass

# DuckDB connection (optional - HTTP fallback used if unavailable)
CON = None
DUCKDB_ERROR = None
DUCKDB_AVAILABLE = False


def _make_duckdb_connection():
    """Create a DuckDB connection configured for remote HTTP parquet scans."""
    import duckdb

    conn = duckdb.connect()

    # Vercel has no HOME env var - DuckDB needs home_directory + extension_directory
    # pointed at a writable location (/tmp) to install/load the httpfs extension.
    _tmp = os.environ.get("TMPDIR", "/tmp")
    try:
        os.makedirs(_tmp, exist_ok=True)
        conn.execute(f"SET home_directory='{_tmp}';")
    except Exception:
        pass

    _extdir = os.path.join(_tmp, "duckdb_ext")
    try:
        os.makedirs(_extdir, exist_ok=True)
        conn.execute(f"SET extension_directory='{_extdir}';")
    except Exception:
        pass

    try:
        conn.execute("LOAD httpfs;")
    except Exception:
        # httpfs is not bundled in the pip wheel - download it once into /tmp
        conn.execute("INSTALL httpfs;")
        conn.execute("LOAD httpfs;")

    conn.execute("SET enable_http_metadata_cache = true;")
    conn.execute("SET http_timeout = 60000;")
    conn.execute("SET threads = 4;")
    return conn


try:
    import duckdb  # noqa: F401

    CON = _make_duckdb_connection()
    DUCKDB_AVAILABLE = True
except Exception as _e:
    CON = None
    DUCKDB_ERROR = f"{type(_e).__name__}: {_e}"

LAST_ERROR = None

# A DuckDB connection is not safe to use from two threads at once. Every
# query therefore runs on its own cursor (a separate connection to the same
# database, so the parquet metadata cache is still shared).
_DUCKDB_LOCK = threading.RLock()


def _warmup_metadata():
    """Prefetch the remote parquet footer in the background.

    The footer is ~16 MB and costs about 7s to download on a cold container.
    Fetching it on a background thread means the first live request does not
    have to pay that cost - by the time it arrives the cache is already warm.
    """
    if CON is None:
        return
    try:
        with _DUCKDB_LOCK:
            cur = CON.cursor()
        try:
            cur.execute(
                "SELECT count(*) FROM parquet_metadata('" + HF_PARQUET_URL + "') "
                "WHERE path_in_schema = 'mobile'"
            ).fetchall()
        finally:
            try:
                cur.close()
            except Exception:
                pass
    except Exception:
        pass


def _warmup_footer_cache():
    """Download the parquet footer to the local disk cache in the background."""
    try:
        from api.proxy import _ensure_footer

        _ensure_footer()
    except Exception:
        pass


if DUCKDB_AVAILABLE:
    try:
        threading.Thread(target=_warmup_metadata, daemon=True).start()
        threading.Thread(target=_warmup_footer_cache, daemon=True).start()
    except Exception:
        pass

# Vercel Hobby supports up to 300s with Fluid compute
TIME_BUDGET = 270.0

# Maximum number of linked records to trace (main number + 4 alt hops)
MAX_STEPS = 5

# In-memory LRU cache of resolved records (survives across warm invocations)
_CACHE = {}
_CACHE_MAX = 2000
CACHE_HITS = 0
CACHE_MISSES = 0

# Persistent disk cache in /tmp - survives container restarts (Vercel Fluid compute
# keeps the instance alive, and /tmp persists between warm invocations).
DISK_CACHE_PATH = os.path.join(os.environ.get("TMPDIR", "/tmp"), "hf_lookup_cache.json")
DISK_CACHE = {}
try:
    if os.path.isfile(DISK_CACHE_PATH):
        with open(DISK_CACHE_PATH, "r", encoding="utf-8") as f:
            DISK_CACHE = json.load(f)
except Exception:
    DISK_CACHE = {}

DISK_CACHE_DIRTY = False


def _disk_cache_save():
    """Persist the disk cache (best effort, non-blocking)."""
    global DISK_CACHE_DIRTY
    if not DISK_CACHE_DIRTY:
        return
    DISK_CACHE_DIRTY = False
    try:
        tmp_path = DISK_CACHE_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(DISK_CACHE, f)
        os.replace(tmp_path, DISK_CACHE_PATH)
    except Exception:
        pass

EMPTY_VALUES = {"null", "none", "", "nulllll", "-", "0", "na", "n/a", "nil", "undefined"}


def _cache_put(key, record):
    """Store a resolved record in both memory and disk caches."""
    global DISK_CACHE_DIRTY
    if len(_CACHE) >= _CACHE_MAX:
        # Evict oldest entry (dicts preserve insertion order)
        try:
            _CACHE.pop(next(iter(_CACHE)))
        except StopIteration:
            pass
    _CACHE[key] = record

    DISK_CACHE[key] = record
    if len(DISK_CACHE) > 5000:
        # keep disk cache bounded
        for k in list(DISK_CACHE.keys())[:1000]:
            DISK_CACHE.pop(k, None)
    DISK_CACHE_DIRTY = True


def clean_field(value):
    """Normalise a dataset value - returns None for empty/placeholder values."""
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in EMPTY_VALUES:
        return None
    return s


# Build descending chunks (independent sorted runs). The accurate index gives
# the true first-row value of every row group, so the chunk boundaries are
# reliable; the parquet statistics alone are stale for about 11% of them.
CHUNKS = []
if ACC_RGS:
    _start = 0
    for _i in range(1, len(ACC_RGS)):
        if ACC_RGS[_i]["f"] > ACC_RGS[_i - 1]["f"]:
            CHUNKS.append((_start, _i - 1))
            _start = _i
    CHUNKS.append((_start, len(ACC_RGS) - 1))
else:
    _start = 0
    for _i in range(1, len(REMOTE_RGS)):
        if REMOTE_RGS[_i]["m"] > REMOTE_RGS[_i - 1]["m"]:
            CHUNKS.append((_start, _i - 1))
            _start = _i
    if REMOTE_RGS:
        CHUNKS.append((_start, len(REMOTE_RGS) - 1))


def _order_by_span(val):
    """Rank row groups by how tightly their range brackets the number."""
    hits = []
    for i, r in enumerate(ACC_RGS):
        lo, hi = min(r["m"], r["f"]), max(r["m"], r["f"])
        if lo <= val <= hi:
            hits.append((hi - lo, abs((lo + hi) // 2 - val), i))
    hits.sort()
    return [h[2] for h in hits]


def _order_by_chunk(val):
    """Rank row groups by binary-searching the descending chunks.

    Each chunk is a monotonic run of row groups, so the number's position
    inside the dataset can be narrowed down with a binary search.
    """
    results = []
    for cs, ce in CHUNKS:
        if not (ACC_RGS[ce]["f"] <= val <= ACC_RGS[cs]["f"]):
            continue
        lo, hi = cs, ce
        pos = cs
        while lo <= hi:
            mid = (lo + hi) // 2
            r = ACC_RGS[mid]
            rlo, rhi = min(r["m"], r["f"]), max(r["m"], r["f"])
            if val > rhi:
                hi = mid - 1
            elif val < rlo:
                lo = mid + 1
            else:
                pos = mid
                break
        for d in (-1, 0, 1):
            i = pos + d
            if cs <= i <= ce:
                r = ACC_RGS[i]
                rlo, rhi = min(r["m"], r["f"]), max(r["m"], r["f"])
                if rlo <= val <= rhi:
                    results.append(i)
    return results


def find_row_groups_by_range(phone_digits):
    """Locate candidate row groups for a phone number.

    Parquet statistics are approximate, so several row groups can claim the
    same number. Two independent rankings are merged: one by bracket
    tightness and one by binary-searching the descending chunks. Together
    they put the true owner in the first couple of candidates.
    """
    val = int(phone_digits)

    if FINE_RGS:
        # Fine index: sample points inside each row group pinpoint the exact
        # segment a number falls into, so the true owner is almost always
        # the first candidate.
        hits = []
        for r in FINE_RGS:
            pts = r.get("s") or []
            if len(pts) < 2:
                continue
            vals = [p[1] for p in pts]
            lo, hi = min(vals), max(vals)
            if not (lo <= val <= hi):
                continue
            # Prefer the row group whose samples bracket the number most tightly
            hits.append((hi - lo, abs((lo + hi) // 2 - val), r))
        if hits:
            hits.sort(key=lambda t: (t[0], t[1]))
            return [h[2] for h in hits]

    if ACC_RGS:
        # Merge the two rankings so the true owner lands near the front.
        a = _order_by_span(val)[:12]
        b = _order_by_chunk(val)[:12]
        merged = []
        seen = set()
        for i in range(max(len(a), len(b))):
            for src in (a, b):
                if i < len(src) and src[i] not in seen:
                    seen.add(src[i])
                    merged.append(src[i])
        if merged:
            return [ACC_RGS[i] for i in merged]

        # Number not claimed by any row group - use the closest ones.
        closest = sorted(
            ACC_RGS,
            key=lambda r: min(abs(r["m"] - val), abs(r["f"] - val)),
        )[:8]
        return closest

    cands = [r for r in REMOTE_RGS if r["m"] <= val <= r["M"]]
    if not cands:
        # No row group claims it - use the numerically closest ones
        cands = sorted(
            REMOTE_RGS,
            key=lambda r: min(abs(r["m"] - val), abs(r["M"] - val)),
        )[:8]
    else:
        # Tightest bracket first - the true owner is nearly always here
        cands.sort(key=lambda r: (r["M"] - r["m"], abs((r["m"] + r["M"]) // 2 - val)))
    return cands


def build_record(row):
    """Build a full profile record from a raw dataset row."""
    return {
        "mobile": clean_field(row[0]),
        "name": clean_field(row[1]),
        "fname": clean_field(row[2]),
        "address": clean_field(row[3]),
        "alt": clean_field(row[4]),
        "circle": clean_field(row[5]),
    }


def duckdb_find_row(rg, phone_digits, conn=None):
    """Phase 1: locate the row number for a number inside a row group.

    Only the `mobile` column is read here, which is roughly twice as fast
    as pulling every column.
    """
    c = conn if conn is not None else CON
    if c is None:
        return None

    # With the fine index we know which segment of the row group the number
    # lives in, so we only have to scan that slice of rows.
    lo_off, hi_off = rg["o"], rg["o"] + rg["r"]
    pts = rg.get("s") if isinstance(rg, dict) else None
    if pts and len(pts) >= 2:
        vals = [p[1] for p in pts]
        seg = None
        for j in range(len(pts) - 1):
            a, b = pts[j][1], pts[j + 1][1]
            if min(a, b) <= int(phone_digits) <= max(a, b):
                seg = j
                break
        if seg is not None:
            start = pts[seg][0]
            end = pts[seg + 1][0] if seg + 1 < len(pts) else rg["r"]
            # Widen slightly - the local ordering is not perfectly sorted
            pad = (end - start) // 4
            lo_off = rg["o"] + max(0, start - pad)
            hi_off = rg["o"] + min(rg["r"], end + pad)

    sql = f"""
    SELECT file_row_number
    FROM parquet_scan('{HF_PARQUET_URL}', file_row_number=true)
    WHERE file_row_number >= {lo_off}
      AND file_row_number < {hi_off}
      AND mobile = '{phone_digits}'
    LIMIT 1
    """
    try:
        if conn is None:
            with _DUCKDB_LOCK:
                rows = c.execute(sql).fetchall()
        else:
            rows = c.execute(sql).fetchall()
        if rows:
            return rows[0][0]
    except Exception as _e:
        global LAST_ERROR
        LAST_ERROR = f"{type(_e).__name__}: {_e}"
    return None


def duckdb_fetch_row(row_number, conn=None):
    """Phase 2: fetch the full record for a known row number.

    The Datasets Server /rows endpoint returns every column for a single row
    in about half a second, which is much faster than pulling the row group
    column chunks again through DuckDB.
    """
    try:
        resp = requests.get(
            f"{HF_ROWS_URL}&offset={row_number}&limit=1", timeout=12
        )
        if resp.status_code == 200:
            rows = resp.json().get("rows", [])
            if rows:
                r = rows[0].get("row", {})
                return {
                    "mobile": clean_field(r.get("mobile")),
                    "name": clean_field(r.get("name")),
                    "fname": clean_field(r.get("fname")),
                    "address": clean_field(r.get("address")),
                    "alt": clean_field(r.get("alt")),
                    "circle": clean_field(r.get("circle")),
                }
    except Exception:
        pass

    # Fallback to DuckDB
    c = conn if conn is not None else CON
    if c is None:
        return None
    sql = f"""
    SELECT mobile, name, fname, address, alt, circle
    FROM parquet_scan('{HF_PARQUET_URL}', file_row_number=true)
    WHERE file_row_number = {row_number}
    LIMIT 1
    """
    try:
        if conn is None:
            with _DUCKDB_LOCK:
                rows = c.execute(sql).fetchall()
        else:
            rows = c.execute(sql).fetchall()
        if rows:
            return build_record(rows[0])
    except Exception as _e:
        global LAST_ERROR
        LAST_ERROR = f"{type(_e).__name__}: {_e}"
    return None


def duckdb_scan(rg, phone_digits, conn=None):
    """Scan one row group of the remote parquet file using DuckDB HTTP range reads."""
    c = conn if conn is not None else CON
    if c is None:
        return None
    start_off = rg["o"]
    end_off = rg["o"] + rg["r"]
    sql = f"""
    SELECT mobile, name, fname, address, alt, circle
    FROM parquet_scan('{HF_PARQUET_URL}', file_row_number=true)
    WHERE file_row_number >= {start_off}
      AND file_row_number < {end_off}
      AND mobile = '{phone_digits}'
    LIMIT 1
    """
    try:
        if conn is None:
            with _DUCKDB_LOCK:
                rows = c.execute(sql).fetchall()
        else:
            rows = c.execute(sql).fetchall()
        if rows:
            return build_record(rows[0])
    except Exception as _e:
        global LAST_ERROR
        LAST_ERROR = f"{type(_e).__name__}: {_e}"
    return None


def http_fetch_batch(offset):
    """Fetch a batch of rows from the Hugging Face Datasets Server."""
    try:
        resp = requests.get(f"{HF_ROWS_URL}&offset={offset}&limit=100", timeout=15)
        if resp.status_code == 200:
            rows = resp.json().get("rows", [])
            if rows:
                return [r.get("row", {}) for r in rows]
    except Exception:
        pass
    return []


def http_scan(rg, phone_digits):
    """Fallback scan using the Hugging Face Datasets Server rows API (parallel batches)."""
    val = int(phone_digits)
    start_off = rg["o"]
    end_off = rg["o"] + rg["r"]
    ratio = (rg["M"] - val) / (rg["M"] - rg["m"]) if rg["M"] > rg["m"] else 0
    base = max(start_off, int(start_off + ratio * rg["r"]) - 3000)
    offsets = list(range(base, min(base + 12000, end_off), 100))

    def check(off):
        for row in http_fetch_batch(off):
            if str(row.get("mobile", "")).strip() == phone_digits:
                return {
                    "mobile": phone_digits,
                    "name": clean_field(row.get("name")),
                    "fname": clean_field(row.get("fname")),
                    "address": clean_field(row.get("address")),
                    "alt": clean_field(row.get("alt")),
                    "circle": clean_field(row.get("circle")),
                }
        return None

    with ThreadPoolExecutor(max_workers=12) as ex:
        for res in ex.map(check, offsets):
            if res:
                return res

    return None


def query_hf_database(phone):
    """Look up a phone number in the full Hugging Face dataset."""
    global CACHE_HITS, CACHE_MISSES
    phone_digits = ''.join(c for c in phone if c.isdigit())[-10:]
    if not phone_digits:
        return None

    # Cache hit - instant return
    if phone_digits in _CACHE:
        CACHE_HITS += 1
        return _CACHE[phone_digits]
    if phone_digits in DISK_CACHE:
        CACHE_HITS += 1
        rec = DISK_CACHE[phone_digits]
        _CACHE[phone_digits] = rec
        return rec
    CACHE_MISSES += 1

    time_start = time.time()
    cands = find_row_groups_by_range(phone_digits)

    # Fast path: the merged ranking puts the true owner in the first couple
    # of candidates. Phase 1 reads only the `mobile` column to locate the
    # row, phase 2 pulls the full record via the Datasets Server.
    for rg in cands[:3]:
        if time.time() - time_start > TIME_BUDGET:
            break
        row_number = duckdb_find_row(rg, phone_digits)
        if row_number is not None:
            record = duckdb_fetch_row(row_number)
            if record:
                _cache_put(phone_digits, record)
                return record

    # Wider sweep: locate the row in the remaining candidates concurrently.
    # Phase 1 only reads the `mobile` column (half the bytes of a full scan),
    # phase 2 then pulls the single matching row.
    if len(cands) > 3 and CON is not None and time.time() - time_start < TIME_BUDGET:
        record = _parallel_two_phase(cands[3:15], phone_digits, time_start)
        if record:
            _cache_put(phone_digits, record)
            return record

    # Sequential sweep over any remaining candidates
    for rg in cands[15:]:
        if time.time() - time_start > TIME_BUDGET:
            break
        row_number = duckdb_find_row(rg, phone_digits)
        if row_number is not None:
            record = duckdb_fetch_row(row_number)
            if record:
                _cache_put(phone_digits, record)
                return record

    # Fallback to HTTP API if DuckDB unavailable
    if CON is None:
        for rg in cands:
            if time.time() - time_start > TIME_BUDGET:
                break
            record = http_scan(rg, phone_digits)
            if record:
                _cache_put(phone_digits, record)
                return record

    return None


def _parallel_two_phase(row_groups, phone_digits, time_start):
    """Locate a row across several row groups concurrently, then fetch it.

    Phase 1 reads only the `mobile` column of each row group to locate the
    matching row number - that is about half the bytes of a full scan.
    Phase 2 then pulls the single matching row with every column.

    Cursors are separate connections to the same DuckDB database, so the
    parquet metadata is downloaded only once and each thread gets its own
    connection to work with.
    """
    if CON is None:
        return None

    # Create all cursors up front, under the lock, so the parent connection
    # is never touched while worker threads are running.
    cursors = []
    try:
        with _DUCKDB_LOCK:
            for _ in row_groups:
                try:
                    cursors.append(CON.cursor())
                except Exception:
                    break
    except Exception:
        return None

    if not cursors:
        return None

    def locate(pair):
        cur, rg = pair
        try:
            return duckdb_find_row(rg, phone_digits, cur)
        except Exception:
            return None

    row_number = None
    try:
        with ThreadPoolExecutor(max_workers=min(len(cursors), 12)) as ex:
            for res in ex.map(locate, list(zip(cursors, row_groups))):
                if res is not None:
                    row_number = res
                    break
                if time.time() - time_start > TIME_BUDGET:
                    break
    except Exception:
        pass
    finally:
        for cur in cursors:
            try:
                cur.close()
            except Exception:
                pass

    if row_number is None:
        return None

    # Phase 2: fetch the full record for the row we located.
    try:
        with _DUCKDB_LOCK:
            cur = CON.cursor()
        try:
            return duckdb_fetch_row(row_number, cur)
        finally:
            try:
                cur.close()
            except Exception:
                pass
    except Exception:
        return None


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "row_groups": len(REMOTE_RGS),
        "duckdb": CON is not None,
        "duckdb_error": DUCKDB_ERROR,
        "last_query_error": LAST_ERROR,
        "cache_size": len(_CACHE),
        "cache_hits": CACHE_HITS,
        "cache_misses": CACHE_MISSES,
        "endpoints": {
            "phone": "/api?phone=9876543210",
            "vehicle": "/api?vehical=MH12DE1433",
            "license": "/api?license=DL0420110149646",
            "ifsc": "/api?ifsc=SBIN0000001",
        },
        "meta": _build_meta(cached=False, cache_hits=CACHE_HITS, latency_ms=0),
    })


@app.route("/api")
def api():
    # Vehicle / license / IFSC lookups are handled by the parivahan module.
    for _param in ("vehical", "vehicle", "license", "ifsc"):
        if _param in request.args:
            from api import parivahan

            _status, _body = parivahan.lookup(_param, request.args.get(_param))
            return jsonify(_body), _status

    phone = request.args.get("phone")
    if not phone:
        return jsonify({
            "success": False,
            "error": "Missing parameter.",
            "examples": {
                "phone": "/api?phone=9876543210",
                "vehicle": "/api?vehical=MH12DE1433",
                "license": "/api?license=DL0420110149646",
                "ifsc": "/api?ifsc=SBIN0000001",
            },
            "meta": _build_meta(cached=False, cache_hits=CACHE_HITS, latency_ms=0),
        }), 400

    current_lookup = ''.join(c for c in phone if c.isdigit())[-10:]
    if not current_lookup:
        return jsonify({"error": "Invalid phone number"}), 400

    history_chain = []
    seen_numbers = set()
    time_start = time.time()

    for step in range(1, MAX_STEPS + 1):
        if not current_lookup or current_lookup in seen_numbers:
            break
        if time.time() - time_start > TIME_BUDGET:
            break

        seen_numbers.add(current_lookup)
        record = query_hf_database(current_lookup)
        if not record:
            if step == 1:
                _disk_cache_save()
                return jsonify({
                    "success": False,
                    "error": f"Record for {phone} not found in database",
                    "meta": _build_meta(
                        cached=False,
                        cache_hits=CACHE_HITS,
                        latency_ms=int((time.time() - time_start) * 1000),
                    ),
                }), 404
            break

        history_chain.append({
            "step": step,
            "queried_number": current_lookup,
            "profile": {
                "mobile": record.get("mobile"),
                "name": record.get("name"),
                "fname": record.get("fname"),
                "address": record.get("address"),
                "alt": record.get("alt"),
                "circle": record.get("circle"),
            },
        })

        # Follow the alt number - stop if it loops back to a number already seen
        next_lookup = record.get("alt")
        if not next_lookup:
            break
        next_digits = ''.join(c for c in str(next_lookup) if c.isdigit())[-10:]
        if not next_digits or next_digits in seen_numbers:
            break
        current_lookup = next_digits

    _disk_cache_save()

    return jsonify({
        "success": True,
        "requested_phone": phone,
        "total_chain_steps": len(history_chain),
        "lookup_chain": history_chain,
        "meta": _build_meta(
            cached=phone in _CACHE,
            cache_hits=CACHE_HITS,
            latency_ms=int((time.time() - time_start) * 1000),
        ),
    })


@app.route("/vehical")
@app.route("/vehicle")
def vehicle_route():
    from api import parivahan

    _status, _body = parivahan.lookup("vehical", request.args.get("vehical"))
    return jsonify(_body), _status


@app.route("/license")
def license_route():
    from api import parivahan

    _status, _body = parivahan.lookup("license", request.args.get("license"))
    return jsonify(_body), _status


@app.route("/ifsc")
def ifsc_route():
    from api import parivahan

    _status, _body = parivahan.lookup("ifsc", request.args.get("ifsc"))
    return jsonify(_body), _status


if __name__ == "__main__":
    app.run(debug=True)