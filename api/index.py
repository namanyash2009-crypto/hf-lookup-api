from flask import Flask, request, jsonify
import os
import json
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

HF_DATASET = "HiTeckGroup/HiTeckNuMinfo"
HF_PARQUET_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/users_data.parquet"
HF_ROWS_URL = f"https://datasets-server.huggingface.co/rows?dataset={HF_DATASET}&config=default&split=train"

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


# Build descending chunks (independent sorted runs) from the row-group index.
# Each chunk is a monotonic run, so a value can be located by binary search.
CHUNKS = []
_start = 0
for _i in range(1, len(REMOTE_RGS)):
    if REMOTE_RGS[_i]["m"] > REMOTE_RGS[_i - 1]["m"]:
        CHUNKS.append((_start, _i - 1))
        _start = _i
if REMOTE_RGS:
    CHUNKS.append((_start, len(REMOTE_RGS) - 1))


def find_row_groups_by_range(phone_digits):
    """Locate candidate row groups for a phone number.

    The dataset is a set of independent descending runs ("chunks"). Parquet
    statistics are approximate, so several row groups can claim the same
    number. Row groups whose range tightly brackets the number are tried
    first, since the true owner is almost always among the tightest.
    """
    val = int(phone_digits)

    if ACC_RGS:
        # Accurate index available: use the true first-row value of each row
        # group, which brackets the number much more precisely.
        hits = []
        for r in ACC_RGS:
            lo = r["m"]
            hi = r["f"]
            if lo > hi:
                lo, hi = hi, lo
            if lo <= val <= hi:
                hits.append((hi - lo, abs((lo + hi) // 2 - val), r))
        if hits:
            hits.sort(key=lambda t: (t[0], t[1]))
            return [h[2] for h in hits]
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
    sql = f"""
    SELECT file_row_number
    FROM parquet_scan('{HF_PARQUET_URL}', file_row_number=true)
    WHERE file_row_number >= {rg["o"]}
      AND file_row_number < {rg["o"] + rg["r"]}
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
    """Phase 2: fetch the full record for a known row number."""
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

    # Fast path: the tightest-bracketing row group holds the record most of
    # the time, so try it alone first (one HTTP range read).
    if cands:
        record = duckdb_scan(cands[0], phone_digits)
        if record:
            _cache_put(phone_digits, record)
            return record

    # Wider sweep: locate the row in the remaining candidates concurrently.
    # Phase 1 only reads the `mobile` column (half the bytes of a full scan),
    # phase 2 then pulls the single matching row.
    if len(cands) > 1 and CON is not None and time.time() - time_start < TIME_BUDGET:
        record = _parallel_two_phase(cands[1:21], phone_digits, time_start)
        if record:
            _cache_put(phone_digits, record)
            return record

    # Sequential sweep over any remaining candidates
    for rg in cands[21:]:
        if time.time() - time_start > TIME_BUDGET:
            break
        record = duckdb_scan(rg, phone_digits)
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
        with ThreadPoolExecutor(max_workers=min(len(cursors), 8)) as ex:
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
    })


@app.route("/api")
def api():
    phone = request.args.get("phone")
    if not phone:
        return jsonify({"error": "Missing parameter. Correct use example: /api?phone=9876543210"}), 400

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
                return jsonify({"error": f"Record for {phone} not found in database"}), 404
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
        "requested_phone": phone,
        "total_chain_steps": len(history_chain),
        "lookup_chain": history_chain,
    })


if __name__ == "__main__":
    app.run(debug=True)