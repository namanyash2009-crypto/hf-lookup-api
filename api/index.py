from flask import Flask, request, jsonify
import os
import json
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

# DuckDB connection (optional - HTTP fallback used if unavailable)
CON = None
DUCKDB_ERROR = None
try:
    import duckdb

    CON = duckdb.connect()

    # Vercel has no HOME env var - DuckDB needs home_directory + extension_directory
    # pointed at a writable location (/tmp) to install/load the httpfs extension.
    _tmp = os.environ.get("TMPDIR", "/tmp")
    try:
        os.makedirs(_tmp, exist_ok=True)
        CON.execute(f"SET home_directory='{_tmp}';")
    except Exception:
        pass

    _extdir = os.path.join(_tmp, "duckdb_ext")
    try:
        os.makedirs(_extdir, exist_ok=True)
        CON.execute(f"SET extension_directory='{_extdir}';")
    except Exception:
        pass

    try:
        CON.execute("LOAD httpfs;")
    except Exception:
        # httpfs is not bundled in the pip wheel - download it once into /tmp
        CON.execute("INSTALL httpfs;")
        CON.execute("LOAD httpfs;")

    CON.execute("SET enable_http_metadata_cache = true;")
    CON.execute("SET http_timeout = 60000;")
except Exception as _e:
    CON = None
    DUCKDB_ERROR = f"{type(_e).__name__}: {_e}"

LAST_ERROR = None

# Vercel Hobby supports up to 300s with Fluid compute
TIME_BUDGET = 280.0


def clean_field(value):
    if not value or str(value).strip().lower() in ["null", "none", "", "nulllll", "-", "0", "na", "n/a"]:
        return None
    return str(value).trim()


def find_row_groups_by_range(phone_digits):
    """Find candidate row groups whose min/max range contains this number."""
    val = int(phone_digits)
    cands = [r for r in REMOTE_RGS if r["m"] <= val <= r["M"]]
    if not cands:
        cands = sorted(REMOTE_RGS, key=lambda r: min(abs(r["m"] - val), abs(r["M"] - val)))
    else:
        cands.sort(key=lambda r: r["M"] - r["m"])
    return cands


def duckdb_scan(rg, phone_digits):
    """Scan one row group of the remote parquet file using DuckDB HTTP range reads."""
    if CON is None:
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
        rows = CON.execute(sql).fetchall()
        if rows:
            r = rows[0]
            return {
                "mobile": str(r[0]).strip(),
                "name": clean_field(r[1]),
                "fname": clean_field(r[2]),
                "address": clean_field(r[3]),
                "alt": clean_field(r[4]),
                "circle": clean_field(r[5]),
            }
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

    # Scan a wide window of batches in parallel around the interpolated offset
    offsets = list(range(base, min(base + 12000, end_off), 100))

    def check(off):
        batch = http_fetch_batch(off)
        for row in batch:
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
    phone_digits = ''.join(c for c in phone if c.isdigit())[-10:]
    if not phone_digits:
        return None

    time_start = time.time()
    cands = find_row_groups_by_range(phone_digits)

    for rg in cands:
        if time.time() - time_start > TIME_BUDGET:
            break

        record = duckdb_scan(rg, phone_digits)
        if record:
            return record

    # Fallback to HTTP API if DuckDB unavailable
    if CON is None:
        for rg in cands:
            if time.time() - time_start > TIME_BUDGET:
                break
            record = http_scan(rg, phone_digits)
            if record:
                return record

    return None


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "row_groups": len(REMOTE_RGS),
        "duckdb": CON is not None,
        "duckdb_error": DUCKDB_ERROR,
        "last_query_error": LAST_ERROR,
    })


@app.route("/diag")
def diag():
    """Diagnostic: run one real query and report timing/errors."""
    t0 = time.time()
    sql = (
        "SELECT mobile, name FROM parquet_scan('" + HF_PARQUET_URL + "', file_row_number=true) "
        "WHERE file_row_number >= 233228287 AND file_row_number < 233351167 "
        "AND mobile = '8168781901' LIMIT 1"
    )
    try:
        rows = CON.execute(sql).fetchall() if CON else None
        return jsonify({
            "ok": True,
            "seconds": round(time.time() - t0, 2),
            "rows": rows,
            "last_query_error": LAST_ERROR,
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "seconds": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {e}",
        }), 500


@app.route("/api")
def api():
    phone = request.args.get("phone")
    if not phone:
        return jsonify({"error": "Missing parameter. Correct use example: /api?phone=8168781901"}), 400

    current_lookup = ''.join(c for c in phone if c.isdigit())[-10:]
    if not current_lookup:
        return jsonify({"error": "Invalid phone number"}), 400

    history_chain = []
    seen_numbers = set()

    for step in range(1, 4):
        if not current_lookup or current_lookup in seen_numbers:
            break

        seen_numbers.add(current_lookup)
        record = query_hf_database(current_lookup)
        if not record:
            if step == 1:
                return jsonify({"error": f"Record for {phone} not found in database"}), 404
            break

        history_chain.append({
            "step": step,
            "queried_number": current_lookup,
            "profile": {
                "name": record.get("name"),
                "fname": record.get("fname"),
                "mobile": record.get("mobile"),
                "alt": record.get("alt"),
                "circle": record.get("circle"),
                "address": record.get("address"),
            },
        })
        current_lookup = record.get("alt")

    return jsonify({
        "requested_phone": phone,
        "total_chain_steps": len(history_chain),
        "lookup_chain": history_chain,
    })


if __name__ == "__main__":
    app.run(debug=True)