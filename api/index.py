from flask import Flask, request, jsonify
import os
import json
import time
import duckdb

app = Flask(__name__)

HF_PARQUET_URL = "https://huggingface.co/datasets/HiTeckGroup/HiTeckNuMinfo/resolve/main/users_data.parquet"

# DuckDB connection with remote HTTP parquet access (Linux wheel supported on Vercel)
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET enable_http_metadata_cache = true;")

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

# Vercel Hobby allows up to 300s with Fluid compute
TIME_BUDGET = 280.0


def clean_field(value):
    if not value or str(value).strip().lower() in ["null", "none", "", "nulllll", "-", "0", "na", "n/a"]:
        return None
    return str(value).strip()


def find_row_groups_by_range(phone_digits):
    """Find candidate row groups whose min/max range contains this number."""
    val = int(phone_digits)
    cands = [r for r in REMOTE_RGS if r["m"] <= val <= r["M"]]
    if not cands:
        cands = sorted(REMOTE_RGS, key=lambda r: min(abs(r["m"] - val), abs(r["M"] - val)))
    else:
        cands.sort(key=lambda r: r["M"] - r["m"])
    return cands


def scan_row_group(rg, phone_digits):
    """Scan one row group of the remote parquet file for an exact mobile match."""
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
        rows = con.execute(sql).fetchall()
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
    except Exception:
        pass
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
        record = scan_row_group(rg, phone_digits)
        if record:
            return record

    return None


@app.route("/health")
def health():
    return jsonify({"status": "ok", "row_groups": len(REMOTE_RGS)})


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