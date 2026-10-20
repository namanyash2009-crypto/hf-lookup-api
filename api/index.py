from flask import Flask, request, jsonify
import os
import json
import requests
import time

app = Flask(__name__)

HF_DATASET = "HiTeckGroup/HiTeckNuMinfo"
HF_ROWS_URL = f"https://datasets-server.huggingface.co/rows?dataset={HF_DATASET}&config=default&split=train"

# Load row group index at cold start (bundled with the function)
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

TIME_BUDGET = 8.0  # seconds (Vercel free tier: 10s)


def clean_field(value):
    if not value or str(value).strip().lower() in ["null", "none", "", "nulllll", "-", "0", "na", "n/a"]:
        return None
    return str(value).strip()


def fetch_batch(offset):
    url = f"{HF_ROWS_URL}&offset={offset}&limit=100"
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            rows = resp.json().get("rows", [])
            if rows:
                return [r.get("row", {}) for r in rows]
    except Exception:
        pass
    return []


def find_row_groups_by_range(phone_digits):
    val = int(phone_digits)
    cands = [r for r in REMOTE_RGS if r["m"] <= val <= r["M"]]
    if not cands:
        cands = sorted(REMOTE_RGS, key=lambda r: min(abs(r["m"] - val), abs(r["M"] - val)))
    else:
        cands.sort(key=lambda r: r["M"] - r["m"])
    return cands[:5]


def scan_within_row_group(rg, phone_digits, time_start):
    val = int(phone_digits)
    start_off = rg["o"]
    span = rg["M"] - rg["m"]
    end_off = start_off + rg["r"]

    ratio = (rg["M"] - val) / span if span > 0 else 0
    offset = max(start_off, int(start_off + ratio * rg["r"]) - 50)

    jump = 100
    checked = set()
    offsets_to_check = [offset]

    for attempt in range(6):
        if time.time() - time_start > TIME_BUDGET:
            return None

        new_offsets = []
        for off in offsets_to_check:
            if off in checked or off < start_off or off >= end_off:
                continue
            checked.add(off)

            batch = fetch_batch(off)
            if not batch:
                continue

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

            first_mob = str(batch[0].get("mobile", "")).strip()
            last_mob = str(batch[-1].get("mobile", "")).strip()
            if first_mob.isdigit() and last_mob.isdigit():
                first_val = int(first_mob)
                last_val = int(last_mob)
                if last_val <= val <= first_val:
                    best = min(
                        batch,
                        key=lambda r: abs(int(str(r.get("mobile", "")).strip()) - val)
                        if str(r.get("mobile", "")).strip().isdigit()
                        else float("inf"),
                    )
                    return {
                        "mobile": phone_digits,
                        "name": clean_field(best.get("name")),
                        "fname": clean_field(best.get("fname")),
                        "address": clean_field(best.get("address")),
                        "alt": clean_field(best.get("alt")),
                        "circle": clean_field(best.get("circle")),
                    }

            new_offsets.append(off - jump)
            new_offsets.append(off + jump)

        offsets_to_check = new_offsets
        jump = min(jump * 2, 3000)

    return None


def query_hf_database(phone):
    phone_digits = ''.join(c for c in phone if c.isdigit())[-10:]
    if not phone_digits:
        return None

    time_start = time.time()
    cands = find_row_groups_by_range(phone_digits)

    for rg in cands:
        if time.time() - time_start > TIME_BUDGET:
            break
        result = scan_within_row_group(rg, phone_digits, time_start)
        if result:
            return result

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