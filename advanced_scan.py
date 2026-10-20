"""
Advanced exhaustive scanner for HiTeckGroup/HiTeckNuMinfo dataset.
Scans ALL row groups (not just min/max candidates) to find a phone number.
Uses DuckDB HTTP range queries to read parquet row groups directly from Hugging Face.

Usage: python advanced_scan.py <phone_number>
"""

import duckdb
import json
import sys
import time

HF_PARQUET_URL = "https://huggingface.co/datasets/HiTeckGroup/HiTeckNuMinfo/resolve/main/users_data.parquet"


def load_row_groups():
    """Load row group metadata from the local index file."""
    with open("rg_index.json") as f:
        raw_rgs = json.load(f)

    clean_rgs = []
    for i, cur in enumerate(raw_rgs):
        clean_rgs.append({
            "id": i,
            "offset": cur["o"],
            "rows": cur["r"],
            "min": cur["m"],
            "max": cur["M"],
            "span": cur["M"] - cur["m"]
        })
    return clean_rgs


def scan_row_group(con, url, rg, target):
    """Scan a single row group for an exact mobile number match."""
    start_off = rg["offset"]
    end_off = rg["offset"] + rg["rows"]

    sql = f"""
    SELECT mobile, name, fname, address, alt, circle 
    FROM parquet_scan('{url}', file_row_number=true) 
    WHERE file_row_number >= {start_off} 
      AND file_row_number < {end_off} 
      AND mobile = '{target}'
    """
    try:
        rows = con.execute(sql).fetchall()
        return rows
    except Exception as e:
        print(f"  [ERROR] RG {rg['id']}: {e}")
        return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python advanced_scan.py <phone_number>")
        sys.exit(1)

    target = sys.argv[1]

    # Validate phone number
    digits = ''.join(c for c in target if c.isdigit())[-10:]
    if not digits:
        print(f"Error: '{target}' is not a valid phone number")
        sys.exit(1)

    print("=" * 70)
    print(f"ADVANCED EXHAUSTIVE SCAN")
    print(f"Target: {digits}")
    print(f"Dataset: HiTeckGroup/HiTeckNuMinfo")
    print(f"Parquet: {HF_PARQUET_URL}")
    print("=" * 70)
    print()

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_http_metadata_cache = true;")

    rgs = load_row_groups()
    print(f"Total row groups in dataset: {len(rgs)}")
    print()

    # Sort row groups by distance to target (closest ranges first)
    val = int(digits)
    sorted_rgs = sorted(rgs, key=lambda r: min(abs(r["min"] - val), abs(r["max"] - val)))

    found = False
    total_time = 0.0
    scanned = 0

    for i, rg in enumerate(sorted_rgs):
        t0 = time.time()
        rows = scan_row_group(con, HF_PARQUET_URL, rg, digits)
        elapsed = time.time() - t0
        total_time += elapsed
        scanned += 1

        if rows:
            print(f"\n{'='*70}")
            print(f"*** FOUND EXACT MATCH in RG id={rg['id']}!")
            print(f"  Offset range: {rg['offset']} - {rg['offset'] + rg['rows']}")
            print(f"  Scan time: {elapsed:.1f}s")
            print(f"  Row groups scanned: {scanned}/{len(rgs)}")
            print(f"{'='*70}")
            for r in rows:
                print(json.dumps({
                    "mobile": r[0],
                    "name": r[1],
                    "fname": r[2],
                    "address": r[3],
                    "alt": r[4],
                    "circle": r[5]
                }, indent=2))
            found = True
            break
        else:
            if i < 5 or i % 50 == 0:
                print(f"  RG {rg['id']} (scan {scanned}/{len(rgs)}): not found ({elapsed:.1f}s)")

    print()
    print("=" * 70)
    if found:
        print(f"RESULT: Number {digits} FOUND in the database!")
    else:
        print(f"RESULT: Number {digits} NOT FOUND after scanning all {len(rgs)} row groups")
    print(f"Total scan time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Row groups scanned: {scanned}/{len(rgs)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
