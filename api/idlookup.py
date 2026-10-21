"""Resolve a row by its `id` column value.

The `id` column mixes numeric RC numbers, driving-licence numbers, masked
values and usernames, and it is not sorted - so row groups cannot be pruned
by value the way the `mobile` column can.  The only way to resolve an id is
to read the id column.

That is still far cheaper than reading the whole file: the id column is about
9% of the 93 GB parquet (8.7 GB compressed), and a scan that projects just
`id` touches only those bytes.

The scan is split into windows of row groups and all windows are submitted at
once to a pool of workers, so the whole file is covered at the maximum
bandwidth the connection can sustain.  The first window that finds the id
wins and the rest stop as soon as they notice.

Every worker uses its own DuckDB connection - a connection is not safe to
use from two threads at once - and all byte-range traffic goes through the
local Range proxy, so the parquet metadata is fetched once (it is cached on
disk) and upstream 429 throttling is retried.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# Row groups per window.  A window of 30 row groups is about 3.7 M rows,
# roughly 18 MB of compressed id data.
WINDOW_RGS = 30

# Windows scanned concurrently.  This is what saturates the link: the id
# column is read in large chunks, so the number of parallel readers is what
# decides how fast the scan runs.
CONCURRENT_WINDOWS = 24

# DuckDB threads used inside a window.
SCAN_THREADS = 4


def _sql_literal(value):
    """Quote a string for safe inclusion in SQL."""
    return "'" + str(value).replace("'", "''") + "'"


def _make_scan_connection(url):
    """Create a DuckDB connection dedicated to id scans."""
    import duckdb

    conn = duckdb.connect()

    _tmp = os.environ.get("TMPDIR", "/tmp")
    # A separate home directory keeps this connection's httpfs metadata cache
    # out of the way of the main connection's cache lock.
    home = os.path.join(_tmp, "duckdb_id_home")
    try:
        os.makedirs(home, exist_ok=True)
        conn.execute(f"SET home_directory='{home}';")
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
        conn.execute("INSTALL httpfs;")
        conn.execute("LOAD httpfs;")

    conn.execute("SET enable_http_metadata_cache = true;")
    conn.execute("SET http_timeout = 60000;")
    conn.execute(f"SET threads = {SCAN_THREADS};")
    return conn


class IdScanner:
    """Scans the remote parquet for rows whose `id` matches."""

    def __init__(self, url, row_groups):
        self.url = url
        self.row_groups = row_groups or []
        self._local = threading.local()
        self._executor = ThreadPoolExecutor(max_workers=CONCURRENT_WINDOWS)

    def _connection(self):
        """Return this thread's DuckDB connection, creating it on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = _make_scan_connection(self.url)
            self._local.conn = conn
        return conn

    def _windows(self):
        """Contiguous row-number windows covering the whole file."""
        windows = []
        rgs = self.row_groups
        for i in range(0, len(rgs), WINDOW_RGS):
            group = rgs[i:i + WINDOW_RGS]
            if not group:
                continue
            lo = group[0]["o"]
            hi = group[-1]["o"] + group[-1]["r"]
            windows.append((lo, hi))
        return windows

    def _scan_window(self, window, key, tolerant, stop):
        """Return the matching row numbers inside one window."""
        if stop.is_set():
            return []

        lo, hi = window
        lit = _sql_literal(key)
        if tolerant:
            # The stored id can carry padding or mixed case, so match on the
            # trimmed and case-folded value as well.
            condition = (
                f"(id = {lit} OR trim(id) = {lit} "
                f"OR lower(trim(id)) = lower({lit}))"
            )
        else:
            # Plain equality keeps the hot loop cheap: no per-row function
            # calls, which matters when scanning a billion rows.
            condition = f"id = {lit}"

        sql = f"""
        SELECT file_row_number
        FROM parquet_scan('{self.url}', file_row_number=true)
        WHERE file_row_number >= {lo}
          AND file_row_number < {hi}
          AND {condition}
        LIMIT 8
        """
        try:
            conn = self._connection()
            rows = conn.execute(sql).fetchall()
            if rows:
                stop.set()
            return [r[0] for r in rows]
        except Exception:
            return []

    def find(self, id_value, time_budget=240.0):
        """Return (row_number, complete).

        row_number is the matching row, or None.  complete is False when the
        time budget ran out before the whole file was scanned, which means a
        "not found" answer is not conclusive.
        """
        key = str(id_value).strip()
        if not key:
            return None, True

        windows = self._windows()
        if not windows:
            return None, True

        start = time.time()
        stop = threading.Event()
        hit = []

        for tolerant in (False, True):
            pending = {
                self._executor.submit(self._scan_window, w, key, tolerant, stop): w
                for w in windows
            }
            timed_out = False

            while pending:
                done, _ = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        hit.extend(fut.result())
                    except Exception:
                        pass
                    pending.pop(fut, None)
                if hit:
                    break
                if time.time() - start > time_budget:
                    timed_out = True
                    break

            for fut in pending:
                fut.cancel()
            stop.set()

            if hit:
                # Lowest row number wins, so the answer is stable no matter
                # which window found it first.
                return min(hit), True
            if timed_out:
                return None, False

        return None, True
