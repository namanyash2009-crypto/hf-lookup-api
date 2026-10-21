"""Parivahan helpers: vehicle / license / IFSC lookups with a clean, cached API.

The upstream service (parivahan-naman.ai.studio) returns raw scraped fields.
We normalise the payload, hide the internal `raw_fields` blob, and attach a
`meta` block with credits, cache stats, latency and a timestamp.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone

import requests

CREDITS = "https://t.me/NamanYadavBots"
UPSTREAM = "https://parivahan-naman.ai.studio/api"

# Query parameter -> upstream parameter
PARAM_MAP = {
    "vehical": "vehical",
    "vehicle": "vehical",
    "license": "license",
    "ifsc": "ifsc",
}

# Keys that hold internal scraped fields and are stripped from every response
HIDDEN_KEYS = ("raw_fields",)


def _now_iso():
    """Current UTC time as an ISO-8601 string with milliseconds."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _strip_hidden(obj):
    """Recursively remove internal/raw fields from a payload."""
    if isinstance(obj, dict):
        return {k: _strip_hidden(v) for k, v in obj.items() if k not in HIDDEN_KEYS}
    if isinstance(obj, list):
        return [_strip_hidden(v) for v in obj]
    return obj


class _Cache:
    """Tiny thread-safe in-memory + disk cache with hit counters."""

    def __init__(self, path, max_entries=5000):
        self.path = path
        self.max_entries = max_entries
        self.data = {}
        self.lock = threading.Lock()
        self._load()

    def _load(self):
        try:
            if os.path.isfile(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.data = {k: v for k, v in raw.items() if isinstance(v, dict)}
        except Exception:
            self.data = {}

    def get(self, key):
        with self.lock:
            entry = self.data.get(key)
            if entry is None:
                return None
            entry["cache_hits"] = entry.get("cache_hits", 0) + 1
            return entry

    def set(self, key, payload):
        with self.lock:
            if len(self.data) >= self.max_entries:
                for k in list(self.data.keys())[:1000]:
                    self.data.pop(k, None)
            self.data[key] = {
                "payload": payload,
                "cache_hits": 0,
                "saved_at": _now_iso(),
            }

    def save(self):
        with self.lock:
            snapshot = dict(self.data)
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snapshot, f)
            os.replace(tmp, self.path)
        except Exception:
            pass


CACHE = _Cache(os.path.join(os.environ.get("TMPDIR", "/tmp"), "parivahan_cache.json"))


def _upstream_fetch(param, value, timeout=20):
    """Call the upstream service and return its parsed JSON."""
    resp = requests.get(
        f"{UPSTREAM}?{param}={value}",
        timeout=timeout,
        headers={"Accept": "application/json"},
    )
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, {"success": False, "error": "Upstream returned a non-JSON response"}


def build_meta(cached, cache_hits, latency_ms):
    """Standard meta block attached to every response."""
    return {
        "credits": CREDITS,
        "cached": cached,
        "cache_hits": cache_hits,
        "latency_ms": latency_ms,
        "timestamp": _now_iso(),
    }


def lookup(param, value):
    """Look up a vehicle / license / IFSC record.

    Returns (http_status, response_body).
    """
    started = time.time()
    value = (value or "").strip()
    upstream_param = PARAM_MAP.get(param, param)

    if not value:
        latency = int((time.time() - started) * 1000)
        return 400, {
            "success": False,
            "error": "Please enter query properly",
            "examples": {
                "master_all": "/api?vehical=all",
                "vehicle": "/api?vehical=MH12DE1433",
                "license": "/api?license=DL0420110149646",
                "ifsc": "/api?ifsc=SBIN0000001",
            },
            "meta": build_meta(False, 0, latency),
        }

    cache_key = f"{upstream_param}:{value.upper()}"
    entry = CACHE.get(cache_key)
    if entry is not None:
        latency = int((time.time() - started) * 1000)
        body = json.loads(json.dumps(entry["payload"]))
        body["meta"] = build_meta(True, entry.get("cache_hits", 0), latency)
        return 200, body

    try:
        status, payload = _upstream_fetch(upstream_param, value)
    except requests.exceptions.Timeout:
        latency = int((time.time() - started) * 1000)
        return 504, {
            "success": False,
            "error": "Upstream request timed out. Please try again.",
            "meta": build_meta(False, 0, latency),
        }
    except Exception as e:
        latency = int((time.time() - started) * 1000)
        return 502, {
            "success": False,
            "error": f"Could not reach upstream service: {e}",
            "meta": build_meta(False, 0, latency),
        }

    latency = int((time.time() - started) * 1000)
    body = _strip_hidden(payload)

    if not isinstance(body, dict):
        body = {"success": False, "error": "Unexpected upstream response"}

    body["meta"] = build_meta(False, 0, latency)

    # Only cache successful lookups
    if body.get("success"):
        CACHE.set(cache_key, body)
        CACHE.save()

    http_status = 200 if body.get("success") else (status if status >= 400 else 404)
    return http_status, body
