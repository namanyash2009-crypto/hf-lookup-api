"""Simulate a Vercel cold container: fresh process, one lookup."""
import sys
import time

sys.path.insert(0, ".")

from api.index import app, CACHE_HITS, CACHE_MISSES

phone = sys.argv[1]
c = app.test_client()
t0 = time.time()
r = c.get(f"/api?phone={phone}")
dt = time.time() - t0
d = r.get_json()
if r.status_code == 200:
    pr = d["lookup_chain"][0]["profile"]
    print(
        "%s -> %d steps in %.1fs | %s | %s | %s | alt=%s"
        % (phone, d["total_chain_steps"], dt, pr["name"], pr["fname"], pr["circle"], pr["alt"])
    )
else:
    print("%s -> %s in %.1fs | %s" % (phone, r.status_code, dt, d.get("error")))
