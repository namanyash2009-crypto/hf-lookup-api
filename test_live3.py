import requests
import json
import time

BASE = "https://naman-hightek.vercel.app"

print("=== health ===")
r = requests.get(f"{BASE}/health", timeout=250)
print(r.status_code, json.dumps(r.json(), indent=2))

print("\n=== missing param ===")
r = requests.get(f"{BASE}/api", timeout=60)
print(r.status_code, r.text)

print("\n=== live lookups ===")
for p in ["8168781901", "8059020030", "9999999701", "9728352122", "7015551442", "0000000000"]:
    t0 = time.time()
    try:
        r = requests.get(f"{BASE}/api?phone={p}", timeout=290)
        d = r.json()
        dt = time.time() - t0
        if r.status_code == 200:
            pr = d["lookup_chain"][0]["profile"]
            print(
                "  %s -> %d steps in %.1fs | %s | %s | %s | alt=%s"
                % (p, d["total_chain_steps"], dt, pr["name"], pr["fname"], pr["circle"], pr["alt"])
            )
        else:
            print("  %s -> %s in %.1fs | %s" % (p, r.status_code, dt, d.get("error")))
    except Exception as e:
        print("  %s -> ERR %s" % (p, e))
