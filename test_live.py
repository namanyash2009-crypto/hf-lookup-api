import requests
import json
import time

BASE = "https://naman-hightek.vercel.app"

for p in ["8168781901", "8059020030", "9999999701", "7015551442"]:
    t0 = time.time()
    r = requests.get(f"{BASE}/api?phone={p}", timeout=280)
    d = r.json()
    print(f"{p} -> {r.status_code} in {time.time()-t0:.1f}s  (steps={d.get('total_chain_steps')})")
    for s in d.get("lookup_chain", []):
        pr = s["profile"]
        print("  Step %d" % s["step"])
        print("    mobile : %s" % pr["mobile"])
        print("    name   : %s" % pr["name"])
        print("    fname  : %s" % pr["fname"])
        print("    address: %s" % pr["address"])
        print("    alt    : %s" % pr["alt"])
        print("    circle : %s" % pr["circle"])
    if "error" in d:
        print("   ERROR:", d["error"])
    print()

# diag must be gone
r = requests.get(f"{BASE}/diag", timeout=60)
print("/diag ->", r.status_code, "(expect 404)")
