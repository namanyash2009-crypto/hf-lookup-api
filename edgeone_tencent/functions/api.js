/**
 * EdgeOne Edge Function - HF Dataset Lookup API
 * Tencent Cloud EdgeOne Edge Functions (JavaScript)
 * 
 * Usage:
 *   GET /api?phone=8168781901
 *   GET /api?phone=8059020030&deep=true
 *   GET /health
 */

const HF_DATASET = "HiTeckGroup/HiTeckNuMinfo";
const HF_ROWS_URL = `https://datasets-server.huggingface.co/rows?dataset=${HF_DATASET}&config=default&split=train`;

// Row group index - will be populated at cold start
let REMOTE_RGS = null;
let RGS_LOADING = false;
let RGS_LOAD_PROMISE = null;

async function loadRemoteRGS() {
  if (RGS_LOADED !== undefined && RGS_LOADED) return REMOTE_RGS;
  if (RGS_LOADING) return RGS_LOAD_PROMISE;
  
  RGS_LOADING = true;
  RGS_LOAD_PROMISE = (async () => {
    try {
      // Try to load rg_index.json from EdgeOne assets
      const resp = await fetch("/rg_index.json", { timeout: 5000 });
      if (resp.status === 200) {
        REMOTE_RGS = await resp.json();
      } else {
        REMOTE_RGS = [];
      }
    } catch (e) {
      REMOTE_RGS = [];
    }
    RGS_LOADED = true;
    return REMOTE_RGS;
  })();
  
  return RGS_LOAD_PROMISE;
}

function cleanField(value) {
  if (!value) return null;
  const s = String(value).trim().toLowerCase();
  if (["null", "none", "", "nulllll", "-", "0", "na", "n/a"].includes(s)) return null;
  return String(value).trim();
}

async function fetchBatch(offset) {
  const url = `${HF_ROWS_URL}&offset=${offset}&limit=100`;
  try {
    const resp = await fetch(url, { timeout: 8000 });
    if (resp.status !== 200) return [];
    const json = await resp.json();
    return (json.rows || []).map(r => r.row || {});
  } catch (e) {
    return [];
  }
}

function findRowGroupsByRange(phoneDigits, rgs) {
  const val = parseInt(phoneDigits, 10);
  let cands = rgs.filter(r => r.m <= val && val <= r.M);
  
  if (cands.length === 0) {
    cands = rgs.sort((a, b) => {
      const da = Math.min(Math.abs(a.m - val), Math.abs(a.M - val));
      const db = Math.min(Math.abs(b.m - val), Math.abs(b.M - val));
      return da - db;
    });
  } else {
    cands.sort((a, b) => (a.M - a.m) - (b.M - b.m));
  }
  
  return cands.slice(0, 5);
}

async function scanWithinRowGroup(rg, phoneDigits, timeStart, timeBudget) {
  const val = parseInt(phoneDigits, 10);
  const startOff = rg.o;
  const span = rg.M - rg.m;
  const endOff = startOff + rg.r;
  
  const ratio = span > 0 ? (rg.M - val) / span : 0;
  let offset = Math.max(startOff, Math.floor(startOff + ratio * rg.r) - 50);
  
  let jump = 100;
  const checked = new Set();
  let offsetsToCheck = [offset];
  
  for (let attempt = 0; attempt < 6; attempt++) {
    if (Date.now() - timeStart > timeBudget) return null;
    
    const newOffsets = [];
    for (const off of offsetsToCheck) {
      if (checked.has(off) || off < startOff || off >= endOff) continue;
      checked.add(off);
      
      const batch = await fetchBatch(off);
      if (!batch || batch.length === 0) continue;
      
      // Check exact match
      for (const row of batch) {
        if (String(row.mobile || "").trim() === phoneDigits) {
          return {
            mobile: phoneDigits,
            name: cleanField(row.name),
            fname: cleanField(row.fname),
            address: cleanField(row.address),
            alt: cleanField(row.alt),
            circle: cleanField(row.circle),
          };
        }
      }
      
      // Check if target is within batch range
      const firstMob = String(batch[0].mobile || "").trim();
      const lastMob = String(batch[batch.length - 1].mobile || "").trim();
      if (firstMob && lastMob && /^\d+$/.test(firstMob) && /^\d+$/.test(lastMob)) {
        const firstVal = parseInt(firstMob, 10);
        const lastVal = parseInt(lastMob, 10);
        if (lastVal <= val && val <= firstVal) {
          const best = batch.reduce((best, row) => {
            const m = String(row.mobile || "").trim();
            if (!/^\d+$/.test(m)) return best;
            const diff = Math.abs(parseInt(m, 10) - val);
            return diff < best.diff ? { row, diff } : best;
          }, { diff: Infinity });
          if (best.row) {
            return {
              mobile: phoneDigits,
              name: cleanField(best.row.name),
              fname: cleanField(best.row.fname),
              address: cleanField(best.row.address),
              alt: cleanField(best.row.alt),
              circle: cleanField(best.row.circle),
            };
          }
        }
      }
      
      // Galloping search adjustment
      if (firstMob && /^\d+$/.test(firstMob)) {
        const firstVal = parseInt(firstMob, 10);
        if (val > firstVal) {
          offset = Math.max(startOff, offset - 200);
        } else {
          const lastVal = parseInt(lastMob, 10);
          const rate = Math.abs(firstVal - lastVal) / Math.max(1, batch.length - 1);
          if (rate <= 0) {
            offset += 100;
          } else {
            const diffFromLast = lastVal - val;
            const step = Math.floor(diffFromLast / rate);
            offset += step + 20;
            offset = Math.min(offset, endOff - 1);
          }
        }
      }
      
      newOffsets.push(off - jump);
      newOffsets.push(off + jump);
    }
    
    offsetsToCheck = newOffsets;
    jump = Math.min(jump * 2, 3000);
  }
  
  return null;
}

async function scanFullDataset(phoneDigits, timeStart, timeBudget, rgs) {
  const val = parseInt(phoneDigits, 10);
  const sortedRgs = rgs.sort((a, b) => {
    const da = Math.min(Math.abs(a.m - val), Math.abs(a.M - val));
    const db = Math.min(Math.abs(b.m - val), Math.abs(b.M - val));
    return da - db;
  });
  
  for (const rg of sortedRgs) {
    if (Date.now() - timeStart > timeBudget) return null;
    const result = await scanWithinRowGroup(rg, phoneDigits, timeStart, timeBudget);
    if (result) return result;
  }
  return null;
}

async function queryHFDatabase(phone, deepScan, rgs) {
  const cleanPhone = cleanField(phone);
  if (!cleanPhone) return null;
  
  const phoneDigits = String(cleanPhone).replace(/\D/g, "").slice(-10);
  if (!phoneDigits) return null;
  
  const timeStart = Date.now();
  const TIME_BUDGET = 8500;
  const val = parseInt(phoneDigits, 10);
  
  // Phase 1: Fast candidate row groups
  const cands = findRowGroupsByRange(phoneDigits, rgs);
  
  for (const rg of cands) {
    if (Date.now() - timeStart > TIME_BUDGET) break;
    const result = await scanWithinRowGroup(rg, phoneDigits, timeStart, TIME_BUDGET);
    if (result) return result;
  }
  
  // Phase 2: Deep scan
  if (deepScan) {
    const result = await scanFullDataset(phoneDigits, timeStart, TIME_BUDGET, rgs);
    if (result) return result;
  }
  
  return null;
}

export default async function (request, env, ctx) {
  const url = new URL(request.url);
  const path = url.pathname;
  
  // Health check
  if (path === "/health") {
    return new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }
  
  // Favicon
  if (path === "/favicon.ico") {
    return new Response(null, { status: 204 });
  }
  
  // Only handle /api
  if (path !== "/api" && !path.startsWith("/api")) {
    return new Response(JSON.stringify({ error: "Not found" }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  }
  
  const phone = url.searchParams.get("phone");
  const deepScan = url.searchParams.get("deep") === "true";
  
  if (!phone) {
    return new Response(
      JSON.stringify({ error: "Missing parameter. Correct use example: /api?phone=8168781901" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  
  // Load row group index
  const rgs = await loadRemoteRGS();
  
  // Process lookup chain
  let currentLookup = cleanField(phone);
  const historyChain = [];
  const seenNumbers = new Set();
  
  for (let step = 1; step <= 3; step++) {
    const cleanDigits = String(currentLookup || "").replace(/\D/g, "").slice(-10);
    if (!cleanDigits || seenNumbers.has(cleanDigits)) break;
    
    seenNumbers.add(cleanDigits);
    
    const record = await queryHFDatabase(cleanDigits, deepScan, rgs);
    if (!record) {
      if (step === 1) {
        return new Response(
          JSON.stringify({ error: `Record for ${phone} not found in database` }),
          {
            status: 404,
            headers: { "Content-Type": "application/json" },
          }
        );
      }
      break;
    }
    
    historyChain.push({
      step,
      queried_number: cleanDigits,
      profile: {
        name: record.name,
        fname: record.fname,
        mobile: record.mobile,
        alt: record.alt,
        circle: record.circle,
        address: record.address,
      },
    });
    
    currentLookup = record.alt;
  }
  
  const responseData = {
    requested_phone: phone,
    total_chain_steps: historyChain.length,
    lookup_chain: historyChain,
  };
  
  return new Response(JSON.stringify(responseData, null, 2), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
