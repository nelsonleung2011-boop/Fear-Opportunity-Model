# Testing, migration, and troubleshooting

## 1. Testing instructions

All commands are optional maintenance tasks. End users never run them.

### Dashboard (offline, deterministic)

```bash
cd C:\Users\User\WorkBuddy\2026-11-16-20-56-14\ETF_Timing_Dashboard
node test_dashboard_v2.js
```

Checks: worker-first URL construction, symbol encoding, worker payload parsing,
symbol verification, retries/backoff, no-look-ahead score reproducibility,
embedded fallback, partial fallback, provider status panel, per-symbol debug
fields, and the one-year chart window.

### Dashboard (optional live Yahoo upstream check)

```bash
node test_dashboard_v2.js --live
```

This temporarily disables the Worker and calls Yahoo directly **only to prove
the upstream API still works**. Normal app operation never does this.

### Cloudflare Worker (offline, deterministic)

```bash
node test_worker.js
```

Checks: `/health`, CORS headers, preflight, symbol verification, clean JSON,
10-minute cache header, cache HIT/MISS, retry on 503, and structured error codes.

### Real Worker smoke test (after deployment)

Open these in a browser:

```
https://etf-market-data.your-subdomain.workers.dev/health
https://etf-market-data.your-subdomain.workers.dev/chart?symbol=SPY&period1=1704067200&period2=1735689600
```

Then open the hosted dashboard and use **Diagnostics → Data Provider Status**
and **Test Live Market Data**. Expected result: Worker PASS, six green ticks,
each showing Source = Worker, the real URL, HTTP 200, and latency in ms.

### Build validation

```bash
python build_dashboard.py
```

Regenerates `dashboard_data.json`, injects it into `dashboard_template.html`,
writes `ETF_Timing_Dashboard.html` and `index.html`, and runs a JavaScript syntax
check.

---

## 2. Migration guide (old direct-Yahoo setup → Worker setup)

| Old behaviour | New behaviour |
|---|---|
| Browser called `query1.finance.yahoo.com` directly | Browser calls the Cloudflare Worker |
| All six symbols failed in 60–90 ms on GitHub Pages | Worker fetches Yahoo server-side |
| "CORS blocked by browser or network unreachable" | Worker status + exact URL/status/latency shown |
| `DATA_CONFIG.proxyEndpoint` was the optional proxy hook | `DEPLOY_CONFIG.workerBase` is the primary setting |
| Direct Yahoo was a candidate in normal operation | Direct Yahoo disabled by default (`directYahooEnabled: false`) |
| Source shown as "Live / online" | Source shown as Worker / Yahoo direct / Embedded |
| Score history chart showed all 6,758 days | Score charts now show the trailing 252 sessions |

### Migration steps

1. Deploy `worker.js` to Cloudflare (free plan is enough). Copy its base URL.
2. Set that URL in the dashboard:
   - easiest: edit `index.html` on GitHub, replace `workerBase: ''` with
     `workerBase: 'https://...workers.dev'`, commit; or
   - locally: edit `DEPLOY_CONFIG.workerBase` in `dashboard_template.html`,
     run `python build_dashboard.py`, upload the new `index.html`.
3. Remove any old custom `proxyEndpoint` value if you had one. It is no longer
   used by the new provider chain.
4. Upload only `index.html` to the repository root. GitHub Pages rebuilds.
5. Open **Diagnostics → Data Provider Status → Re-check providers**. Worker
   must be **PASS**; Yahoo direct **FAIL/disabled** is expected and correct.
6. Press **Test Live Market Data**. All six symbols should pass through Worker.
7. Press **Generate Latest Score** and confirm "Current provider: Worker" with
   100% live confidence.
8. Update the end-user link if the repository or custom domain changed.

### Rollback

If the Worker is unavailable, no code change is needed: the dashboard falls
back to the embedded dataset and shows the friendly "Live market quotes are
currently unavailable" message. To temporarily use another Worker, either
change `workerBase` or append
`?worker=https%3A%2F%2Fother-worker.example.com` to the page URL.

---

## 3. Troubleshooting matrix

| Symptom | Likely cause | Fix |
|---|---|---|
| Worker FAIL, health URL unreachable | Worker not deployed / wrong URL | Redeploy `worker.js`, re-copy base URL |
| Worker PASS but symbols fail | Yahoo 429 or upstream change | Check Worker logs; increase `CACHE_TTL_SECONDS` |
| Yahoo direct FAIL | Expected — CORS disabled | Ignore; Worker is the supported path |
| Embedded dataset Available | Always true | No action |
| Current provider = Embedded | Worker failed for all symbols | Check Worker status and Test Live output |
| Partial retrieval | Some symbols cached/failed individually | Inspect per-symbol URL/status/latency |
| 429 from Worker | Yahoo rate limit | Wait; cached responses recover in ~10 min |
| Symbol mismatch | Worker/dashboard symbol lists diverged | Update both `ALLOWED_SYMBOLS` and `DATA_CONFIG.symbols` |
| Charts blank | ECharts CDN blocked | Internet required only for charts |
| Score chart too long | Older build | Confirm version shows "Trailing 12 months" title |

---

## 4. Files and roles after migration

| File | Role | Uploaded to GitHub Pages? | Deployed to Cloudflare? |
|---|---|---|---|
| `index.html` | published dashboard | **Yes** (only file needed) | No |
| `ETF_Timing_Dashboard.html` | same app, local copy | No | No |
| `worker.js` | market-data proxy | No | **Yes** |
| `dashboard_template.html` | source with `workerBase` config | No | No |
| `build_dashboard.py` | rebuild helper | No | No |
| `dashboard_data.json`, CSVs | build inputs | No | No |
| `test_dashboard_v2.js`, `test_worker.js` | validation | No | No |
| `CLOUDFLARE_WORKER_GUIDE.md` | deployment steps | Optional | No |
| `GITHUB_PAGES_GUIDE.md` | publishing steps | Optional | No |
| `END_USER_GUIDE.md` | user instructions | Optional | No |
| `Launch Dashboard.cmd`, `serve_dashboard.ps1` | Windows local launcher | No | No |
