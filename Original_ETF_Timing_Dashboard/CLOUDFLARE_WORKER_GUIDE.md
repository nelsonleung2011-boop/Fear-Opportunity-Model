# Cloudflare Worker deployment guide

This guide connects the dashboard to live market data using `worker.js`.
It takes about 5 minutes and uses the Cloudflare **free** plan.

## What changes

Before:

```
Browser (GitHub Pages) --X--> Yahoo Finance   (blocked by CORS)
```

After:

```
Browser (GitHub Pages) --> Cloudflare Worker --> Yahoo Finance
                          (CORS + cache + retry)
```

The Worker is public and read-only. It accepts only the six symbols the
dashboard uses, caches each response for 10 minutes, and never sees user data.

---

## 1. Deploy `worker.js`

1. Sign in to <https://dash.cloudflare.com/>.
2. Open **Workers & Pages → Create application → Workers → Create Worker**.
3. Name it, for example, `etf-market-data`, then choose **Deploy**.
4. Click **Edit code**, delete the sample code, paste the entire contents of
   `worker.js`, then click **Deploy**.
5. Cloudflare shows a URL such as:

   ```
   https://etf-market-data.your-subdomain.workers.dev
   ```

   Copy it. This is your Worker base URL (no trailing `/`).

6. Test it in a browser:

   - `https://etf-market-data.your-subdomain.workers.dev/health`
   - `https://etf-market-data.your-subdomain.workers.dev/chart?symbol=SPY&period1=1704067200&period2=1735689600`

   The first returns JSON with `"ok": true`. The second returns JSON with
   `"ok": true` and a `data` array of `{date, close}` pairs.

### Custom domain (optional)

If you do not want a `workers.dev` address, add a custom domain in
**Worker → Settings → Domains & Routes**, then use that domain as the Worker
base URL.

---

## 2. Connect the dashboard to the Worker

### Option A — edit the published `index.html` on GitHub (easiest)

1. In your GitHub repository, open `index.html`.
2. Press the pencil (Edit) icon.
3. Find this line near the top of the script:

   ```js
   workerBase: '',
   ```

4. Replace it with your Worker URL:

   ```js
   workerBase: 'https://etf-market-data.your-subdomain.workers.dev',
   ```

5. Commit the change. GitHub Pages rebuilds automatically in 1–2 minutes.

### Option B — edit the local template and rebuild (maintainers)

1. In `dashboard_template.html`, set `DEPLOY_CONFIG.workerBase` to the Worker URL.
2. Run `python build_dashboard.py` once. This refreshes both
   `ETF_Timing_Dashboard.html` and `index.html`.
3. Upload the new `index.html` to GitHub.

### Option C — temporary test without editing code

Append the Worker URL to the page URL:

```
https://your-name.github.io/your-repo/?worker=https%3A%2F%2Fetf-market-data.your-subdomain.workers.dev
```

This is useful for testing before committing, but end users should use a
properly configured `index.html`.

---

## 3. Verify

1. Open the hosted dashboard.
2. Go to **Diagnostics → Data Provider Status** and press **Re-check providers**.

   Expected:

   | Card | Expected |
   |---|---|
   | Cloudflare Worker | **PASS** |
   | Yahoo direct | **FAIL / disabled by default** (expected) |
   | Embedded dataset | **Available** |
   | Current provider in use | **Worker** after a score/test |

3. Press **Test Live Market Data**. Each of the six symbols should show:

   - Source: **Worker**
   - URL Called: the real `/chart?symbol=...` request
   - Response: HTTP 200
   - Latency: milliseconds

4. Return to **Date Explorer** and press **Generate Latest Score**. The status
   line should read "Current provider: Worker" with 100% live confidence.

---

## Worker API reference

| Endpoint | Description |
|---|---|
| `GET /health` | Worker version, provider, cache TTL, allowed symbols |
| `GET /chart?symbol=SPY&period1=...&period2=...` | Daily closes for one symbol |
| `OPTIONS *` | CORS preflight |

### Successful response

```json
{
  "ok": true,
  "provider": "Yahoo Finance",
  "via": "Cloudflare Worker",
  "symbol": "^VIX",
  "data": [{ "date": "2024-01-02", "close": 13.2 }],
  "retrievedAt": "2026-11-17T00:00:00.000Z",
  "latencyMs": 412,
  "attempts": 1,
  "cached": false,
  "cacheTtlSeconds": 600,
  "upstreamUrl": "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?..."
}
```

### Error response

```json
{
  "ok": false,
  "error": { "code": "UPSTREAM_HTTP_429", "message": "Yahoo Finance returned HTTP 429.", "status": 429 },
  "symbol": "SPY"
}
```

| Error code | Meaning |
|---|---|
| `SYMBOL_NOT_ALLOWED` | Symbol is not one of the six allowed symbols |
| `INVALID_DATE_RANGE` | Missing/invalid `period1` or `period2` |
| `UPSTREAM_HTTP_429` | Yahoo rate limit (retried inside the Worker) |
| `UPSTREAM_TIMEOUT` | Yahoo did not answer within 8 seconds (retried) |
| `UPSTREAM_SYMBOL_MISMATCH` | Yahoo returned a different symbol — rejected |
| `UPSTREAM_NO_DATA` | No numeric daily closes returned |
| `ROUTE_NOT_FOUND` | Unknown path |

---

## Caching and rate limits

- `Cache-Control: max-age=600` (10 minutes) on every successful `/chart` response.
- The Worker also stores the payload in Cloudflare's Cache API and will not
  serve a cached body older than 15 minutes.
- Up to 3 upstream attempts per request with 200/400 ms backoff.
- Six symbols per score = six Worker requests, but each is cached independently.

The Worker is intentionally stateless: no KV, D1, secrets, or API keys required.

## Security notes

- Only GET and OPTIONS are allowed.
- Only `SPY`, `QQQ`, `^VIX`, `TLT`, `^TNX`, `^IRX` are accepted.
- CORS is open (`*`) because the data is public and the endpoint is read-only.
  Restrict `Access-Control-Allow-Origin` to your GitHub Pages domain if you
  want to lock it down.

## Troubleshooting

| Symptom | Action |
|---|---|
| Worker status FAIL | Confirm `/health` works in a browser; confirm URL has no trailing slash |
| HTTP 429 often | Yahoo rate limiting — increase cache TTL in `worker.js` (`CACHE_TTL_SECONDS`) |
| Symbol mismatch | Do not change symbols in `DATA_CONFIG` without updating `ALLOWED_SYMBOLS` in `worker.js` |
| Charts blank | ECharts CDN blocked — scores and tables still work |
| Live data unavailable | Expected fallback if the Worker is not deployed; no action needed for end users |
