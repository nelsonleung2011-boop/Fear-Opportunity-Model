/**
 * ETF Timing Dashboard - market data proxy (Cloudflare Worker)
 * ------------------------------------------------------------
 * Browser -> Cloudflare Worker -> Yahoo Finance
 *
 * Why this exists: Yahoo's chart endpoint does not send browser-friendly CORS
 * headers, so a GitHub Pages site cannot read it directly. This Worker fetches
 * Yahoo server-side, verifies the returned symbol, caches the result, and
 * returns clean JSON with CORS headers.
 *
 * Deploy: see CLOUDFLARE_WORKER_GUIDE.md
 * Endpoints:
 *   GET /health                                  -> worker/provider health
 *   GET /chart?symbol=SPY&period1=..&period2=..  -> daily closes
 *   OPTIONS *                                    -> CORS preflight
 */

const WORKER_VERSION = '1.0.0';
const YAHOO_BASE = 'https://query1.finance.yahoo.com/v8/finance/chart/';
const CACHE_TTL_SECONDS = 600;      // 10 minutes (requirement: 5-15 minutes)
const MAX_CACHE_AGE_SECONDS = 900;  // never serve a cached body older than 15 minutes
const UPSTREAM_TIMEOUT_MS = 8000;
const MAX_ATTEMPTS = 3;

const ALLOWED_SYMBOLS = new Set([
  'SPY', 'QQQ', '^VIX', 'TLT', '^TNX', '^IRX'
]);

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Accept',
  'Access-Control-Max-Age': '86400'
};

class ProxyError extends Error {
  constructor(code, message, status) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

export default {
  async fetch(request, env, ctx) {
    try {
      return await handleRequest(request);
    } catch (err) {
      const e = err instanceof ProxyError
        ? err
        : new ProxyError('WORKER_EXCEPTION', String((err && err.message) || err || 'unknown error'), 500);
      return errorResponse(e.code, e.message, e.status, null, true);
    }
  }
};

async function handleRequest(request) {
  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }
  if (request.method !== 'GET') {
    return errorResponse('METHOD_NOT_ALLOWED', 'Only GET and OPTIONS are supported.', 405, null, false);
  }

  const url = new URL(request.url);
  if (url.pathname === '/health' || url.pathname === '/') {
    return jsonResponse({
      ok: true,
      worker: 'ETF Timing Dashboard data proxy',
      version: WORKER_VERSION,
      provider: 'Yahoo Finance',
      cacheTtlSeconds: CACHE_TTL_SECONDS,
      symbols: Array.from(ALLOWED_SYMBOLS),
      time: new Date().toISOString()
    }, 200, { 'Cache-Control': 'no-store' });
  }

  if (url.pathname !== '/chart') {
    return errorResponse('ROUTE_NOT_FOUND', 'Unknown path. Use /chart or /health.', 404, null, false);
  }

  const symbol = normaliseSymbol(url.searchParams.get('symbol') || '');
  const p1 = Number(url.searchParams.get('period1'));
  const p2 = Number(url.searchParams.get('period2'));

  if (!ALLOWED_SYMBOLS.has(symbol)) {
    return errorResponse('SYMBOL_NOT_ALLOWED',
      'Unsupported symbol "' + symbol + '". Allowed: ' + Array.from(ALLOWED_SYMBOLS).join(', '),
      400, symbol, false);
  }
  if (!Number.isFinite(p1) || !Number.isFinite(p2) || p1 <= 0 || p2 <= p1) {
    return errorResponse('INVALID_DATE_RANGE',
      'period1 and period2 must be positive Unix seconds with period2 > period1.',
      400, symbol, false);
  }

  const upstreamUrl = YAHOO_BASE + encodeURIComponent(symbol) +
    '?period1=' + Math.floor(p1) + '&period2=' + Math.floor(p2) +
    '&interval=1d&events=history';
  const cacheKey = new Request(upstreamUrl, { method: 'GET' });
  const cache = getCache();

  if (cache) {
    try {
      const cached = await cache.match(cacheKey);
      if (cached) {
        const generatedAt = Number(cached.headers.get('X-Generated-At') || 0);
        const ageSeconds = generatedAt ? Math.floor((Date.now() - generatedAt) / 1000) : MAX_CACHE_AGE_SECONDS + 1;
        if (ageSeconds <= MAX_CACHE_AGE_SECONDS) {
          const payload = await cached.clone().json();
          payload.cached = true;
          payload.cacheAgeSeconds = ageSeconds;
          return jsonResponse(payload, 200, {
            'Cache-Control': 'public, max-age=' + Math.max(60, CACHE_TTL_SECONDS - ageSeconds),
            'X-Worker-Cache': 'HIT',
            'X-Cache-Age': String(ageSeconds)
          });
        }
        await cache.delete(cacheKey);
      }
    } catch (e) {
      // Cache failures must never break the data path.
    }
  }

  const upstream = await fetchWithRetry(upstreamUrl, symbol);
  const rows = normaliseRows(upstream.json, symbol);
  const payload = {
    ok: true,
    provider: 'Yahoo Finance',
    via: 'Cloudflare Worker',
    workerVersion: WORKER_VERSION,
    requestedSymbol: symbol,
    symbol: symbol,
    data: rows,
    retrievedAt: new Date().toISOString(),
    latencyMs: upstream.latencyMs,
    attempts: upstream.attempts,
    cached: false,
    cacheAgeSeconds: 0,
    cacheTtlSeconds: CACHE_TTL_SECONDS,
    upstreamUrl: upstreamUrl,
    upstreamStatus: upstream.status
  };

  const response = jsonResponse(payload, 200, {
    'Cache-Control': 'public, max-age=' + CACHE_TTL_SECONDS,
    'X-Worker-Cache': 'MISS'
  });

  if (cache) {
    try {
      await cache.put(cacheKey, new Response(JSON.stringify(payload), {
        status: 200,
        headers: {
          'Content-Type': 'application/json; charset=utf-8',
          'Cache-Control': 'public, max-age=' + CACHE_TTL_SECONDS,
          'X-Generated-At': String(Date.now())
        }
      }));
    } catch (e) {
      // Best-effort caching only.
    }
  }
  return response;
}

/** Fetch Yahoo with retries on timeout, 429 and 5xx responses. */
async function fetchWithRetry(url, symbol) {
  let lastError = null;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const started = Date.now();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
    try {
      const resp = await fetch(url, {
        method: 'GET',
        signal: controller.signal,
        headers: {
          'Accept': 'application/json,text/plain,*/*',
          'User-Agent': 'Mozilla/5.0 (compatible; ETFDashboardProxy/' + WORKER_VERSION + ')',
          'Referer': 'https://finance.yahoo.com/'
        }
      });
      const text = await resp.text();
      if (resp.ok) {
        let json;
        try {
          json = JSON.parse(text);
        } catch (e) {
          throw new ProxyError('UPSTREAM_INVALID_JSON', 'Yahoo returned invalid JSON.', 502);
        }
        const chartError = json && json.chart && json.chart.error;
        if (chartError) {
          const code = String(chartError.code || 'UPSTREAM_CHART_ERROR');
          const description = String(chartError.description || 'Yahoo chart API error');
          const status = /not found|not exist|invalid symbol/i.test(description) ? 404 : 422;
          throw new ProxyError('UPSTREAM_' + code.toUpperCase().replace(/[^A-Z0-9]+/g, '_'),
            description, status);
        }
        return {
          json: json,
          status: resp.status,
          attempts: attempt,
          latencyMs: Date.now() - started
        };
      }

      lastError = new ProxyError('UPSTREAM_HTTP_' + resp.status,
        'Yahoo Finance returned HTTP ' + resp.status + '.', resp.status === 429 ? 429 : (resp.status >= 500 ? 502 : 422));
      if (resp.status === 404 || (resp.status >= 400 && resp.status < 500 && resp.status !== 429)) break;
    } catch (err) {
      if (err instanceof ProxyError && err.status !== 502 && err.status !== 429) throw err;
      lastError = err instanceof ProxyError
        ? err
        : new ProxyError('UPSTREAM_TIMEOUT', 'Yahoo request timed out or the network was unreachable.', 504);
    } finally {
      clearTimeout(timer);
    }
    if (attempt < MAX_ATTEMPTS) await delay(200 * Math.pow(2, attempt - 1));
  }
  throw lastError || new ProxyError('UPSTREAM_FAILED', 'Yahoo request failed.', 502);
}

/** Convert Yahoo chart JSON to [{date, close}] and verify the returned symbol. */
function normaliseRows(json, expectedSymbol) {
  const result = json && json.chart && Array.isArray(json.chart.result) ? json.chart.result[0] : null;
  if (!result) throw new ProxyError('UPSTREAM_BAD_SHAPE', 'Yahoo response has no chart result.', 502);

  const reported = result.meta && result.meta.symbol ? String(result.meta.symbol) : null;
  if (!reported) throw new ProxyError('UPSTREAM_SYMBOL_MISSING', 'Yahoo response has no meta.symbol.', 502);
  if (normaliseSymbol(reported) !== normaliseSymbol(expectedSymbol)) {
    throw new ProxyError('UPSTREAM_SYMBOL_MISMATCH',
      'Requested ' + expectedSymbol + ' but Yahoo returned ' + reported + '.', 502);
  }

  const timestamps = Array.isArray(result.timestamp) ? result.timestamp : [];
  const quote = result.indicators && result.indicators.quote && result.indicators.quote[0];
  const closes = quote && Array.isArray(quote.close) ? quote.close : [];
  const byDate = new Map();
  for (let i = 0; i < timestamps.length; i++) {
    const ts = timestamps[i], close = closes[i];
    if (typeof ts !== 'number' || typeof close !== 'number' || !Number.isFinite(close)) continue;
    const date = new Date(ts * 1000).toISOString().slice(0, 10);
    byDate.set(date, close);
  }
  const rows = Array.from(byDate.entries())
    .sort((a, b) => (a[0] < b[0] ? -1 : 1))
    .map(([date, close]) => ({ date: date, close: close }));
  if (!rows.length) throw new ProxyError('UPSTREAM_NO_DATA', 'Yahoo returned no numeric daily closes.', 502);
  return rows;
}

function normaliseSymbol(symbol) {
  let s = String(symbol || '').trim();
  try { s = decodeURIComponent(s); } catch (e) { /* keep raw value */ }
  return s.toUpperCase();
}

function getCache() {
  try { return caches.default; } catch (e) { return null; }
}

function jsonResponse(body, status, extraHeaders) {
  const headers = Object.assign({
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Proxy-Version': WORKER_VERSION,
    'X-Data-Provider': 'Yahoo Finance via Cloudflare Worker'
  }, CORS_HEADERS, extraHeaders || {});
  return new Response(JSON.stringify(body), { status: status, headers: headers });
}

function errorResponse(code, message, status, symbol, cacheable) {
  return jsonResponse({
    ok: false,
    provider: 'Yahoo Finance',
    via: 'Cloudflare Worker',
    symbol: symbol || null,
    error: { code: code, message: message, status: status },
    time: new Date().toISOString()
  }, status, { 'Cache-Control': cacheable ? 'public, max-age=30' : 'no-store' });
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
