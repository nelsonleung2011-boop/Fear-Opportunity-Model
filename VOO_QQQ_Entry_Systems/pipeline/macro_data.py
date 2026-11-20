#!/usr/bin/env python3
"""
Point-in-time macro / valuation / sector data layer for ETF Entry Decision System v4.

CRITICAL DESIGN RULE
--------------------
Macro and valuation series are NOT aligned "as of" their nominal observation
date. Every observation is mapped to its FIRST-KNOWN date = nominal date +
publication lag, then forward-filled onto the daily trading calendar. A value
can therefore never influence a feature before it could physically have been
known. This is the only alignment that satisfies "no future information".

Revision risk is real and NOT fully removable with these sources: FRED and
multpl serve the LATEST REVISED vintage, not real-time vintages. The engine
quantifies this with a with/without sensitivity test and discloses it.

Sources (all public, no API key):
  * multpl.com monthly tables  -- Shiller CAPE, trailing P/E, earnings yield,
                                  dividend yield (derived from Shiller data)
  * FRED fredgraph.csv         -- WALCL, M2SL, NFCI, GDP
  * Yahoo chart API            -- ^W5000 (Wilshire 5000), SOXX, SMH prices
  * Yale Shiller ie_data.xls   -- 2023 CAPE vintage, used ONLY as an audit
                                  benchmark for historical revision magnitude

Outputs (cached, so the engine is reproducible offline):
  data/macro/macro_observations.json   native dated observations
  data/macro/provenance.json           source, coverage, publication lag
  data/raw/SOXX.csv, SMH.csv, W5000.csv
"""
from __future__ import annotations

import html
import io
import json
import re
import urllib.request
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MACRO = ROOT / "data" / "macro"
RAW = ROOT / "data" / "raw"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# ---------------------------------------------------------------------------
# Publication lags, in CALENDAR days after the nominal observation date.
# Deliberately conservative: a longer lag discards real information but can
# never introduce look-ahead. Values are documented in the provenance file.
# ---------------------------------------------------------------------------
LAGS = {
    # multpl / Shiller: quarterly earnings are reported with a lag and CAPE
    # uses a 10-year trailing real-earnings average. One full quarter is safe.
    "CAPE": 95, "PE_TRAILING": 95, "EARNINGS_YIELD": 95, "DIVIDEND_YIELD": 95,
    # FRED GDP: quarter label = quarter START; BEA advance estimate is released
    # ~4 weeks after quarter END (= ~120 days after quarter start).
    "GDP": 130,
    # FRED WALCL: weekly Wednesday snapshot released with the H.4.1 on Thursday.
    "WALCL": 8,
    # FRED M2SL: monthly money stock released with roughly a 6-8 week lag.
    "M2SL": 75,
    # Chicago Fed NFCI: weekly, published the following week.
    "NFCI": 10,
    # Yahoo ^W5000: daily index level, known at the close (use +1 day buffer).
    "W5000": 1,
    # Daily ETF closes: known at the close of the same session.
    "ETF_CLOSE": 0,
}

FRED_SERIES = {
    "WALCL": "Federal Reserve total assets, weekly (millions USD)",
    "M2SL": "M2 money stock, monthly, seasonally adjusted (billions USD)",
    "NFCI": "Chicago Fed National Financial Conditions Index, weekly",
    "GDP": "Nominal GDP, quarterly, SAAR (billions USD)",
}
MULTPL = {
    "CAPE": ("https://www.multpl.com/shiller-pe/table/by-month", "month_start",
             "Shiller cyclically adjusted P/E (CAPE)"),
    "DIVIDEND_YIELD": ("https://www.multpl.com/s-p-500-dividend-yield/table/by-month", "month_end",
                       "S&P 500 dividend yield, percent"),
    "PE_TRAILING": ("https://www.multpl.com/s-p-500-pe-ratio/table/by-month", "month_start",
                    "S&P 500 trailing P/E (SUBSTITUTE; forward P/E unavailable)"),
    "EARNINGS_YIELD": ("https://www.multpl.com/s-p-500-earnings-yield/table/by-month", "month_start",
                       "S&P 500 trailing earnings yield, percent"),
}
YAHOO_NEW = {
    "SOXX": "SOXX", "SMH": "SMH", "W5000": "%5EW5000",
}


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_fred(series_id: str) -> list[dict]:
    txt = http_get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + series_id).decode(
        "utf-8", "ignore").strip().splitlines()
    out = []
    for line in txt[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        d, v = parts[0].strip(), parts[-1].strip()
        if v in (".", "", "NA"):
            continue
        try:
            out.append({"date": d, "value": float(v)})
        except ValueError:
            continue
    return out


def fetch_multpl(url: str, convention: str) -> list[dict]:
    s = http_get(url).decode("utf-8", "ignore")
    rows = re.findall(r"<tr[^>]*>\s*<td[^>]*>\s*([^<]+?)\s*</td>\s*"
                      r"<td[^>]*>\s*([^<]+?)\s*</td>", s)
    out = []
    for a, b in rows:
        ds = html.unescape(a).strip()
        vs = re.sub(r"[^\d.\-]", "", html.unescape(b))
        try:
            d = dt.datetime.strptime(ds, "%b %d, %Y").date()
            v = float(vs)
        except (ValueError, TypeError):
            continue
        # Only keep canonical monthly rows. multpl adds a partial current-month
        # row (e.g. "Nov 18, 2026") which must NOT be treated as a full month.
        if convention == "month_start" and d.day > 7:
            continue
        if convention == "month_end" and d.day < 25:
            continue
        out.append({"date": d.isoformat(), "value": v})
    # dedupe, keep the most recent value for a duplicated date
    seen = {}
    for r in out:
        seen[r["date"]] = r["value"]
    return [{"date": k, "value": v} for k, v in sorted(seen.items())]


def fetch_yahoo(symbol: str, provider_symbol: str) -> list[dict]:
    p2 = int(dt.datetime(2026, 11, 19, tzinfo=dt.timezone.utc).timestamp())
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/" + provider_symbol
           + "?period1=946684800&period2=%d&interval=1d&events=div%%2Csplit" % p2)
    j = json.loads(http_get(url).decode("utf-8", "ignore"))
    r0 = j["chart"]["result"][0]
    ts = r0["timestamp"]
    quote = (r0["indicators"].get("quote") or [{}])[0].get("close") or []
    adj = (r0["indicators"].get("adjclose") or [{}])[0].get("adjclose") or []
    rows = []
    for i, t in enumerate(ts):
        d = dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat()
        c = quote[i] if i < len(quote) else None
        a = adj[i] if i < len(adj) else c
        if c is None or not np.isfinite(c):
            continue
        rows.append({"date": d, "close": float(c), "adj_close": float(a) if a and np.isfinite(a) else float(c)})
    return rows


def build(refresh: bool = False) -> dict:
    MACRO.mkdir(parents=True, exist_ok=True)
    obs_file = MACRO / "macro_observations.json"
    prov_file = MACRO / "provenance.json"
    if obs_file.exists() and prov_file.exists() and not refresh:
        return json.loads(prov_file.read_text(encoding="utf-8"))

    obs, prov = {}, {}
    for sid, desc in FRED_SERIES.items():
        rows = fetch_fred(sid)
        obs[sid] = rows
        prov[sid] = {"source": "FRED fredgraph.csv", "description": desc,
                     "n": len(rows), "first": rows[0]["date"], "last": rows[-1]["date"],
                     "nominal_frequency": "quarterly" if sid == "GDP" else
                                          ("monthly" if sid == "M2SL" else "weekly"),
                     "publication_lag_days": LAGS[sid],
                     "vintage": "LATEST REVISED (not real-time; see audit)"}
        print(f"[macro] FRED {sid:6s} {len(rows):5d} rows {rows[0]['date']} -> {rows[-1]['date']}")

    for name, (url, conv, desc) in MULTPL.items():
        rows = fetch_multpl(url, conv)
        obs[name] = rows
        prov[name] = {"source": "multpl.com (Shiller-derived) monthly table",
                      "description": desc, "n": len(rows), "first": rows[0]["date"],
                      "last": rows[-1]["date"], "nominal_frequency": "monthly",
                      "date_convention": conv, "publication_lag_days": LAGS[name],
                      "vintage": "LATEST REVISED (not real-time; see audit)"}
        print(f"[macro] multpl {name:16s} {len(rows):5d} rows {rows[0]['date']} -> {rows[-1]['date']}")

    w = fetch_yahoo("W5000", YAHOO_NEW["W5000"])
    obs["W5000"] = w
    pd.DataFrame(w)[["date", "close", "adj_close"]].to_csv(RAW / "W5000.csv", index=False)
    prov["W5000"] = {"source": "Yahoo ^W5000 (Wilshire 5000 Total Market Index, price)",
                     "description": "broad-market price index for the market-cap/GDP proxy",
                     "n": len(w), "first": w[0]["date"], "last": w[-1]["date"],
                     "nominal_frequency": "daily", "publication_lag_days": LAGS["W5000"],
                     "caveat": "price index, NOT total market capitalisation; used as a proportional proxy"}

    # Sector ETFs that were missing from the offline dataset. Write in exactly
    # the same schema as the existing data/raw/*.csv files (date, close, adj_close).
    for sym in ("SOXX", "SMH"):
        rows = fetch_yahoo(sym, YAHOO_NEW[sym])
        pd.DataFrame(rows)[["date", "close", "adj_close"]].to_csv(RAW / f"{sym}.csv", index=False)
        prov[sym] = {"source": "Yahoo chart API, dividend-adjusted",
                     "description": f"{sym} sector ETF daily prices",
                     "n": len(rows), "first": rows[0]["date"], "last": rows[-1]["date"],
                     "nominal_frequency": "daily", "publication_lag_days": LAGS["ETF_CLOSE"],
                     "caveat": "surviving fund only; no delisted sector funds in universe"}
        print(f"[macro] Yahoo {sym:6s} {len(rows):5d} rows {rows[0]['date']} -> {rows[-1]['date']}")

    obs_file.write_text(json.dumps(obs, separators=(",", ":")), encoding="utf-8")
    prov["_meta"] = {
        "built_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "alignment": "first-known-date = nominal date + publication_lag_days, then ffill",
        "excluded_candidates": {
            "FORWARD_PE": "No free historical real-time forward-earnings series exists; "
                          "fabricating a proxy would violate the no-lookahead rule. EXCLUDED.",
            "MARKET_CAP_TO_GDP": "Mathematically identical to the Buffett Indicator; "
                                 "counted once as val_buffett_proxy.",
            "SHILLER_XLS": "Yale ie_data.xls is stale (ends 2023-09); retained only as a "
                           "2023 CAPE vintage for revision-magnitude audit.",
        },
    }
    prov_file.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    return prov


def pit_daily(index: pd.DatetimeIndex, rows: list[dict], lag_days: int,
              value_key: str = "value") -> pd.Series:
    """Align a native-dated series onto a daily calendar using FIRST-KNOWN dates.

    No observation is visible before ``nominal_date + lag``. Values are
    forward-filled only AFTER that date, so an expanded-history rebuild cannot
    change any earlier feature value.
    """
    dates, vals = [], []
    for r in rows:
        v = r.get(value_key)
        if v is None or not np.isfinite(float(v)):
            continue
        d = pd.Timestamp(r["date"]) + pd.Timedelta(days=int(lag_days))
        dates.append(d)
        vals.append(float(v))
    if not dates:
        return pd.Series(np.nan, index=index, dtype=float)
    s = pd.Series(vals, index=pd.DatetimeIndex(dates)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    combined = index.union(s.index)
    return s.reindex(combined).ffill().reindex(index)


if __name__ == "__main__":
    import sys
    build(refresh="--refresh" in sys.argv)
