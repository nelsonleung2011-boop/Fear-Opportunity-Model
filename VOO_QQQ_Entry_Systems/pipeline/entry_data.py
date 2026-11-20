#!/usr/bin/env python3
"""
v5 Entry Quality System -- additional point-in-time data.

This module does NOT change anything in v2/v3/v4. It only adds the two macro
series the new framework needs but v4 did not cache:

  * DFII10         10-year TIPS real yield             (FRED)
  * BAMLH0A0HYM2   ICE BofA US High Yield OAS          (FRED)
  * T10Y2Y         10y-2y Treasury curve               (FRED)

and, for context only:

  * FNG            CNN Fear & Greed Index              (CNN, ~1y history)

All observations are aligned by FIRST-KNOWN DATE = nominal date + publication
lag, exactly like pipeline/macro_data.py. FRED serves latest-revised vintages;
this is disclosed rather than hidden.

If a source is unavailable, the series is cached as EMPTY with an explicit
reason. The engine then falls back to a documented price-based proxy instead
of silently substituting numbers. Never fabricate a history.
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MACRO = ROOT / "data" / "macro"
OBS = MACRO / "entry_observations.json"
PROV = MACRO / "entry_provenance.json"

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
}

# Calendar days after the nominal observation date. Conservative on purpose:
# a longer lag throws away information but cannot introduce look-ahead.
LAGS = {"DFII10": 1, "BAMLH0A0HYM2": 1, "T10Y2Y": 1, "FNG": 0}

FRED_SERIES = {
    "DFII10": "10-Year Treasury Inflation-Indexed Security, Constant Maturity (percent)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option-Adjusted Spread (percent)",
    "T10Y2Y": "10-Year Treasury Constant Maturity minus 2-Year (percentage points)",
}


def http_get(url: str, timeout: int = 25, headers: dict | None = None) -> bytes:
    h = dict(UA)
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_fred(series_id: str, attempts: int = 3, pause: float = 0.6) -> list[dict]:
    """FRED fredgraph.csv with retry/backoff. Empty on failure, never guessed."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + series_id
    for i in range(attempts):
        try:
            txt = http_get(url, headers={"User-Agent": "curl/8.4.0",
                                         "Accept": "text/csv,*/*"}).decode("utf-8", "ignore").strip().splitlines()
            out = []
            for line in txt[1:]:
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                d, v = parts[0].strip(), parts[-1].strip()
                if v in (".", "", "NA") or not d:
                    continue
                try:
                    out.append({"date": d, "value": float(v)})
                except ValueError:
                    continue
            if out:
                return out
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        if i < attempts - 1:
            time.sleep(pause * (i + 1))
    return []


def fetch_cnn_fng() -> list[dict]:
    """CNN Fear & Greed daily/weekly history. Context only (short history)."""
    try:
        raw = http_get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={"Referer": "https://www.cnn.com/",
                     "Accept": "application/json,text/plain,*/*"},
        ).decode("utf-8", "ignore")
        j = json.loads(raw)
        rows = []
        for p in j.get("fear_and_greed_historical", {}).get("data", []):
            x, y = p.get("x"), p.get("y")
            if x is None or y is None or not np.isfinite(float(y)):
                continue
            d = dt.datetime.fromtimestamp(float(x) / 1000.0, dt.timezone.utc).date().isoformat()
            rows.append({"date": d, "value": float(y)})
        return rows
    except Exception:
        return []


def build(refresh: bool = False) -> dict:
    MACRO.mkdir(parents=True, exist_ok=True)
    if OBS.exists() and PROV.exists() and not refresh:
        return json.loads(PROV.read_text(encoding="utf-8"))

    obs: dict[str, list[dict]] = {}
    prov: dict = {}
    for sid, desc in FRED_SERIES.items():
        rows = fetch_fred(sid)
        obs[sid] = rows
        prov[sid] = {
            "source": "FRED fredgraph.csv",
            "description": desc,
            "publication_lag_days": LAGS[sid],
            "n": len(rows),
            "first": rows[0]["date"] if rows else None,
            "last": rows[-1]["date"] if rows else None,
            "vintage": "LATEST REVISED (not real-time)",
            "status": "OK" if rows else "UNAVAILABLE -- price proxy used by engine",
        }
        print(f"[entry-data] FRED {sid:14s} {len(rows):5d} rows", flush=True)
        time.sleep(0.6)

    fng = fetch_cnn_fng()
    obs["FNG"] = fng
    prov["FNG"] = {
        "source": "CNN production.dataviz.cnn.io fearandgreed/graphdata",
        "description": "CNN Fear & Greed Index",
        "publication_lag_days": LAGS["FNG"],
        "n": len(fng),
        "first": fng[0]["date"] if fng else None,
        "last": fng[-1]["date"] if fng else None,
        "status": "OK (context only)" if len(fng) >= 60 else "INSUFFICIENT HISTORY",
        "usage": "Displayed as current context ONLY. Never scored: ~1 year of history "
                 "cannot support a since-2000 percentile.",
    }
    print(f"[entry-data] CNN FNG {len(fng)} rows", flush=True)

    prov["_meta"] = {
        "built_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "alignment": "first-known-date = nominal date + publication_lag_days, then ffill",
        "fallback_rule": "An unavailable or short series is NEVER back-filled or invented. "
                         "The engine uses a documented price-based proxy and labels it.",
    }
    OBS.write_text(json.dumps(obs, separators=(",", ":")), encoding="utf-8")
    PROV.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    return prov


if __name__ == "__main__":
    import sys
    build(refresh="--refresh" in sys.argv)
