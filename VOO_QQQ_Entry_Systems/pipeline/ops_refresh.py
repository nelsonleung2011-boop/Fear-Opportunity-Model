#!/usr/bin/env python3
"""Daily data refresh for the v7 operational pipeline.

Design rules (these matter more than the number of sources):

* A failing source NEVER overwrites good cached data with an empty or short
  series. Each source is validated before its file is replaced.
* Historical observations are MERGED by date, not blindly replaced, so a
  transient provider glitch cannot delete 25 years of history.
* FRED/multpl serve latest-revised vintages. That is disclosed in the Data tab;
  refreshing updates values for recent dates but is not a real-time-vintage feed.
* Yahoo ``adj_close`` is retroactively dividend-adjusted, so price files are
  replaced wholesale (when valid) instead of appended, to keep the series
  internally consistent.

No credentials are required. All endpoints are public.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import entry_data as ED
from pipeline import macro_data as MD
from pipeline import ops_price as OP

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW2 = ROOT / "data_raw"
MACRO = ROOT / "data" / "macro"
OPS = ROOT / "data" / "ops"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

PRICE_SYMBOLS = {
    "SPY": "SPY", "QQQ": "QQQ", "VIX": "%5EVIX", "TLT": "TLT",
    "TNX": "%5ETNX", "IRX": "%5EIRX", "HYG": "HYG", "LQD": "LQD",
    "RSP": "RSP", "IWM": "IWM", "W5000": "%5EW5000",
    "SOXX": "SOXX", "SMH": "SMH", "XLC": "XLC", "XLRE": "XLRE",
    "XLB": "XLB", "XLE": "XLE", "XLF": "XLF", "XLI": "XLI",
    "XLK": "XLK", "XLP": "XLP", "XLU": "XLU", "XLV": "XLV", "XLY": "XLY",
}
AUX_SYMBOLS = {
    "fvx": ("%5EFVX", RAW2 / "fvx.csv"),
    "ief": ("IEF", RAW2 / "ief.csv"),
    "shy": ("SHY", RAW2 / "shy.csv"),
    "_vix9d": ("%5EVIX9D", RAW2 / "_vix9d.csv"),
}


def http_get(url: str, timeout: int = 25, attempts: int = 3) -> bytes:
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Accept": "*/*",
                              "Referer": "https://finance.yahoo.com/"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            if i < attempts - 1:
                time.sleep(0.8 * (i + 1))
    raise RuntimeError(f"HTTP failed after {attempts} attempts: {last}")


def fetch_yahoo(provider_symbol: str) -> list[dict]:
    """Full daily history with a DYNAMIC end date (macro_data.py hard-codes one)."""
    end = int((dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)).timestamp())
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/" + provider_symbol
           + "?period1=946684800&period2=%d&interval=1d&events=div%%2Csplit" % end)
    j = json.loads(http_get(url).decode("utf-8", "ignore"))
    r0 = j["chart"]["result"][0]
    ts = r0["timestamp"]
    quote = (r0["indicators"].get("quote") or [{}])[0].get("close") or []
    adj = (r0["indicators"].get("adjclose") or [{}])[0].get("adjclose") or []
    rows = []
    for i, t in enumerate(ts):
        c = quote[i] if i < len(quote) else None
        a = adj[i] if i < len(adj) else c
        if c is None or not np.isfinite(c):
            continue
        d = dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat()
        rows.append({"date": d, "close": float(c),
                     "adj_close": float(a) if a is not None and np.isfinite(a) else float(c)})
    return rows


def _valid_replacement(rows: list[dict], old_len: int, old_last: str | None) -> tuple[bool, str]:
    """Guard against a provider returning a truncated or stale series."""
    if len(rows) < 100:
        return False, f"only {len(rows)} rows returned"
    if old_len and len(rows) < int(old_len * 0.95):
        return False, f"history shrank {old_len} -> {len(rows)}"
    new_last = rows[-1]["date"]
    if old_last and new_last < old_last:
        return False, f"last date went backwards {old_last} -> {new_last}"
    for r in rows[:3] + rows[-3:]:
        if not np.isfinite(r["close"]) or r["close"] <= 0:
            return False, "non-positive close in head/tail"
    return True, ""


def merge_csv(path: Path, rows: list[dict]) -> dict:
    """Validate then replace a price CSV. Returns a per-symbol status record."""
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["date", "close", "adj_close"])
    old_len, old_last = len(old), (str(old["date"].iloc[-1])[:10] if len(old) else None)
    ok, why = _valid_replacement(rows, old_len, old_last)
    if not ok:
        return {"status": "KEPT CACHE", "reason": why, "last": old_last}
    new = pd.DataFrame(rows)[["date", "close", "adj_close"]]
    new = new.drop_duplicates("date", keep="last").sort_values("date")
    path.write_text(new.to_csv(index=False), encoding="utf-8")
    return {"status": "UPDATED", "added": max(0, len(new) - old_len),
            "from": old_last, "last": str(new["date"].iloc[-1]),
            "rows": len(new)}


def merge_observations(old: list[dict], new: list[dict]) -> list[dict]:
    """Union by date; refreshed values win (they include vendor revisions)."""
    by_date = {str(r["date"])[:10]: dict(r) for r in old}
    for r in new:
        d = str(r["date"])[:10]
        merged = dict(by_date.get(d, {}))
        merged.update(r)
        by_date[d] = merged
    return [by_date[k] for k in sorted(by_date)]


def refresh_prices() -> dict:
    out = {}
    for name, sym in PRICE_SYMBOLS.items():
        path = RAW / f"{name}.csv"
        try:
            out[name] = merge_csv(path, fetch_yahoo(sym))
        except Exception as exc:
            last = None
            if path.exists():
                try:
                    last = str(pd.read_csv(path)["date"].iloc[-1])[:10]
                except Exception:
                    pass
            out[name] = {"status": "KEPT CACHE", "reason": type(exc).__name__ + ": " + str(exc)[:100],
                         "last": last}
        time.sleep(0.12)
    for name, (sym, path) in AUX_SYMBOLS.items():
        try:
            out[name] = merge_csv(path, fetch_yahoo(sym))
        except Exception as exc:
            out[name] = {"status": "KEPT CACHE", "reason": type(exc).__name__ + ": " + str(exc)[:100]}
        time.sleep(0.12)
    return out


def _fred(sid: str) -> list[dict]:
    for i in range(3):
        try:
            rows = MD.fetch_fred(sid)
            if len(rows) > 100:
                return rows
        except Exception:
            pass
        time.sleep(0.9 * (i + 1))
    return []


def refresh_macro() -> dict:
    """Refresh both observation caches and their provenance, per source."""
    report: dict = {}

    # --- macro_observations.json (v4 macro layer) ---
    obs_f, prov_f = MACRO / "macro_observations.json", MACRO / "provenance.json"
    obs = json.loads(obs_f.read_text(encoding="utf-8"))
    prov = json.loads(prov_f.read_text(encoding="utf-8"))
    for sid in MD.FRED_SERIES:
        try:
            rows = _fred(sid)
            before = len(obs.get(sid, []))
            if rows:
                obs[sid] = merge_observations(obs.get(sid, []), rows)
                report[sid] = {"status": "UPDATED", "added": len(obs[sid]) - before,
                               "last": obs[sid][-1]["date"]}
            else:
                report[sid] = {"status": "KEPT CACHE", "last": obs[sid][-1]["date"]}
        except Exception as exc:
            report[sid] = {"status": "KEPT CACHE", "reason": type(exc).__name__}
        time.sleep(0.2)
    for name, (url, conv, _desc) in MD.MULTPL.items():
        try:
            rows = MD.fetch_multpl(url, conv)
            before = len(obs.get(name, []))
            if len(rows) > 100:
                obs[name] = merge_observations(obs.get(name, []), rows)
                report[name] = {"status": "UPDATED", "added": len(obs[name]) - before,
                                "last": obs[name][-1]["date"]}
            else:
                report[name] = {"status": "KEPT CACHE", "last": obs[name][-1]["date"]}
        except Exception as exc:
            report[name] = {"status": "KEPT CACHE", "reason": type(exc).__name__}
        time.sleep(0.2)
    for name, sym in MD.YAHOO_NEW.items():
        # SOXX/SMH are stored as price CSVs, not in macro_observations.json.
        # W5000 is the only Yahoo series that also lives in the macro cache.
        if name not in obs:
            report[name] = {"status": "UPDATED VIA PRICE FILE",
                            "last": (RAW / f"{name}.csv").exists() and str(
                                pd.read_csv(RAW / f"{name}.csv")["date"].iloc[-1])[:10]}
            continue
        try:
            rows = fetch_yahoo(sym)
            prior = obs[name][-1]["date"] if obs[name] else None
            if rows and (prior is None or rows[-1]["date"] >= prior):
                obs[name] = merge_observations(obs[name], rows)
                report[name] = {"status": "UPDATED", "last": obs[name][-1]["date"]}
            else:
                report[name] = {"status": "KEPT CACHE", "last": prior}
        except Exception as exc:
            report[name] = {"status": "KEPT CACHE", "reason": type(exc).__name__}
        time.sleep(0.12)
    for k, v in obs.items():
        if k in prov and isinstance(prov[k], dict) and v:
            prov[k]["n"] = len(v)
            prov[k]["last"] = v[-1]["date"]
            prov[k]["first"] = v[0]["date"]
    prov.setdefault("_meta", {})["last_refresh"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    obs_f.write_text(json.dumps(obs, separators=(",", ":")), encoding="utf-8")
    prov_f.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- entry_observations.json (v5 daily macro layer) ---
    eobs_f, eprov_f = MACRO / "entry_observations.json", MACRO / "entry_provenance.json"
    eobs = json.loads(eobs_f.read_text(encoding="utf-8"))
    eprov = json.loads(eprov_f.read_text(encoding="utf-8"))
    for sid in ED.FRED_SERIES:
        try:
            rows = ED.fetch_fred(sid)
            before = len(eobs.get(sid, []))
            if rows:
                eobs[sid] = merge_observations(eobs.get(sid, []), rows)
                report[sid] = {"status": "UPDATED", "added": len(eobs[sid]) - before,
                               "last": eobs[sid][-1]["date"]}
            else:
                report[sid] = {"status": "KEPT CACHE", "last": eobs[sid][-1]["date"]}
        except Exception as exc:
            report[sid] = {"status": "KEPT CACHE", "reason": type(exc).__name__}
        time.sleep(0.2)
    try:
        fng = ED.fetch_cnn_fng()
        if fng:
            eobs["FNG"] = merge_observations(eobs.get("FNG", []), fng)
            report["FNG"] = {"status": "UPDATED", "last": eobs["FNG"][-1]["date"]}
        else:
            report["FNG"] = {"status": "KEPT CACHE", "last": eobs["FNG"][-1]["date"]}
    except Exception as exc:
        report["FNG"] = {"status": "KEPT CACHE", "reason": type(exc).__name__}
    for k, v in eobs.items():
        if k in eprov and isinstance(eprov[k], dict) and v:
            eprov[k]["n"] = len(v)
            eprov[k]["last"] = v[-1]["date"]
            eprov[k]["first"] = v[0]["date"]
    eprov.setdefault("_meta", {})["last_refresh"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    eobs_f.write_text(json.dumps(eobs, separators=(",", ":")), encoding="utf-8")
    eprov_f.write_text(json.dumps(eprov, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def refresh_all(offline: bool = False) -> dict:
    OPS.mkdir(parents=True, exist_ok=True)
    if offline:
        return {"status": "OFFLINE (cached data used)", "prices": {}, "macro": {},
                "sources_updated": 0, "sources_failed": 0}
    prices = refresh_prices()
    macro = refresh_macro()
    allsrc = {**prices, **macro}
    updated = sum(1 for v in allsrc.values()
                  if v.get("status") in ("UPDATED", "UPDATED VIA PRICE FILE"))
    failed = len(allsrc) - updated
    report = {"status": "OK" if updated else "ALL SOURCES KEPT CACHE",
              "checked_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "prices": prices, "macro": macro,
              "sources_updated": updated, "sources_failed": failed,
              "last_dates": OP.last_dates()}
    log_f = OPS / "refresh_log.json"
    log = json.loads(log_f.read_text(encoding="utf-8")) if log_f.exists() else []
    log.append({"checked_at": report["checked_at"], "status": report["status"],
                "sources_updated": updated, "sources_failed": failed})
    log_f.write_text(json.dumps(log[-90:], indent=1), encoding="utf-8")
    (OPS / "last_refresh.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    import sys
    r = refresh_all(offline="--offline" in sys.argv)
    print(f"[refresh] {r['status']} | updated {r['sources_updated']} | kept cache {r['sources_failed']}")
