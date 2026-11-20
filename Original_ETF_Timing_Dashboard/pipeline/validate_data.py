#!/usr/bin/env python3
"""
Data quality validation (Epic 2, Task 2.1 & 2.2).

For every symbol, check: missing dates, duplicate dates, null/infinite/NaN
values, insufficient length. Emits an audit record per symbol:

    {"symbol":"SPY","source":"Yahoo","last_updated":"...",
     "data_points":7011,"data_quality":"PASS"}

and a min-length gate: SPY>252, QQQ>252, TLT>126 trading days.
"""
import os
import json
import math
import datetime as dt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
DATA_DIR = os.path.join(ROOT, "data")

MIN_LEN = {"SPY": 252, "QQQ": 252, "TLT": 126}

CORE = ["SPY", "QQQ", "VIX", "TLT", "TNX", "IRX"]

# Price series must be strictly positive. Rate / index series may be zero or
# negative (T-bill yields went below zero in 2020-2021) and are exempt.
RATE_OR_INDEX = {"VIX", "TNX", "IRX"}


def validate_symbol(key, prov):
    path = os.path.join(RAW_DIR, key + ".csv")
    rec = {
        "symbol": key,
        "source": prov.get("provider", "unknown") if prov.get("ok") else "unavailable",
        "last_updated": prov.get("last", "") if prov.get("ok") else "",
        "fetched_at": prov.get("fetched_at", ""),
        "data_points": 0,
        "data_quality": "FAIL",
        "checks": {},
        "issues": [],
    }
    if not prov.get("ok") or not os.path.exists(path):
        rec["issues"].append("fetch failed: " + "; ".join(prov.get("errors", [])))
        return rec
    df = pd.read_csv(path)
    if "date" not in df or "close" not in df:
        rec["issues"].append("missing date/close columns")
        return rec
    n = len(df)
    rec["data_points"] = n
    dates = pd.to_datetime(df["date"], errors="coerce")
    # Validate the series the model actually consumes: dividend-adjusted close
    # when the provider supplied it, otherwise raw close.
    basis = prov.get("price_basis", "close")
    col = "adj_close" if ("adj_close" in df.columns and basis == "adjusted") else "close"
    rec["price_basis"] = basis
    rec["series_column"] = col
    close = pd.to_numeric(df[col], errors="coerce")
    if col == "adj_close" and close.isna().all():
        close = pd.to_numeric(df["close"], errors="coerce")
        rec["price_basis"] = basis = "close"
        rec["series_column"] = col = "close"

    dup = int(dates.duplicated().sum())
    miss_date = int(dates.isna().sum())
    null_v = int(close.isna().sum())
    inf_v = int((~close.isna() & ~close.map(lambda x: math.isfinite(x) if pd.notna(x) else False)).sum())
    if key in RATE_OR_INDEX:
        nonpos = 0  # not a meaningful check for yields/indices
    else:
        nonpos = int((close <= 0).sum())
    # gaps > 10 calendar days (excluding reasonable holiday stretches)
    s = dates.dropna().sort_values()
    gaps = int((s.diff().dt.days > 10).sum())

    # staleness: how old is the newest observation relative to the run date
    stale_days = None
    if len(s):
        stale_days = int((pd.Timestamp.utcnow().tz_localize(None).normalize()
                          - s.iloc[-1].normalize()).days)

    rec["checks"] = {
        "duplicate_dates": dup, "missing_dates": miss_date, "null_values": null_v,
        "infinite_values": inf_v, "nonpositive_prices": nonpos, "large_gaps": gaps,
    }
    rec["stale_days"] = stale_days
    for k, v in rec["checks"].items():
        if v > 0:
            rec["issues"].append("%s=%d" % (k, v))

    length_ok = True
    if key in MIN_LEN:
        need = MIN_LEN[key]
        if n <= need:
            rec["issues"].append("length %d <= required %d" % (n, need))
            length_ok = False
    if n < 30:
        rec["issues"].append("length %d < 30 hard floor" % n)
        length_ok = False

    hard_fail = (null_v + inf_v + dup + miss_date + nonpos) > 0 or not length_ok
    warn = gaps > 0 or (n < 252) or (stale_days is not None and stale_days > 7)
    if stale_days is not None and stale_days > 7:
        rec["issues"].append("stale: newest bar is %d days old" % stale_days)
    if hard_fail:
        rec["data_quality"] = "FAIL"
    elif warn:
        rec["data_quality"] = "WARNING"
    else:
        rec["data_quality"] = "PASS"
    return rec


def main():
    prov = json.load(open(os.path.join(RAW_DIR, "provenance.json")))
    audit = {k: validate_symbol(k, prov.get(k, {})) for k in sorted(prov)}
    core_fail = [k for k in CORE if audit.get(k, {}).get("data_quality") == "FAIL"]
    summary = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "symbols": len(audit),
        "pass": sum(1 for v in audit.values() if v["data_quality"] == "PASS"),
        "warning": sum(1 for v in audit.values() if v["data_quality"] == "WARNING"),
        "fail": sum(1 for v in audit.values() if v["data_quality"] == "FAIL"),
        "core_failures": core_fail,
        "gate": "PASS" if not core_fail else "FAIL",
    }
    out = {"summary": summary, "symbols": audit}
    with open(os.path.join(DATA_DIR, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Audit gate:", summary["gate"], "| PASS", summary["pass"],
          "WARN", summary["warning"], "FAIL", summary["fail"])
    if core_fail:
        print("Core failures:", core_fail)
    for k in ["SPY", "QQQ", "VIX", "TLT", "XLRE", "XLC"]:
        v = audit[k]
        print("  %-5s %-6s pts=%-5d %s" % (k, v["data_quality"], v["data_points"], v["checks"]))
    return 0 if not core_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
