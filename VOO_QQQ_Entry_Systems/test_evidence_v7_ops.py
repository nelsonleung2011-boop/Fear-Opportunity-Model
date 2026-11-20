#!/usr/bin/env python3
"""Tests for v7 operational layer: refresh safety, dynamic as-of, percentile
ranking, alert dedupe, email content, data provenance and payload validity."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline import ops_alerts as AL           # noqa: E402
from pipeline import ops_data_audit as AUDIT     # noqa: E402
from pipeline import ops_history as HIST         # noqa: E402
from pipeline import ops_price as PRICE          # noqa: E402
from pipeline import ops_refresh as REFRESH      # noqa: E402

CHECKS = 0
FAILS = 0


def ck(cond, msg):
    global CHECKS, FAILS
    CHECKS += 1
    if not cond:
        FAILS += 1
        print("  FAIL:", msg)


PAYLOAD = json.loads((ROOT / "data" / "evidence_v7.json").read_text(encoding="utf-8"))
NOW = dt.datetime(2026, 11, 19, 9, 0, tzinfo=dt.timezone.utc)


def payload(**over):
    """Deep-ish copy with overrides for the fields the alert engine reads."""
    p = json.loads(json.dumps(PAYLOAD))
    for k, v in over.items():
        if k == "voo":
            p["instruments"]["VOO"]["score"] = v
        elif k == "qqq":
            p["instruments"]["QQQ"]["score"] = v
        elif k == "regime":
            p["hero"]["market_regime"] = v
        elif k == "detector":
            p["instruments"]["VOO"]["detector"]["state"] = v
        elif k == "date":
            p["meta"]["as_of"] = v
    return p


def fresh_state():
    return {"last": {}, "last_fired": {}, "last_weekly": None,
            "consecutive_yes": 0, "runs": 0}


# ---------------------------------------------------------------------------
# 1. Dynamic price frames — the hard-coded as-of bug
# ---------------------------------------------------------------------------
def test_dynamic_frames():
    idx = pd.date_range("2020-01-01", "2027-03-05", freq="B")
    d = {"SPY": pd.Series(np.arange(len(idx), dtype=float) + 100, index=idx)}
    f = PRICE.build_frame(d)
    ck(f.index.max() > pd.Timestamp("2026-11-16"),
       "build_frame does not hard-code the 2026-11-16 cutoff")
    f2 = PRICE.build_frame(d, pd.Timestamp("2024-06-28"))
    ck(f2.index.max() == pd.Timestamp("2024-06-28"), "explicit cutoff honoured")

    adj, raw = PRICE.load_frames(as_of="2026-11-10")
    ck(raw.index.max() == pd.Timestamp("2026-11-10"), "load_frames honours as-of")
    ck(adj.index.max() == raw.index.max(), "adjusted and raw frames stay aligned")
    ck("SPY" in raw and "VIX9D" in raw and "XLK" in raw, "all required symbols loaded")
    ck(PRICE.resolve_as_of(raw, "2025-01-31") == pd.Timestamp("2025-01-31"),
       "explicit as-of wins over data-derived anchor")

    snap = PRICE.price_snapshot(8)
    ck(len(snap["dates"]) == 8, "price snapshot has 8 rows")
    ck(set(snap["series"]) == {"SPY", "QQQ", "VIX", "HYG", "LQD", "RSP", "IWM"},
       "price snapshot covers core series")
    ck(all(len(v) == 8 for v in snap["series"].values()), "snapshot columns aligned")


# ---------------------------------------------------------------------------
# 2. Refresh safety
# ---------------------------------------------------------------------------
def test_refresh_safety():
    old = [{"date": "2020-01-02", "value": 1.0}, {"date": "2020-01-03", "value": 2.0}]
    new = [{"date": "2020-01-03", "value": 2.5}, {"date": "2020-01-06", "value": 3.0}]
    m = REFRESH.merge_observations(old, new)
    ck(len(m) == 3, "observation merge keeps union of dates")
    ck(m[1]["value"] == 2.5, "refreshed value replaces revised value")
    ck([r["date"] for r in m] == sorted(r["date"] for r in m), "merged rows sorted")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "X.csv"
        pd.DataFrame({"date": ["2026-01-02", "2026-01-05"],
                      "close": [10.0, 11.0], "adj_close": [9.0, 10.0]}).to_csv(p, index=False)
        tiny = [{"date": "2026-01-02", "close": 10.0, "adj_close": 9.0}]
        ck(REFRESH.merge_csv(p, tiny)["status"] == "KEPT CACHE", "short history rejected")
        ck(len(pd.read_csv(p)) == 2, "rejected fetch leaves cached file intact")

        rows = [{"date": f"{y}-{m:02d}-{d:02d}", "close": 10.0 + m, "adj_close": 9.0 + m}
                for y in range(2020, 2023) for m in range(1, 13) for d in (1, 8, 15, 22)]
        rows += [{"date": "2026-01-06", "close": 15.0, "adj_close": 14.0}]
        ck(len(rows) > 100, "valid fixture exceeds the minimum-history guard")
        res = REFRESH.merge_csv(p, rows)
        ck(res["status"] == "UPDATED", "valid longer history accepted")
        ck(pd.read_csv(p)["date"].iloc[-1] == "2026-01-06", "new last row written")
        ck(list(pd.read_csv(p).columns) == ["date", "close", "adj_close"],
           "csv schema preserved")

        backwards = [{"date": f"2020-{m:02d}-01", "close": 10.0, "adj_close": 9.0}
                     for m in range(1, 13)]
        ck(REFRESH.merge_csv(p, backwards)["status"] == "KEPT CACHE",
           "backwards last date rejected")


# ---------------------------------------------------------------------------
# 3. Percentile ranking
# ---------------------------------------------------------------------------
def test_percentiles():
    s = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ck(HIST.percentile_rank(50, s) == 50.0, "median score = 50th percentile")
    ck(HIST.percentile_rank(100, s) == 100.0, "maximum = 100th percentile")
    ck(HIST.percentile_rank(0, s) == 0.0, "below minimum = 0th percentile")
    ck(HIST.percentile_rank(35, s) == 30.0, "ties counted as <= current")
    ck(HIST.percentile_rank(42, pd.Series([np.nan, None])) == 50.0,
       "empty/NaN history returns neutral 50")

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "score_history.csv"
        old = HIST.LIVE
        HIST.LIVE = f
        try:
            HIST.append_live(payload(), 12.6, 25.7)
            HIST.append_live(payload(), 12.6, 25.7)
            d = pd.read_csv(f)
            ck(len(d) == 1, "same-day rerun does not duplicate history")
            ck(float(d["voo"].iloc[0]) == PAYLOAD["instruments"]["VOO"]["score"],
               "live row stores the real score")
            p2 = payload(date="2026-11-17", voo=55.0)
            HIST.append_live(p2, 60.0, 60.0)
            ck(len(pd.read_csv(f)) == 2, "a new date appends one row")
        finally:
            HIST.LIVE = old


# ---------------------------------------------------------------------------
# 4. Alert rules — no spam, four triggers only
# ---------------------------------------------------------------------------
def test_alert_rules():
    st = fresh_state()
    a, st = AL.evaluate(payload(), st, now=NOW)
    ck(len(a) == 1 and a[0]["type"] == "weekly_summary",
       "first run sends only the weekly summary (no false threshold alert)")
    ck(st["last_weekly"] == "2026-W47", "ISO week recorded")

    hi = payload(voo=85.0, qqq=72.0)
    a, st = AL.evaluate(hi, st, now=NOW)
    types = [x["type"] for x in a]
    ck("threshold_crossed" in types, "VOO crossing 80 fires an alert")
    ck("weekly_summary" not in types, "weekly summary not repeated in the same week")
    ck(any(f"VOO {PAYLOAD['instruments']['VOO']['score']:.1f} -> 85.0" in x["detail"]
           for x in a), "crossing detail is explicit")

    a, st = AL.evaluate(hi, st, now=NOW)
    ck(not any(x["type"] == "threshold_crossed" for x in a),
       "score staying above 80 does NOT resend (no daily spam)")
    ck(all(x["type"] != "weekly_summary" for x in a), "no duplicate weekly summary")

    a, st = AL.evaluate(payload(voo=85.0, qqq=82.0), st, now=NOW)
    ck(any("QQQ" in x["detail"] for x in a if x["type"] == "threshold_crossed"),
       "QQQ crossing 80 fires separately")

    a, st = AL.evaluate(payload(voo=85.0, qqq=82.0, regime="PANIC"), st, now=NOW)
    pan = [x for x in a if x["type"] == "regime_change"]
    ck(len(pan) == 1 and pan[0]["priority"] == "urgent", "regime change to PANIC is urgent")

    a, st = AL.evaluate(payload(voo=85.0, qqq=82.0, regime="PANIC", detector="YES"), st, now=NOW)
    ck(any(x["type"] == "detector_triggered" and x["priority"] == "urgent" for x in a),
       "detector transition into YES fires an urgent alert")
    a, st = AL.evaluate(payload(voo=85.0, qqq=82.0, regime="PANIC", detector="YES"), st, now=NOW)
    ck(not any(x["type"] == "detector_triggered" for x in a),
       "detector staying YES does not resend")
    ck(st["consecutive_yes"] >= 2, "consecutive YES days tracked")

    a, st = AL.evaluate(payload(voo=85.0, qqq=82.0, regime="PANIC", detector="YES"),
                        st, now=NOW)
    ck(len(a) == 0, "an unchanged day sends nothing at all")

    a, _ = AL.evaluate(payload(), st, now=dt.datetime(2026, 11, 23, 9, tzinfo=dt.timezone.utc))
    ck(any(x["type"] == "weekly_summary" for x in a), "new ISO week produces one summary")

    rv = AL.rules_view(payload(), st)
    ck(len(rv) == 5, "five rule rows shown in the Operations tab")
    ck(any(r["name"] == "VOO >= 80" for r in rv), "VOO threshold rule listed")
    ck(any("Weekly summary" in r["name"] for r in rv), "weekly summary rule listed")


# ---------------------------------------------------------------------------
# 5. Email content + delivery
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = ["Date", "Current VOO Score", "Current QQQ Score",
                   "Historical Percentile", "Top Positive Contributors",
                   "Top Negative Contributors", "Current Regime",
                   "Historical Analogues", "Deployment Recommendation"]


def test_email():
    alert = {"type": "threshold_crossed", "priority": "high",
             "detail": "VOO 79 -> 85 crossed 80", "time": "2026-11-19T09:00:00Z"}
    em = AL.render_email(alert, payload(), {"VOO": 12.6, "QQQ": 25.7})
    ck(set(em["fields"]) == set(REQUIRED_FIELDS), "email has exactly the 9 required fields")
    for f in REQUIRED_FIELDS:
        ck(f in em["html"] and f in em["text"], "email body contains: " + f)
    ck("ETF ENTRY DECISION SYSTEM" in em["html"], "email has branded header")
    ck("80" in em["subject"] or "VOO" in em["subject"], "subject identifies the event")
    ck("never on an unchanged day" in em["html"], "no-spam policy stated in email")

    with tempfile.TemporaryDirectory() as td:
        old = AL.OUTBOX
        AL.OUTBOX = Path(td) / "outbox"
        try:
            with patch.dict(os.environ, {}, clear=True):
                cfg = AL.smtp_config()
                ck(cfg["configured"] is False, "no SMTP env -> not configured")
            with patch.dict(os.environ, {}, clear=True):
                res = AL.deliver(em, dry_run=False)
            ck(res["delivery"] == "dry-run", "missing credentials -> dry-run, not an exception")
            ck(Path(res["outbox"]).exists(), "dry-run email archived as HTML")
            ck(Path(res["outbox"]).with_suffix(".txt").exists(), "plain-text copy archived")
            ck(Path(res["outbox"]).with_suffix(".json").exists(), "machine-readable copy archived")
            res2 = AL.deliver(em, dry_run=True)
            ck(res2["delivery"] == "dry-run" and "explicit dry-run" in res2["reason"],
               "explicit dry-run overrides any configuration")
        finally:
            AL.OUTBOX = old

    with patch.dict(os.environ, {"SMTP_HOST": "smtp.example.com", "SMTP_USER": "u",
                                "SMTP_PASS": "p", "ALERT_TO": "a@b.com,c@d.com"}, clear=True):
        cfg = AL.smtp_config()
        ck(cfg["configured"] is True, "full env -> configured")
        ck(cfg["to"] == ["a@b.com", "c@d.com"], "multiple recipients parsed")


# ---------------------------------------------------------------------------
# 6. Data provenance ledger
# ---------------------------------------------------------------------------
def test_data_audit():
    a = AUDIT.build(PAYLOAD)
    ck(len(a["sources"]) >= 20, "source inventory is complete")
    ck(len(a["score_inputs"]) == len(PAYLOAD["instruments"]["VOO"]["explain"]["components"]) + 1,
       "one ledger row per scored contributor plus the drawdown bonus")
    ck(a["summary"]["n_fabricated"] == 0, "no fabricated series")
    ck(a["summary"]["n_proxies"] > 0, "proxies are counted, not hidden")
    ck(any(v["level"] == "CAVEAT" for v in a["verdict"]), "caveats disclosed")
    ck(any("LATEST REVISED" in v["text"] for v in a["verdict"]), "revised-vintage caveat")
    ck(any("hard-coded 2026-11-16" in v["text"] for v in a["verdict"]), "frozen-date fix")
    ck(a["checks"]["real_time_vintages"] is False,
       "system does not falsely claim real-time vintages")
    for s in a["sources"]:
        ck(s["staleness_days"] >= 0, "non-negative staleness: " + s["name"])
        ck(bool(s["source"]) and bool(s["role"]), "source and role populated: " + s["name"])
    fut = [s for s in a["sources"] if "after as-of" in s["vintage"]]
    ck(all("excluded" in s["vintage"] for s in fut),
       "any row dated after the scoring date is flagged as excluded")
    ck(all(s["staleness_days"] >= 0 for s in a["sources"]),
       "no source reports a negative age after a live refresh")

    base = PAYLOAD["instruments"]["VOO"]["base_score"]
    parts = PAYLOAD["instruments"]["VOO"]["explain"]["components"]
    ck(abs(sum(p["contribution"] for p in parts) - base) < 0.15,
       "explained contributions reproduce the base score")
    ck(abs(sum(r["contribution"] for r in a["score_inputs"][:-1]) - base) < 0.2,
       "data ledger contributions match the audit trail")


# ---------------------------------------------------------------------------
# 7. Built payload is valid JSON and carries both new tabs
# ---------------------------------------------------------------------------
def test_built_payload():
    html = (ROOT / "index_v7.html").read_text(encoding="utf-8")
    m = re.search(r'<script id="payload" type="application/json">([\s\S]*?)</script>', html)
    ck(m is not None, "payload script present")
    d = json.loads(m.group(1))
    ck("ops" in d and "data_audit" in d, "ops + data audit embedded")
    ck("NaN" not in m.group(1) and "Infinity" not in m.group(1),
       "payload contains no bare NaN/Infinity (browser-parseable)")
    ck(len(d["ops"]["score_history"]) > 0, "score history embedded")
    ck("percentiles" in d["ops"] and "VOO" in d["ops"]["percentiles"], "percentiles embedded")
    ck("id=\"tab-' + t[0]" in html, "tab buttons receive real ids")
    ck("addEventListener('click', handleClick)" in html, "delegated click handler present")
    ck('id="pane-data"' in html and 'id="pane-ops"' in html, "new pane containers exist")


def main():
    test_dynamic_frames()
    test_refresh_safety()
    test_percentiles()
    test_alert_rules()
    test_email()
    test_data_audit()
    test_built_payload()
    print(f"v7 operational: {CHECKS} checks, {FAILS} failures")
    if FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
