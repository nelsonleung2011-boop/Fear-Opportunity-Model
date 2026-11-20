#!/usr/bin/env python3
"""Historical score series and percentile ranking for v7 operations.

Two distinct things, deliberately kept separate:

* BACKFILL  — month-end VOO/QQQ Buying Scores recomputed over the full history
  using the SAME pre-registered formula as today. Every input is point-in-time:
  expanding percentiles and k-NN risk probabilities use only data known at each
  historical date. This is the reference distribution for percentile ranking.
* LIVE      — one row per daily run, appended after the real daily refresh.

The backfill is cached and recomputed only when the scoring date advances or
when explicitly requested, so a daily run stays cheap.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import entry_decision_engine as V7
from pipeline import entry_opportunity_engine as V6
from pipeline import entry_quality_engine as V5

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "data" / "ops"
BACKFILL = OPS / "score_history_backfill.csv"
LIVE = OPS / "score_history.csv"
META = OPS / "history_meta.json"

COLS = ["date", "source", "voo", "qqq", "valuation", "opportunity",
        "risk_meter", "detector", "regime", "deployment_pct", "pct_voo", "pct_qqq"]


def _bonus(dd: float, table: list) -> float:
    return float(np.interp(dd * 100.0, [x[0] for x in table], [x[1] for x in table]))


def build_backfill(frames: tuple, as_of: pd.Timestamp) -> pd.DataFrame:
    """Recompute month-end scores since 2000 with point-in-time inputs only."""
    adj, raw_px = frames
    pit, _ = V5.build_macro_pit(raw_px.index)
    daily = V5.build_daily_components(raw_px, pit)
    dates = [d for d in pd.date_range("2000-01-31", as_of, freq="ME") if d in daily.index]

    mc_pct = V5.expanding_pct(daily["mcclellan_neg"], daily.index, 252)
    credit_pct = V5.expanding_pct(daily["credit_stress_price"], daily.index, 252)
    fg_pct = V5.expanding_pct(daily["fear_greed_constructed"], daily.index, 252)
    fng_index = 100.0 - fg_pct * 100.0

    probs = V5.risk_probabilities(daily, dates, adj)
    risks = V5.risk_subsignals(daily, dates, probs)

    rows = []
    for d in dates:
        val = V6.valuation_components(daily, d)["score"]
        opp = V6.fear_components(daily, d, mc_pct)["score"]
        meter = str(risks.loc[d, "meter"]) if d in risks.index else "Moderate"
        risk_score = V7.RISK_OPPORTUNITY.get(meter, 70.0)
        dd_v = V6.dd_depth(raw_px["SPY"], d)
        dd_q = V6.dd_depth(raw_px["QQQ"], d)
        sv, _, _ = V6.buy_score(val, opp, risk_score, _bonus(dd_v, V6.VOO_DD_BONUS))
        sq, _, _ = V6.buy_score(val, opp, risk_score, _bonus(dd_q, V6.QQQ_DD_BONUS))
        det = V6.detector(daily, d, dd_v, float(credit_pct.get(d, 0.5)),
                          float(fng_index.get(d, 50.0)))["state"]
        rows.append({"date": str(d.date()), "source": "backfill",
                     "voo": round(sv, 2), "qqq": round(sq, 2),
                     "valuation": round(val, 2), "opportunity": round(opp, 2),
                     "risk_meter": meter, "detector": det, "regime": "",
                     "deployment_pct": V7.deployment_for(sv)["percent"],
                     "pct_voo": None, "pct_qqq": None})
    return pd.DataFrame(rows, columns=COLS)


def percentile_rank(score: float, series) -> float:
    """Percent of historical entry dates at or below today's score."""
    a = np.asarray([x for x in series if x is not None and np.isfinite(x)], dtype=float)
    if not len(a):
        return 50.0
    return float(round(100.0 * np.mean(a <= float(score)), 1))


def load_backfill(frames: tuple, as_of: pd.Timestamp, rebuild: bool = False) -> pd.DataFrame:
    OPS.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    need = (rebuild or not BACKFILL.exists()
            or meta.get("as_of") != str(as_of.date())
            or meta.get("rows", 0) < 100)
    if need:
        df = build_backfill(frames, as_of)
        df.to_csv(BACKFILL, index=False)
        META.write_text(json.dumps({
            "as_of": str(as_of.date()), "rows": len(df),
            "built_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "method": "month-end, point-in-time, same formula as the live score",
        }, indent=1), encoding="utf-8")
        return df
    return pd.read_csv(BACKFILL)


def append_live(payload: dict, pct_voo: float, pct_qqq: float) -> pd.DataFrame:
    """Upsert today's real observation (re-running the same day never duplicates)."""
    OPS.mkdir(parents=True, exist_ok=True)
    v, q = payload["instruments"]["VOO"], payload["instruments"]["QQQ"]
    row = {"date": payload["meta"]["as_of"], "source": "live",
           "voo": v["score"], "qqq": q["score"],
           "valuation": payload["valuation"]["score"],
           "opportunity": payload["opportunity"]["score"],
           "risk_meter": payload["risk"]["meter"],
           "detector": v["detector"]["state"],
           "regime": payload["hero"]["market_regime"],
           "deployment_pct": v["deployment"]["percent"],
           "pct_voo": pct_voo, "pct_qqq": pct_qqq}
    df = pd.read_csv(LIVE) if LIVE.exists() else pd.DataFrame(columns=COLS)
    df = df[df["date"] != row["date"]]
    df = pd.concat([df, pd.DataFrame([row], columns=COLS)], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(LIVE, index=False)
    return df


def combined_history() -> pd.DataFrame:
    """Backfill + live observations for display in the Operations tab."""
    parts = []
    for f, src in ((BACKFILL, "backfill"), (LIVE, "live")):
        if f.exists():
            d = pd.read_csv(f)
            if len(d):
                parts.append(d)
    if not parts:
        return pd.DataFrame(columns=COLS)
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return out
