#!/usr/bin/env python3
"""
ETF Entry Quality System v6 -- Entry OPPORTUNITY Engine
=======================================================

This is deliberately NOT v5 with different weights. It answers one question:

    "If I had excess cash today, how aggressively should I deploy it?"

It does not forecast returns. It does not optimise for AUC, calibration or
statistical elegance. It is built to recognise panic-driven, rare accumulation
windows: a VIX of 45 during a 25% drawdown matters far more than a one-point
difference in CAPE.

WEIGHTS (pre-registered)
  Layer 1 Valuation          30%   five valuation inputs
  Layer 2 Fear / Opportunity 50%   six inputs with FIXED weights:
        VIX 25%, drawdown 25%, % above 200DMA 20%, distance below 200DMA 15%,
        put/call proxy 10%, breadth/McClellan proxy 5%
  Layer 3 Risk               20%   risk moderates the opportunity score

NONLINEARITY
  Fixed knots reward extreme fear disproportionately. VIX uses ABSOLUTE levels
  (not percentiles): below 20 contributes little, 30-40 strong, 40-50 very
  strong, above 50 maximum. Drawdown bonuses are additive on top.

RARE OPPORTUNITY DETECTOR
  Counts five simultaneously observable conditions. >=3 = YES, 2 = WATCHLIST,
  otherwise NO. Validated against 2008 GFC, March 2020 and 2022 periods.

No v1-v5 file is modified. Inputs reuse v5's PIT macro alignment and v4's
price frames. Output: data/evidence_v6.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings = __import__("warnings")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import evidence_engine_v4 as V4          # noqa: E402
from pipeline import entry_quality_engine as V5        # noqa: E402

OUT = ROOT / "data" / "evidence_v6.json"
SEED = 20261119

W_VAL, W_FEAR, W_RISK = 0.30, 0.50, 0.20
FEAR_WEIGHTS = {"vix": 0.25, "drawdown": 0.25, "above_200dma": 0.20,
                "below_200dma": 0.15, "putcall": 0.10, "mcclellan": 0.05}

# Absolute, pre-registered knots. x ascending, y = 0..100.
VIX_KNOTS = [(0, 0), (15, 5), (20, 15), (25, 30), (30, 55), (35, 75),
             (40, 88), (45, 95), (50, 100), (90, 100)]
DD_KNOTS = [(0, 0), (0.05, 10), (0.10, 25), (0.15, 45), (0.20, 65),
            (0.25, 78), (0.30, 88), (0.40, 100)]
ABOVE_KNOTS = [(0, 100), (20, 85), (40, 65), (60, 40), (80, 20), (100, 0)]
BELOW_KNOTS = [(-0.15, 0), (-0.10, 10), (-0.05, 25), (0, 45), (0.05, 65),
               (0.10, 82), (0.25, 100)]
PUTCALL_KNOTS = [(-0.25, 0), (-0.15, 10), (-0.05, 30), (0, 55), (0.05, 75),
                 (0.15, 90), (0.35, 100)]
# Additive drawdown bonuses, in percentage points
VOO_DD_BONUS = [(0, 0), (5, 0), (10, 5), (15, 10), (20, 20), (30, 35), (40, 50)]
QQQ_DD_BONUS = [(0, 0), (10, 5), (20, 15), (30, 30), (40, 50)]

BANDS = [(90, "Rare Opportunity", "Aggressive deployment. Historically rare entry conditions."),
         (80, "Strong Entry", "Increase purchases by 50%."),
         (60, "Attractive", "Increase purchases by 25%."),
         (40, "Neutral", "Normal DCA."),
         (20, "Expensive", "Normal DCA only."),
         (0, "Extremely Unattractive", "Pause new lump-sum buying. Continue DCA.")]

# Risk meter -> opportunity subscore (high risk lowers the composite modestly)
RISK_OPPORTUNITY = {"Very Low": 100.0, "Low": 90.0, "Moderate": 70.0,
                    "High": 45.0, "Extreme": 25.0}

# Detector thresholds (pre-registered; no tuning on outcomes)
DET_VIX, DET_DD, DET_ABOVE = 35.0, 0.20, 20.0
DET_CREDIT_PCT, DET_FNG = 0.80, 25.0

ANALOGUES = [
    ("2008 GFC", "2008-11-20"),
    ("2011 Euro Crisis", "2011-10-03"),
    ("2018 Q4 Selloff", "2018-12-24"),
    ("2020 COVID", "2020-03-23"),
    ("2022 Bear Market", "2022-10-12"),
]
# Fixed normalisation ranges for similarity (pre-registered, auditable)
SIM_RANGE = {"vix": (0, 80), "dd": (0, 0.55), "above": (0, 100),
             "below": (-0.15, 0.25), "valuation": (0, 100), "credit": (0, 100)}

HIST_EXAMPLES = ["2008-10-27", "2008-11-20", "2009-03-09", "2011-10-03",
                 "2018-12-24", "2020-03-23", "2022-06-16", "2022-10-12"]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def knot_score(x, knots: list[tuple[float, float]], neutral: float = 50.0) -> float:
    """Piecewise-linear interpolation through pre-registered knots."""
    if x is None or not np.isfinite(x):
        return neutral
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    return float(np.clip(np.interp(float(x), xs, ys), 0.0, 100.0))


def band_for(score: float) -> tuple[str, str]:
    for thr, label, guidance in BANDS:
        if score >= thr:
            return label, guidance
    return BANDS[-1][1], BANDS[-1][2]


def dd_depth(px: pd.Series, at: pd.Timestamp) -> float:
    hist = px.loc[:at]
    if hist.empty:
        return 0.0
    return float(max(0.0, 1.0 - hist.iloc[-1] / hist.cummax().iloc[-1]))


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def fear_components(daily: pd.DataFrame, at: pd.Timestamp, mc_pct: pd.Series) -> dict:
    row = daily.loc[at]
    above_pct = 100.0 * (1.0 - float(row["pct_above_200dma_neg"]))
    pc_raw = row["putcall_proxy"]
    mc = 100.0 * float(mc_pct.get(at, 0.5)) ** 2
    comps = {
        "vix": {"raw": float(row["vix_level"]),
                "score": knot_score(row["vix_level"], VIX_KNOTS),
                "weight": FEAR_WEIGHTS["vix"],
                "why": "Absolute VIX level. Below 20 adds little; above 50 receives maximum credit."},
        "drawdown": {"raw": float(row["dd_depth"]),
                     "score": knot_score(row["dd_depth"], DD_KNOTS),
                     "weight": FEAR_WEIGHTS["drawdown"],
                     "why": "Depth below the all-time high, scored with an accelerating curve."},
        "above_200dma": {"raw": above_pct,
                         "score": knot_score(above_pct, ABOVE_KNOTS),
                         "weight": FEAR_WEIGHTS["above_200dma"],
                         "why": "Few stocks above their 200DMA marks broad capitulation."},
        "below_200dma": {"raw": float(row["below_200dma"]),
                         "score": knot_score(row["below_200dma"], BELOW_KNOTS),
                         "weight": FEAR_WEIGHTS["below_200dma"],
                         "why": "Price location versus trend; deeper below trend scores higher."},
        "putcall": {"raw": (None if not np.isfinite(pc_raw) else float(pc_raw)),
                    "score": knot_score(pc_raw, PUTCALL_KNOTS),
                    "weight": FEAR_WEIGHTS["putcall"],
                    "why": "VIX9D/VIX term structure as a put-demand proxy; 50 = neutral when unavailable."},
        "mcclellan": {"raw": float(row["mcclellan_neg"]),
                      "score": mc,
                      "weight": FEAR_WEIGHTS["mcclellan"],
                      "why": "Breadth impulse percentile, squared so only extremes contribute."},
    }
    total = sum(c["score"] * c["weight"] for c in comps.values())
    return {"components": comps, "score": float(total)}


def valuation_components(daily: pd.DataFrame, at: pd.Timestamp) -> dict:
    specs = [
        ("cape", "Shiller CAPE", -1, 36),
        ("buffett", "Buffett Indicator", -1, 36),
        ("earnings_yield", "Earnings Yield", +1, 36),
        ("equity_risk_premium", "Equity Risk Premium", +1, 36),
        ("real_yield", "Real Yield (10Y TIPS)", -1, 36),
    ]
    out = {}
    for key, name, direction, minp in specs:
        p = float(V5.expanding_pct(daily[key], [at], minp).iloc[0])
        if direction < 0:
            p = 1.0 - p
        # mild nonlinearity: extremes count more than small differences
        out[key] = {"raw": (None if not np.isfinite(daily[key].loc[at])
                            else round(float(daily[key].loc[at]), 4)),
                    "pct": round(p * 100, 1),
                    "score": round(100.0 * p ** 1.5, 1),
                    "name": name, "direction": direction}
    score = float(np.mean([c["score"] for c in out.values()]))
    return {"components": out, "score": score}


def detector(daily: pd.DataFrame, at: pd.Timestamp, dd: float,
             credit_pct: float, fng: float) -> dict:
    row = daily.loc[at]
    above = 100.0 * (1.0 - float(row["pct_above_200dma_neg"]))
    checks = {
        "VIX > 35": bool(row["vix_level"] > DET_VIX),
        "Drawdown > 20%": bool(dd > DET_DD),
        "% Above 200DMA < 20%": bool(above < DET_ABOVE),
        "Credit Stress Elevated": bool(credit_pct >= DET_CREDIT_PCT),
        "Fear & Greed < 25": bool(fng < DET_FNG),
    }
    n = sum(checks.values())
    state = "YES" if n >= 3 else ("WATCHLIST" if n == 2 else "NO")
    return {"state": state, "n_conditions": n, "checks": checks,
            "rule": ">=3 simultaneous conditions = YES; exactly 2 = WATCHLIST; otherwise NO."}


def buy_score(val: float, fear: float, risk_score: float,
              bonus: float) -> tuple[float, float, float]:
    base = W_VAL * val + W_FEAR * fear + W_RISK * risk_score
    raw = base + bonus
    return float(np.clip(raw, 0, 100)), float(base), float(raw)


def analogue_similarity(cur: dict, anchors: dict) -> list[dict]:
    out = []
    for label, a in anchors.items():
        terms = {}
        for k, (lo, hi) in SIM_RANGE.items():
            diff = abs(cur[k] - a[k]) / max(hi - lo, 1e-9)
            terms[k] = float(np.clip(1.0 - diff, 0.0, 1.0))
        sim = float(np.mean(list(terms.values())) * 100.0)
        out.append({"episode": label, "date": a["date"],
                    "similarity": round(sim, 1),
                    "terms": {k: round(v * 100, 1) for k, v in terms.items()},
                    "features": {k: round(a[k], 2) for k in SIM_RANGE}})
    return sorted(out, key=lambda x: -x["similarity"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> dict:
    adj, raw_px = V4.load_frames()
    pit, macro_meta = V5.build_macro_pit(raw_px.index)
    daily = V5.build_daily_components(raw_px, pit)
    today = pd.Timestamp(raw_px.index.max())

    mc_pct = V5.expanding_pct(daily["mcclellan_neg"], daily.index, 252)
    credit_pct = V5.expanding_pct(daily["credit_stress_price"], daily.index, 252)
    fg_pct = V5.expanding_pct(daily["fear_greed_constructed"], daily.index, 252)
    # CNN convention: low = fear. Our composite percentile is high = fear.
    fng_index = 100.0 - fg_pct * 100.0

    valuation = valuation_components(daily, today)
    fear = fear_components(daily, today, mc_pct)

    # Risk probabilities need the full PIT history of prior states. Asking only
    # for today would yield NaN; v5 builds them over all month-end dates.
    hist_dates = [d for d in pd.date_range("2000-01-31", today, freq="ME")
                  if d in daily.index]
    risk_dates = hist_dates + [today]
    probs = V5.risk_probabilities(daily, risk_dates, adj)
    risks = V5.risk_subsignals(daily, risk_dates, probs)
    risk_row = risks.loc[today]
    meter = str(risk_row["meter"])
    risk_score = RISK_OPPORTUNITY[meter]

    instruments = {}
    for sym, proxy, bonus_tbl in (("VOO", "SPY", VOO_DD_BONUS),
                                  ("QQQ", "QQQ", QQQ_DD_BONUS)):
        dd = dd_depth(raw_px[proxy], today)
        bonus = float(np.interp(dd * 100.0, [x[0] for x in bonus_tbl],
                                [x[1] for x in bonus_tbl]))
        score, base, unbounded = buy_score(valuation["score"], fear["score"],
                                          risk_score, bonus)
        label, guidance = band_for(score)
        det = detector(daily, today, dd, float(credit_pct[today]), float(fng_index[today]))
        instruments[sym] = {
            "score": round(score, 1), "base_score": round(base, 1),
            "drawdown_bonus": round(bonus, 1), "unbounded_score": round(unbounded, 1),
            "label": label, "guidance": guidance,
            "drawdown_pct": round(dd * 100, 2),
            "detector": det,
        }

    # ---- historical analogue feature vectors -----------------------------
    def feat(at: pd.Timestamp) -> dict:
        row = daily.loc[at]
        return {
            "date": str(at.date()),
            "vix": float(row["vix_level"]),
            "dd": float(row["dd_depth"]),
            "above": 100.0 * (1.0 - float(row["pct_above_200dma_neg"])),
            "below": float(row["below_200dma"]),
            "valuation": float(V5.expanding_pct(daily["cape"], [at], 36).iloc[0] * 100),
            "credit": 100.0 * float(credit_pct[at]),
        }

    cur_feat = feat(today)
    anchor_feats = {}
    for label, ds in ANALOGUES:
        at = pd.Timestamp(ds)
        if at in daily.index:
            anchor_feats[label] = feat(at)
    analogues = analogue_similarity(cur_feat, anchor_feats)

    # ---- validate rare detector on required historical episodes ----------
    hist_examples = []
    for ds in HIST_EXAMPLES:
        at = pd.Timestamp(ds)
        if at not in daily.index:
            continue
        dd = dd_depth(raw_px["SPY"], at)
        det = detector(daily, at, dd, float(credit_pct[at]), float(fng_index[at]))
        hist_examples.append({"date": ds, "spy_drawdown_pct": round(dd * 100, 1),
                              "detector": det["state"], "n_conditions": det["n_conditions"]})

    # ---- buying-score band outcomes (descriptive, not a forecast) --------
    monthly = pd.date_range("2003-01-31", raw_px.index.max(), freq="ME")
    monthly = [d for d in monthly if d in daily.index]
    band_rows = []
    for d in monthly:
        v = valuation_components(daily, d)["score"]
        f = fear_components(daily, d, mc_pct)["score"]
        dd = dd_depth(raw_px["SPY"], d)
        bonus = float(np.interp(dd * 100.0, [x[0] for x in VOO_DD_BONUS],
                                [x[1] for x in VOO_DD_BONUS]))
        sc, _, _ = buy_score(v, f, RISK_OPPORTUNITY["Moderate"], bonus)
        lab, _ = band_for(sc)
        pos = int(adj.index.get_loc(d))
        r = V5.path_outcomes(adj, "SPY", pos, 252)["r"]
        if np.isfinite(r):
            band_rows.append({"score": sc, "label": lab, "r12": float(r)})
    band_outcomes = []
    for i, (thr, label, guidance) in enumerate(BANDS):
        lo = thr
        hi = BANDS[i - 1][0] if i > 0 else 100
        vals = [x["r12"] for x in band_rows if lo <= x["score"] < hi]
        band_outcomes.append({"band": label, "range": [lo, hi], "n": len(vals),
                              "median_12m_return_pct": (round(float(np.median(vals) * 100), 1)
                                                        if vals else None),
                              "win_rate_pct": (round(float(np.mean(np.asarray(vals) > 0) * 100), 1)
                                               if vals else None)})

    primary = instruments["VOO"]
    q = instruments["QQQ"]
    payload = {
        "meta": {
            "version": "6.0 — Entry Opportunity Engine (panic-driven accumulation)",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "as_of": str(today.date()),
            "weights": {"valuation": W_VAL, "fear_opportunity": W_FEAR, "risk": W_RISK},
            "fear_weights": FEAR_WEIGHTS,
            "philosophy": "Not a forecast engine. Extreme fear outweighs small valuation differences.",
            "price_basis": "RAW closes for levels; ADJUSTED closes only for descriptive outcome study",
            "macro_alignment": "first-known date (nominal + publication lag)",
            "inherited": "v4 price frames; v5 PIT macro alignment and risk primitives",
            "seed": SEED,
        },
        "final_page": {
            "question": "If I had excess cash today, how aggressively should I deploy it?",
            "voo_score": primary["score"], "voo_label": primary["label"],
            "qqq_score": q["score"], "qqq_label": q["label"],
            "detector": primary["detector"]["state"],
            "guidance": primary["guidance"],
            "always_answered": True,
        },
        "layers": {
            "valuation": valuation,
            "fear": fear,
            "risk": {"meter": meter, "opportunity_score": round(risk_score, 1),
                     "p10_3m": round(float(risk_row["p10"]) * 100, 1),
                     "p20_12m": round(float(risk_row["p20"]) * 100, 1),
                     "sub_signals": [
                         {"name": "3M -10% drawdown", "level": str(risk_row["dd10_band"])},
                         {"name": "12M -20% drawdown", "level": str(risk_row["dd20_band"])},
                         {"name": "Recession pressure", "level": str(risk_row["recession"])},
                         {"name": "Credit stress", "level": str(risk_row["credit"])},
                         {"name": "Liquidity stress", "level": str(risk_row["liquidity"])}]},
        },
        "instruments": instruments,
        "fear_greed": {"index": round(float(fng_index[today]), 1),
                       "convention": "0 = extreme fear, 100 = extreme greed (CNN scale)",
                       "source": "constructed from seven long-history PIT components; "
                                 "CNN's own feed has ~1 year and is context only"},
        "analogues": analogues,
        "current_features": {k: round(v, 2) for k, v in cur_feat.items() if k != "date"},
        "historical_validation": hist_examples,
        "band_outcomes": band_outcomes,
        "band_outcomes_note": "Descriptive check only. These groups were NOT used to choose knots, "
                              "weights or bonuses, and the samples are small. The engine does not "
                              "claim that high opportunity scores predict higher returns.",
        "disclosures": [
            "Absolute VIX/drawdown knots are pre-registered, not fitted to returns.",
            "Drawdown bonuses are additive and the final score is capped at 100.",
            "Rare detector uses SPY drawdown for the market-level signal; per-ETF drawdown is shown separately.",
            "Put/Call proxy is unavailable before 2011 and receives a neutral 50 in that period.",
            "This engine identifies panic-driven entry conditions; it does not predict returns.",
            "Similarity scores are descriptive distances across six fixed normalised features.",
        ],
        "data_quality": {"macro_sources": macro_meta},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    print(f"[v6] wrote {OUT.relative_to(ROOT)} | as-of {payload['meta']['as_of']}")
    print(f"[v6] valuation {valuation['score']:.1f} | fear {fear['score']:.1f} | "
          f"risk-opportunity {risk_score:.0f} | detector {primary['detector']['state']}")
    print(f"[v6] VOO {primary['score']} ({primary['label']}) | "
          f"QQQ {q['score']} ({q['label']}) | guidance: {primary['guidance']}")
    yes = [x["date"] for x in hist_examples if x["detector"] == "YES"]
    print(f"[v6] rare detector YES on {len(yes)}/{len(hist_examples)} historical examples: {yes}")
    return payload


if __name__ == "__main__":
    run()
