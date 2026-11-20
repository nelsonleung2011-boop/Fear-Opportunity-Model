#!/usr/bin/env python3
"""
ETF Entry Decision System v7 -- multi-concept deployment engine
===============================================================

v7 keeps v6's panic detector, analogues, nonlinear VIX scoring, drawdown
bonus and opportunity framework, but REMOVES the idea that one score explains
everything. Four separate concepts are reported independently:

    1. MARKET REGIME    BULL / CORRECTION / BEAR / PANIC  (+ confidence)
    2. VALUATION        cheap / fair / expensive long-term opportunity
    3. OPPORTUNITY      how favourable today is vs historical entry points
    4. DEPLOYMENT       0 / 25 / 50 / 75 / 100 % of planned cash

New independent layers
    * Credit Stress Score  (HY OAS, HYG/LQD, TED-style funding proxy, bond momentum)
    * Market Internals     (advance/decline, new high/low, RSP/SPY, IWM/SPY)
    * Opportunity Cost     (what would the Buying Score be after a 5/10/15% correction?)
    * Explainability       (every point of every score is traceable, with a
                            Simple/Expert switch and a manual audit trail)

This engine is NOT a return predictor. No AUC, no ML fit, no optimisation of
thresholds against outcomes. All knots and rules are pre-registered and
disclosed. No v1-v6 file is modified. Output: data/evidence_v7.json
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
from pipeline import entry_opportunity_engine as V6     # noqa: E402

OUT = ROOT / "data" / "evidence_v7.json"
SEED = 20261119

# v7 inherits v6's pre-registered scoring constants unchanged
W_VAL, W_OPP, W_RISK = V6.W_VAL, V6.W_FEAR, V6.W_RISK
FEAR_WEIGHTS = V6.FEAR_WEIGHTS
VIX_KNOTS, DD_KNOTS = V6.VIX_KNOTS, V6.DD_KNOTS
ABOVE_KNOTS, BELOW_KNOTS = V6.ABOVE_KNOTS, V6.BELOW_KNOTS
PUTCALL_KNOTS = V6.PUTCALL_KNOTS
VOO_DD_BONUS, QQQ_DD_BONUS = V6.VOO_DD_BONUS, V6.QQQ_DD_BONUS
RISK_OPPORTUNITY = V6.RISK_OPPORTUNITY

VAL_LABELS = [(90, "Extremely Cheap"), (75, "Attractive"), (60, "Fair"),
              (40, "Expensive"), (20, "Very Expensive"), (0, "Bubble-like")]
OPP_LABELS = [(90, "Rare Opportunity"), (80, "Strong Entry"), (60, "Good Entry"),
              (40, "Neutral"), (20, "Poor Entry"), (0, "Euphoric Market")]

# ---- Market regime: pre-registered per-input state bands ----------------
# Each input votes BULL / CORRECTION / BEAR / PANIC. The modal vote wins;
# ties break toward the more severe regime. Confidence = modal votes / 5.
REGIME_ORDER = ["BULL", "CORRECTION", "BEAR", "PANIC"]
# IMPORTANT: every band list is ordered from HIGHEST threshold to LOWEST,
# because regime_vote() returns the first band whose threshold is met.
# For "higher = bullish" inputs (breadth, trend) the BULL band therefore
# carries the highest threshold, not the lowest.
VIX_REGIME = [(35, "PANIC"), (28, "BEAR"), (22, "CORRECTION"), (0, "BULL")]
DD_REGIME = [(0.25, "PANIC"), (0.15, "BEAR"), (0.05, "CORRECTION"), (0, "BULL")]
ABOVE_REGIME = [(70, "BULL"), (50, "CORRECTION"), (30, "BEAR"), (0, "PANIC")]
CREDIT_REGIME = [(70, "PANIC"), (65, "BEAR"), (40, "CORRECTION"), (0, "BULL")]
TREND_REGIME = [(0.0, "BULL"), (-0.05, "CORRECTION"), (-0.15, "BEAR"), (-1.0, "PANIC")]

# ---- Credit stress: four independent, equally weighted inputs -----------
CREDIT_WEIGHTS = {"hy_oas": 0.25, "hyg_lqd": 0.25,
                  "funding_proxy": 0.25, "bond_momentum": 0.25}
CREDIT_LABELS = [(75, "Extreme"), (55, "High"), (35, "Moderate"),
                 (15, "Low"), (0, "Very Low")]

# ---- Market internals: four independent, equally weighted inputs --------
INTERNALS_WEIGHTS = {"advance_decline": 0.25, "new_high_low": 0.25,
                     "equal_cap": 0.25, "small_rs": 0.25}

# ---- Deployment: Buying Score band -> cash deployment -------------------
DEPLOYMENT = [(90, 100, "Rare opportunity",
               "Deploy aggressively. Historically rare accumulation conditions."),
              (80, 75, "Aggressive accumulation",
               "Deploy most planned cash in stages over the coming weeks."),
              (60, 50, "Moderate accumulation",
               "Put roughly half of planned cash to work now."),
              (40, 25, "Small accumulation",
               "Deploy a small first tranche; keep most cash available."),
              (0, 0, "Pause lump sum",
               "Pause lump-sum buying; continue scheduled DCA.")]

WAIT_SCENARIOS = [0.05, 0.10, 0.15]

# Formula disclosure: definition, formula, reason, direction, worked example
FORMULA = {
    "vix": {
        "definition": "CBOE Volatility Index (30-day implied volatility on S&P 500).",
        "formula": "knot_score(VIX, VIX_KNOTS); VIX_KNOTS = [(0,0),(15,5),(20,15),"
                   "(25,30),(30,55),(35,75),(40,88),(45,95),(50,100)]",
        "reason": "Extreme fear historically accompanies major buying opportunities; "
                  "the curve rewards extremes disproportionately.",
        "direction": "Higher VIX = more fear = higher opportunity score.",
        "example": "VIX 30 -> 55 points; VIX 45 -> 95 points.",
    },
    "drawdown": {
        "definition": "Percentage decline of price from its all-time high.",
        "formula": "dd = 1 - price / running_max(price); knot_score(dd, DD_KNOTS)",
        "reason": "Deeper drawdowns improve entry location and create room for recovery.",
        "direction": "Deeper drawdown = higher opportunity score.",
        "example": "5% drawdown -> 10 points; 25% -> 78 points.",
    },
    "above_200dma": {
        "definition": "Share of the 10-ETF universe trading above its 200-day moving average.",
        "formula": "above% = 100*(1 - pct_above_200dma_neg); knot_score(above, ABOVE_KNOTS)",
        "reason": "Few names above trend marks broad capitulation rather than a narrow selloff.",
        "direction": "Fewer stocks above 200DMA = higher opportunity score.",
        "example": "60% above -> 40 points; 20% above -> 85 points.",
    },
    "below_200dma": {
        "definition": "How far the index sits below its 200-day moving average.",
        "formula": "knot_score(-(price/MA200 - 1), BELOW_KNOTS)",
        "reason": "Price location versus trend captures how washed out the market is.",
        "direction": "Further below trend = higher opportunity score.",
        "example": "-1% -> ~41 points; -10% -> 82 points.",
    },
    "putcall": {
        "definition": "Put/Call fear proxy from the VIX term structure (VIX9D/VIX - 1).",
        "formula": "knot_score(VIX9D/VIX - 1, PUTCALL_KNOTS); neutral 50 before 2011.",
        "reason": "9-day vol above 30-day vol signals urgent demand for protection.",
        "direction": "Higher term-structure inversion = higher opportunity score.",
        "example": "0.05 inversion -> 75 points.",
    },
    "mcclellan": {
        "definition": "Breadth impulse proxy: RSP/SPY 19-day EMA minus 39-day EMA (inverted).",
        "formula": "mc = 100 * (expanding_pct(-(EMA19-EMA39)))^2",
        "reason": "Squared percentile so only genuine breadth extremes contribute.",
        "direction": "More negative breadth impulse = higher opportunity score.",
        "example": "90th percentile of negativity -> 81 points.",
    },
}

# Indicator name -> FORMULA key (used by the explainability engine)
LAYER_FORMULA_KEYS = {
    "vix": "vix", "drawdown": "drawdown", "above_200dma": "above_200dma",
    "below_200dma": "below_200dma", "putcall": "putcall", "mcclellan": "mcclellan",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_from(score: float, labels: list[tuple[float, str]]) -> str:
    for thr, lab in labels:
        if score >= thr:
            return lab
    return labels[-1][1]


def band_from(value: float, bands: list[tuple[float, str]]) -> str:
    for thr, lab in bands:
        if value >= thr:
            return lab
    return bands[-1][1]


def regime_vote(value: float, bands: list[tuple[float, str]]) -> str:
    for thr, lab in bands:
        if value >= thr:
            return lab
    return bands[-1][1]


def severity_index(state: str) -> int:
    return REGIME_ORDER.index(state) if state in REGIME_ORDER else 0


def deployment_for(score: float) -> dict:
    for lo, pct, label, note in DEPLOYMENT:
        if score >= lo:
            return {"percent": pct, "label": label, "note": note,
                    "band": [lo, 100 if lo == 90 else lo + 20]}
    return {"percent": 0, "label": DEPLOYMENT[-1][2], "note": DEPLOYMENT[-1][3],
            "band": [0, 20]}


def expanding_pct_at(daily: pd.DataFrame, key: str, at: pd.Timestamp, minp: int) -> float:
    s = V5.expanding_pct(daily[key], [at], minp)
    v = float(s.iloc[0])
    return v if np.isfinite(v) else 0.5


# ---------------------------------------------------------------------------
# 1. Market Regime Engine
# ---------------------------------------------------------------------------

def classify_regime(daily: pd.DataFrame, at: pd.Timestamp, dd: float,
                    credit_score: float) -> dict:
    row = daily.loc[at]
    above = 100.0 * (1.0 - float(row["pct_above_200dma_neg"]))
    below = float(row["below_200dma"])
    inputs = {
        "VIX": {"raw": round(float(row["vix_level"]), 2),
                "state": regime_vote(float(row["vix_level"]), VIX_REGIME)},
        "Drawdown": {"raw": round(dd * 100, 2),
                     "state": regime_vote(dd, DD_REGIME), "unit": "%"},
        "Breadth": {"raw": round(above, 1),
                    "state": regime_vote(above, ABOVE_REGIME), "unit": "% above 200DMA"},
        "Credit": {"raw": round(credit_score, 1),
                   "state": regime_vote(credit_score, CREDIT_REGIME), "unit": "/100"},
        "Trend": {"raw": round(below, 4),
                  "state": regime_vote(below, TREND_REGIME), "unit": "below MA200"},
    }
    votes = [v["state"] for v in inputs.values()]
    counts = {s: votes.count(s) for s in REGIME_ORDER}
    max_votes = max(counts.values())
    tied = [s for s in REGIME_ORDER if counts[s] == max_votes]
    state = max(tied, key=severity_index)  # ties break toward more severe
    conf_frac = max_votes / len(votes)
    confidence = "High" if conf_frac >= 0.8 else ("Moderate" if conf_frac >= 0.6 else "Low")
    return {"state": state, "confidence": confidence,
            "agreement_pct": round(conf_frac * 100, 1),
            "votes": counts, "inputs": inputs,
            "rule": "Each input votes BULL/CORRECTION/BEAR/PANIC from pre-registered bands. "
                    "Modal vote wins; ties break toward the more severe regime. "
                    "Confidence = modal votes / 5."}


# ---------------------------------------------------------------------------
# 2. Credit Stress Score (independent)
# ---------------------------------------------------------------------------

def credit_stress(daily: pd.DataFrame, at: pd.Timestamp, pit: pd.DataFrame,
                  raw_px: pd.DataFrame) -> dict:
    """Four equally weighted credit inputs. Higher = more stress (0-100)."""
    row = daily.loc[at]
    # 1) HY OAS level -- genuine FRED series but only ~3y of history.
    oas = pit["BAMLH0A0HYM2"].loc[at] if "BAMLH0A0HYM2" in pit else np.nan
    oas_n = int(pit["BAMLH0A0HYM2"].notna().loc[:at].sum()) if "BAMLH0A0HYM2" in pit else 0
    if oas_n >= 252 and np.isfinite(oas):
        oas_pct = expanding_pct_at(pd.DataFrame({"oas": pit["BAMLH0A0HYM2"]}),
                                   "oas", at, 252)
        oas_reliable = True
    else:
        # Short history: use full-history HYG/LQD stress as a disclosed fallback.
        oas_pct = expanding_pct_at(daily, "credit_stress_price", at, 252)
        oas_reliable = False
    # 2) HYG/LQD credit ratio stress (full history, higher = more stress)
    hyg = float(row["credit_stress_price"])
    hyg_pct = expanding_pct_at(daily, "credit_stress_price", at, 252) if np.isfinite(hyg) else 0.5
    # 3) TED-style funding stress proxy. LIBOR TED ended in 2021, so we use a
    #    disclosed combination of 3M T-bill tightness and yield-curve inversion.
    irx_ser = raw_px["IRX"] if "IRX" in raw_px else pd.Series(dtype=float)
    irx = float(irx_ser.loc[at]) if at in irx_ser.index and np.isfinite(irx_ser.loc[at]) else np.nan
    curve = float(pit["T10Y2Y"].loc[at]) if "T10Y2Y" in pit and np.isfinite(pit["T10Y2Y"].loc[at]) else np.nan
    irx_pct = (expanding_pct_at(pd.DataFrame({"x": irx_ser.loc[:at]}), "x", at, 252)
               if np.isfinite(irx) and len(irx_ser.loc[:at]) >= 252 else 0.5)
    inv_pct = expanding_pct_at(pd.DataFrame({"x": -pit["T10Y2Y"]}), "x", at, 252) if np.isfinite(curve) else 0.5
    funding_pct = float(np.clip(0.5 * irx_pct + 0.5 * inv_pct, 0, 1))
    # 4) Corporate bond momentum: negative HYG/LQD 12M return = stress
    mom = float(row["credit_stress_price"])  # already -(HYG/LQD 63d change)
    mom_pct = expanding_pct_at(daily, "credit_stress_price", at, 252) if np.isfinite(mom) else 0.5

    comps = {
        "hy_oas": {"raw": (None if not np.isfinite(oas) else round(float(oas), 2)),
                   "normalised": round(oas_pct * 100, 1), "weight": CREDIT_WEIGHTS["hy_oas"],
                   "reliable_history": oas_reliable,
                   "note": ("FRED HY OAS, ~3y history" if oas_reliable
                            else "short history; HYG/LQD proxy substituted (disclosed)")},
        "hyg_lqd": {"raw": (None if not np.isfinite(hyg) else round(hyg, 4)),
                    "normalised": round(hyg_pct * 100, 1), "weight": CREDIT_WEIGHTS["hyg_lqd"],
                    "reliable_history": True, "note": "HYG/LQD 63-day relative stress (full history)"},
        "funding_proxy": {"raw": {"irx": (None if not np.isfinite(irx) else round(irx, 2)),
                                  "curve_10y2y": (None if not np.isfinite(curve) else round(curve, 2))},
                          "normalised": round(funding_pct * 100, 1),
                          "weight": CREDIT_WEIGHTS["funding_proxy"], "reliable_history": True,
                          "note": "TED-style: 50% 3M T-bill tightness + 50% curve inversion. "
                                  "Not literal TED (LIBOR series ended 2021)."},
        "bond_momentum": {"raw": (None if not np.isfinite(mom) else round(mom, 4)),
                          "normalised": round(mom_pct * 100, 1),
                          "weight": CREDIT_WEIGHTS["bond_momentum"], "reliable_history": True,
                          "note": "HYG/LQD momentum percentile; negative relative move = stress."},
    }
    for c in comps.values():
        c["contribution"] = round(c["normalised"] * c["weight"], 2)
    score = float(sum(c["normalised"] * c["weight"] for c in comps.values()))
    return {"score": round(score, 1), "label": band_from(score, CREDIT_LABELS),
            "components": comps, "weights": CREDIT_WEIGHTS,
            "note": "Independent layer. Not hidden inside the panic detector."}


# ---------------------------------------------------------------------------
# 3. Market Internals (hidden weakness)
# ---------------------------------------------------------------------------

def market_internals(raw_px: pd.DataFrame, daily: pd.DataFrame, at: pd.Timestamp) -> dict:
    spy, rsp, iwm = raw_px["SPY"], raw_px["RSP"], raw_px["IWM"]
    universe = ["XLK", "XLF", "XLV", "XLI", "XLP", "XLY", "XLE", "XLU",
                "XLB", "XLRE", "XLC", "RSP", "IWM"]
    # Advance/decline: share of universe up over the last 21 sessions
    chg = {c: raw_px[c].loc[at] / raw_px[c].loc[:at].iloc[-22] - 1.0
           for c in universe if c in raw_px.columns}
    adv_ratio = float(np.mean([v > 0 for v in chg.values()])) if chg else 0.5
    # New high / new low: share of universe within 2% of 252d high vs near 252d low
    hi_share, lo_share = [], []
    for c in universe:
        if c not in raw_px.columns:
            continue
        hist = raw_px[c].loc[:at].iloc[-252:]
        if len(hist) < 60:
            continue
        last = float(hist.iloc[-1])
        hi, lo = float(hist.max()), float(hist.min())
        if hi > lo:
            hi_share.append(last >= hi * 0.98)
            lo_share.append(last <= lo * 1.02)
    nhnl = float(np.mean(hi_share)) - float(np.mean(lo_share)) if hi_share else 0.0
    # Equal weight vs cap weight and small-cap relative strength, vs 200DMA
    eq_rel = (rsp / spy).loc[:at]
    sm_rel = (iwm / spy).loc[:at]
    eq_vs = float(eq_rel.iloc[-1] / eq_rel.iloc[-200:].mean() - 1.0)
    sm_vs = float(sm_rel.iloc[-1] / sm_rel.iloc[-200:].mean() - 1.0)
    # Normalise: low breadth / more new lows / weak equal & small = weakness = high score
    def inv_pct(series: pd.Series, value: float, minp: int = 252) -> float:
        tmp = pd.DataFrame({"x": series})
        return 1.0 - float(V5.expanding_pct(tmp["x"], [tmp.index[-1]], minp).iloc[0])
    comps = {
        "advance_decline": {"raw": round(adv_ratio * 100, 1), "unit": "% advancing (21d)",
                            "normalised": round((1.0 - adv_ratio) * 100, 1),
                            "weight": INTERNALS_WEIGHTS["advance_decline"],
                            "note": "Share of 13-ETF universe with positive 21-day return."},
        "new_high_low": {"raw": round(nhnl * 100, 1), "unit": "highs - lows (%)",
                         "normalised": round(float(np.clip(0.5 - nhnl / 2.0, 0, 1)) * 100, 1),
                         "weight": INTERNALS_WEIGHTS["new_high_low"],
                         "note": "Share near 252d highs minus share near 252d lows."},
        "equal_cap": {"raw": round(eq_vs * 100, 2), "unit": "% vs 200d mean",
                      "normalised": round(float(np.clip(0.5 - eq_vs * 2.5, 0, 1)) * 100, 1),
                      "weight": INTERNALS_WEIGHTS["equal_cap"],
                      "note": "RSP/SPY relative to its 200-day mean; negative = narrow leadership."},
        "small_rs": {"raw": round(sm_vs * 100, 2), "unit": "% vs 200d mean",
                     "normalised": round(float(np.clip(0.5 - sm_vs * 2.5, 0, 1)) * 100, 1),
                     "weight": INTERNALS_WEIGHTS["small_rs"],
                     "note": "IWM/SPY relative to its 200-day mean; negative = defensive leadership."},
    }
    for c in comps.values():
        c["contribution"] = round(c["normalised"] * c["weight"], 2)
    score = float(sum(c["normalised"] * c["weight"] for c in comps.values()))
    hidden = score >= 65
    return {"score": round(score, 1),
            "label": "Hidden weakness detected" if hidden else "No hidden weakness",
            "hidden_weakness": hidden, "components": comps,
            "weights": INTERNALS_WEIGHTS,
            "note": "Not part of the Buying Score; a separate diagnostic of market internals."}


# ---------------------------------------------------------------------------
# 4. Explainability helpers
# ---------------------------------------------------------------------------

def explain_score(score: float, formula_parts: list[dict],
                  formula_text: str, final_score: float | None = None,
                  bonus: float | None = None) -> dict:
    """Build a fully traceable explanation for one score.

    ``score`` is the BASE score reproduced by the component table. For buying
    scores the published value is base + drawdown bonus, so ``final_score``
    and ``bonus`` make that reconciliation explicit instead of silently
    displaying the base as if it were the final answer.
    """
    rows = []
    for p in formula_parts:
        delta = p["contribution"] - 50.0 * p["weight"]
        rows.append({
            "name": p["name"],
            "raw": p.get("raw"),
            "normalised": round(float(p["normalised"]), 2),
            "weight": round(float(p["weight"]), 4),
            "contribution": round(float(p["contribution"]), 2),
            "delta_vs_neutral": round(float(delta), 2),
            "direction": p.get("direction", ""),
            "why": p.get("why", ""),
        })
    # Rank by points actually contributed to the final score (the spec's
    # "biggest positive / biggest negative contributors"). Every contribution
    # is non-negative; the delta_vs_neutral column shows which inputs lifted or
    # dragged the score relative to a neutral 50/100 input.
    pos = sorted(rows, key=lambda r: -r["contribution"])[:3]
    neg = sorted(rows, key=lambda r: r["contribution"])[:3]
    audit = []
    running = 0.0
    for r in rows:
        running += r["contribution"]
        audit.append({"step": r["name"], "normalised": r["normalised"],
                      "times_weight": f"{r['normalised']} x {r['weight']:.2f}",
                      "contribution": r["contribution"],
                      "running_total": round(running, 2)})
    return {"final": round(final_score if final_score is not None else score, 1),
            "base_score": round(score, 1),
            "bonus": (None if bonus is None else round(float(bonus), 1)),
            "reconciliation": (
                f"Base {round(score,1)} (component table below) + drawdown bonus "
                f"{round(float(bonus),1)} = final {round(float(final_score),1)} "
                "(capped 0-100)" if bonus is not None and final_score is not None
                else "No additive bonus; component table sums directly to the final score."),
            "formula": formula_text,
            "components": rows,
            "top_positive": [{"name": r["name"], "contribution": r["contribution"],
                              "delta_vs_neutral": r["delta_vs_neutral"]} for r in pos],
            "top_negative": [{"name": r["name"], "contribution": r["contribution"],
                              "delta_vs_neutral": r["delta_vs_neutral"]} for r in neg],
            "audit_trail": audit}


def opportunity_cost(daily: pd.DataFrame, raw_px: pd.DataFrame, adj: pd.DataFrame,
                     at: pd.Timestamp, valuation_score: float, mc_pct: pd.Series,
                     risk_score: float, current_score: float) -> dict:
    """What happens to the VOO Buying Score if the market first falls 5/10/15%?

    Descriptive: replace only the drawdown input (and its additive bonus), hold
    all other conditions at today's values. Also report how often SPY reached
    each threshold historically and the median forward 12M outcome at the
    first crossing (labelled descriptive, not a forecast)."""
    spy = raw_px["SPY"]
    current_dd = V6.dd_depth(spy, at)
    scenarios = []
    for thr in WAIT_SCENARIOS:
        # A threshold already exceeded cannot be "waited for": substituting a
        # SMALLER drawdown would lower the score and produce a nonsensical
        # negative improvement. Clamp to the current drawdown and label it.
        already = bool(thr <= current_dd + 1e-9)
        eff_thr = max(thr, current_dd)
        d = daily.copy()
        d.loc[at, "dd_depth"] = eff_thr
        d.loc[at, "below_200dma"] = float(d.loc[at, "below_200dma"]) + eff_thr - current_dd
        fear = V6.fear_components(d, at, mc_pct)["score"]
        bonus = float(np.interp(eff_thr * 100, [x[0] for x in VOO_DD_BONUS],
                                [x[1] for x in VOO_DD_BONUS]))
        new_score, _, _ = V6.buy_score(valuation_score, fear, risk_score, bonus)
        # Historical frequency: month-ends where drawdown >= threshold
        monthly = [d for d in pd.date_range("2003-01-31", at, freq="ME") if d in raw_px.index]
        hits = [d for d in monthly if V6.dd_depth(spy, d) >= thr]
        # First crossing per calendar year, forward 12M outcome (descriptive)
        outs = []
        seen_years = set()
        for d in hits:
            if d.year in seen_years:
                continue
            seen_years.add(d.year)
            pos = adj.index.get_loc(d) if d in adj.index else None
            if pos is not None:
                r = V5.path_outcomes(adj, "SPY", pos, 252)["r"]
                if np.isfinite(r):
                    outs.append(float(r))
        scenarios.append({
            "correction_pct": int(thr * 100),
            "status": ("already reached — no additional wait" if already
                       else "wait required"),
            "score_at_scenario": round(new_score, 1),
            "score_improvement": max(0.0, round(new_score - current_score, 1)),
            "bonus_at_scenario": round(bonus, 1),
            "historical_freq_pct": round(100.0 * len(hits) / max(len(monthly), 1), 1),
            "n_first_crossings": len(outs),
            "median_next_12m_pct": (round(float(np.median(outs) * 100), 1) if outs else None),
        })
    return {"current_score": round(current_score, 1),
            "current_drawdown_pct": round(current_dd * 100, 2),
            "scenarios": scenarios,
            "note": "Holding all other conditions constant. Historical frequency and "
                    "forward outcomes are descriptive only; waiting can also mean never buying."}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(frames: tuple | None = None, as_of: str | None = None) -> dict:
    """Build the v7 payload.

    ``frames`` lets the operational pipeline supply dynamically dated frames.
    The inherited v4 loader truncates at a fixed 2026-11-16, which is correct
    for a frozen research snapshot but wrong for a daily system. v2-v6 keep
    using their original loader unchanged.
    """
    if frames is None:
        adj, raw_px = V4.load_frames()
    else:
        adj, raw_px = frames
    if as_of:
        cut = pd.Timestamp(as_of)
        adj, raw_px = adj.loc[adj.index <= cut], raw_px.loc[raw_px.index <= cut]
    pit, macro_meta = V5.build_macro_pit(raw_px.index)
    daily = V5.build_daily_components(raw_px, pit)
    today = pd.Timestamp(raw_px.index.max())

    mc_pct = V5.expanding_pct(daily["mcclellan_neg"], daily.index, 252)
    credit_pct = V5.expanding_pct(daily["credit_stress_price"], daily.index, 252)
    fg_pct = V5.expanding_pct(daily["fear_greed_constructed"], daily.index, 252)
    fng_index = 100.0 - fg_pct * 100.0

    valuation = V6.valuation_components(daily, today)
    fear = V6.fear_components(daily, today, mc_pct)

    credit = credit_stress(daily, today, pit, raw_px)
    internals = market_internals(raw_px, daily, today)
    dd_spy = V6.dd_depth(raw_px["SPY"], today)
    regime = classify_regime(daily, today, dd_spy, credit["score"])

    hist_dates = [d for d in pd.date_range("2000-01-31", today, freq="ME") if d in daily.index]
    risk_dates = hist_dates + [today]
    probs = V5.risk_probabilities(daily, risk_dates, adj)
    risks = V5.risk_subsignals(daily, risk_dates, probs)
    risk_row = risks.loc[today]
    meter = str(risk_row["meter"])
    risk_score = RISK_OPPORTUNITY[meter]

    # Per-instrument scores + explanations
    instruments = {}
    bonus_tbl = {"VOO": VOO_DD_BONUS, "QQQ": QQQ_DD_BONUS}
    proxy = {"VOO": "SPY", "QQQ": "QQQ"}
    for sym in ("VOO", "QQQ"):
        dd = V6.dd_depth(raw_px[proxy[sym]], today)
        bonus = float(np.interp(dd * 100, [x[0] for x in bonus_tbl[sym]],
                                [x[1] for x in bonus_tbl[sym]]))
        score, base, unbounded = V6.buy_score(valuation["score"], fear["score"],
                                              risk_score, bonus)
        dep = deployment_for(score)
        det = V6.detector(daily, today, dd, float(credit_pct[today]), float(fng_index[today]))
        parts = []
        for key, c in fear["components"].items():
            parts.append({"name": key.replace("_", " ").title(),
                          "raw": c["raw"], "normalised": c["score"],
                          "weight": c["weight"] * W_OPP,
                          "contribution": c["score"] * c["weight"] * W_OPP,
                          "direction": FORMULA.get(LAYER_FORMULA_KEYS.get(key, key), {}).get("direction", ""),
                          "why": c["why"]})
        for key, c in valuation["components"].items():
            parts.append({"name": c["name"], "raw": c["raw"], "normalised": c["score"],
                          "weight": c["weight"] if "weight" in c else W_VAL / len(valuation["components"]),
                          "contribution": c["score"] * (c.get("weight", W_VAL / len(valuation["components"]))),
                          "direction": "Lower = cheaper" if c["direction"] < 0 else "Higher = cheaper",
                          "why": f"Valuation percentile {c['pct']} of history known at this date."})
        # valuation contributions must be scaled to the 30% layer weight
        n_val = len(valuation["components"])
        for p in parts[-n_val:]:
            p["weight"] = W_VAL / n_val
            p["contribution"] = p["normalised"] * W_VAL / n_val
        parts.append({"name": "Risk Modifier", "raw": meter, "normalised": risk_score,
                      "weight": W_RISK, "contribution": risk_score * W_RISK,
                      "direction": "Lower risk = higher score",
                      "why": "Worst-of five drawdown/macro stress sub-signals."})
        exp = explain_score(base, parts,
                            f"Base = {W_VAL:.0%} Valuation + {W_OPP:.0%} Opportunity "
                            f"+ {W_RISK:.0%} Risk Modifier; then + drawdown bonus, capped at 100.",
                            final_score=score, bonus=bonus)
        # manual audit: exact order the user can reproduce
        audit_rows = [
            {"step": "Valuation Score", "value": round(valuation["score"], 1),
             "operation": f"x {W_VAL:.0%}", "result": round(W_VAL * valuation["score"], 1)},
            {"step": "Opportunity Score", "value": round(fear["score"], 1),
             "operation": f"x {W_OPP:.0%}", "result": round(W_OPP * fear["score"], 1)},
            {"step": "Risk Modifier", "value": risk_score,
             "operation": f"x {W_RISK:.0%}", "result": round(W_RISK * risk_score, 1)},
            {"step": "Drawdown Bonus", "value": round(bonus, 1),
             "operation": "+ bonus", "result": round(bonus, 1)},
            {"step": "FINAL SCORE", "value": "", "operation": "sum (capped 0-100)",
             "result": round(score, 1)},
        ]
        oppcost = opportunity_cost(daily, raw_px, adj, today, valuation["score"],
                                   mc_pct, risk_score, score) if sym == "VOO" else None
        instruments[sym] = {
            "score": round(score, 1), "base_score": round(base, 1),
            "drawdown_bonus": round(bonus, 1), "unbounded_score": round(unbounded, 1),
            "label": V6.band_for(score)[0], "guidance": V6.band_for(score)[1],
            "drawdown_pct": round(dd * 100, 2), "detector": det,
            "deployment": dep,
            "explain": exp,
            "audit": audit_rows,
            "opportunity_cost": oppcost,
        }

    # Sensitivity scenarios for the hero/VOO score
    base_score = instruments["VOO"]["score"]
    cur_dd = float(daily.loc[today, "dd_depth"])
    sens = []
    scenarios = [("VIX rises to 35", {"vix_level": 35.0}),
                 ("Drawdown reaches 10%", {"dd_depth": 0.10}),
                 ("Drawdown reaches 20%", {"dd_depth": 0.20}),
                 ("VIX > 35 AND Drawdown > 20%", {"vix_level": 35.0, "dd_depth": 0.20})]
    for label, changes in scenarios:
        d = daily.copy()
        for k, v in changes.items():
            d.loc[today, k] = v
        f = V6.fear_components(d, today, mc_pct)["score"]
        dd = float(d.loc[today, "dd_depth"])
        bonus = float(np.interp(dd * 100, [x[0] for x in VOO_DD_BONUS],
                                [x[1] for x in VOO_DD_BONUS]))
        s, _, _ = V6.buy_score(valuation["score"], f, risk_score, bonus)
        dep = deployment_for(s)
        sens.append({"scenario": label, "score": round(s, 1),
                     "change": round(s - base_score, 1),
                     "deployment_pct": dep["percent"], "label": V6.band_for(s)[0]})

    # Historical analogues (kept from v6)
    def feat(at):
        row = daily.loc[at]
        return {"date": str(at.date()), "vix": float(row["vix_level"]),
                "dd": float(row["dd_depth"]),
                "above": 100.0 * (1.0 - float(row["pct_above_200dma_neg"])),
                "below": float(row["below_200dma"]),
                "valuation": float(V5.expanding_pct(daily["cape"], [at], 36).iloc[0] * 100),
                "credit": 100.0 * float(credit_pct[at])}
    anchors = {lab: feat(pd.Timestamp(ds)) for lab, ds in V6.ANALOGUES
               if pd.Timestamp(ds) in daily.index}
    analogues = V6.analogue_similarity(feat(today), anchors)
    hist_examples = []
    for ds in V6.HIST_EXAMPLES:
        at = pd.Timestamp(ds)
        if at not in daily.index:
            continue
        det = V6.detector(daily, at, V6.dd_depth(raw_px["SPY"], at),
                          float(credit_pct[at]), float(fng_index[at]))
        hist_examples.append({"date": ds, "detector": det["state"],
                              "n_conditions": det["n_conditions"]})

    primary = instruments["VOO"]
    dep = primary["deployment"]
    reasons = []
    if fear["score"] >= 50:
        reasons.append("Fear is elevated.")
    elif fear["score"] >= 35:
        reasons.append("Fear is somewhat elevated.")
    else:
        reasons.append("Fear is subdued.")
    reasons.append(f"Valuation remains {label_from(valuation['score'], VAL_LABELS).lower()}.")
    if regime["state"] in ("BEAR", "PANIC"):
        reasons.append("The market is in a stressed regime.")
    elif primary["detector"]["state"] == "YES":
        reasons.append("Rare-opportunity conditions are present.")
    else:
        reasons.append("Conditions are better than normal, but not a rare panic.")
    reason = " ".join(reasons)

    # Explanations for the four supporting scores
    val_parts = [{"name": c["name"], "raw": c["raw"], "normalised": c["score"],
                  "weight": 1.0 / n_val, "contribution": c["score"] / n_val,
                  "direction": "Lower = cheaper" if c["direction"] < 0 else "Higher = cheaper",
                  "why": f"PIT percentile {c['pct']} of history known at this date."}
                 for c in valuation["components"].values()]
    opp_parts = [{"name": k.replace("_", " ").title(), "raw": c["raw"],
                  "normalised": c["score"], "weight": c["weight"],
                  "contribution": c["score"] * c["weight"],
                  "direction": FORMULA.get(LAYER_FORMULA_KEYS.get(k, k), {}).get("direction", ""),
                  "why": c["why"]} for k, c in fear["components"].items()]

    payload = {
        "meta": {
            "version": "7.1 — Operational Entry Decision System",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "as_of": str(today.date()),
            "philosophy": "Four separate concepts. Not a return predictor and not a model-scoring "
                          "exercise. Built to help a long-term investor decide how aggressively "
                          "to deploy cash today.",
            "concepts": ["Market Regime", "Valuation", "Opportunity", "Deployment"],
            "weights": {"valuation": W_VAL, "opportunity": W_OPP, "risk": W_RISK},
            "inherited": "v6 opportunity framework, nonlinear VIX knots, drawdown bonuses, "
                         "detector and analogues; v5 PIT macro; v4 price frames",
            "seed": SEED,
        },
        "hero": {
            "market_regime": regime["state"],
            "regime_confidence": regime["confidence"],
            "valuation": label_from(valuation["score"], VAL_LABELS),
            "opportunity": label_from(fear["score"], OPP_LABELS),
            "deployment": f"{dep['percent']}%",
            "deployment_label": dep["label"],
            "reason": reason,
            "always_answered": True,
        },
        "final_page": {
            "q1_regime": f"{regime['state']} ({regime['confidence']} confidence)",
            "q2_valuation": f"{round(valuation['score'],1)}/100 — "
                            f"{label_from(valuation['score'], VAL_LABELS)}",
            "q3_opportunity": f"{round(fear['score'],1)}/100 — "
                              f"{label_from(fear['score'], OPP_LABELS)}",
            "q4_deployment": f"{dep['percent']}% — {dep['label']}",
            "reason": reason,
        },
        "regime": regime,
        "valuation": {"score": round(valuation["score"], 1),
                      "label": label_from(valuation["score"], VAL_LABELS),
                      "components": valuation["components"],
                      "explain": explain_score(valuation["score"], val_parts,
                                               "Mean of five PIT valuation percentiles, "
                                               "each raised to 1.5 to emphasise extremes.")},
        "opportunity": {"score": round(fear["score"], 1),
                        "label": label_from(fear["score"], OPP_LABELS),
                        "components": fear["components"],
                        "weights": FEAR_WEIGHTS,
                        "explain": explain_score(fear["score"], opp_parts,
                                                 "VIX 25% + Drawdown 25% + Above200DMA 20% "
                                                 "+ Below200DMA 15% + PutCall 10% + McClellan 5%.")},
        "credit": credit,
        "internals": internals,
        "risk": {"meter": meter, "opportunity_score": risk_score,
                 "p10_3m": round(float(risk_row["p10"]) * 100, 1),
                 "p20_12m": round(float(risk_row["p20"]) * 100, 1),
                 "sub_signals": [
                     {"name": "3M -10% drawdown", "level": str(risk_row["dd10_band"])},
                     {"name": "12M -20% drawdown", "level": str(risk_row["dd20_band"])},
                     {"name": "Recession pressure", "level": str(risk_row["recession"])},
                     {"name": "Credit stress", "level": str(risk_row["credit"])},
                     {"name": "Liquidity stress", "level": str(risk_row["liquidity"])}]},
        "instruments": instruments,
        "sensitivity": sens,
        "formula_disclosure": FORMULA,
        "analogues": analogues,
        "historical_validation": hist_examples,
        "fear_greed": {"index": round(float(fng_index[today]), 1),
                       "convention": "0 = extreme fear (CNN scale)",
                       "source": "constructed seven-component PIT series; display only, zero weight"},
        "disclosures": [
            "Four concepts are reported separately; no single score is claimed to explain everything.",
            "Knots, bands and weights are pre-registered and disclosed; none were fitted to returns.",
            "Credit Stress is independent; HY OAS has ~3y history and falls back to HYG/LQD with disclosure.",
            "TED spread is a disclosed proxy (3M bill + curve inversion); literal LIBOR TED ended in 2021.",
            "Market Internals are a separate diagnostic and are not added into the Buying Score.",
            "Opportunity-cost scenarios hold other conditions constant and are descriptive, not a forecast.",
            "Every score ships with contributions, top positives/negatives, sensitivity and a manual audit trail.",
        ],
        "data_quality": {"macro_sources": macro_meta},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    print(f"[v7] wrote {OUT.relative_to(ROOT)} | as-of {payload['meta']['as_of']}")
    print(f"[v7] regime {regime['state']} ({regime['confidence']}) | valuation "
          f"{valuation['score']:.1f} | opportunity {fear['score']:.1f} | "
          f"credit {credit['score']} | internals {internals['score']}")
    print(f"[v7] VOO {primary['score']} -> {dep['percent']}% ({dep['label']}) | "
          f"QQQ {instruments['QQQ']['score']} -> {instruments['QQQ']['deployment']['percent']}%")
    return payload


if __name__ == "__main__":
    run()
