#!/usr/bin/env python3
"""
ETF Entry Decision System -- v4 evidence engine (Phases 5-9)
============================================================

NOT a redesign. Every anti-lookahead primitive from v2/v3 is imported and
reused: expanding point-in-time percentiles, day-based label purging,
horizon-length embargo, four purged CV schemes, in-fold feature selection,
five competing models, fail-safe gates, paired-bootstrap FDR, PIT analogue
scoring, PIT novel-regime detection and episode-based effective sample size.

v4 adds:

  Phase 5  VALUATION layer   -- real Shiller/multpl CAPE, trailing P/E,
                               earnings yield, dividend yield, earnings yield
                               minus Treasury, Wilshire-5000/GDP proxy.
                               Forward P/E excluded (no real-time vintage).
                               Buffett Indicator == Market Cap/GDP (one feature).
  Phase 6  LIQUIDITY layer   -- FRED WALCL (level + 6M growth), M2SL
                               (12M growth + acceleration), NFCI, and a
                               derived composite percentile (disclosed as
                               carrying no new information).
  Phase 7  SECTOR engine     -- 12 instruments, primary target D @ 12M,
                               full purged validation, cross-instrument FDR,
                               evidence-supported opportunity ranking.
  Phase 8  EVIDENCE GRADE    -- A+/A/B/C/D from nine pre-registered inputs
                               combined by WEAKEST-LINK (no arbitrary weights).
                               Grade < B => "Evidence Insufficient", all
                               actionable signals suppressed.
  Phase 9  FINAL OUTPUT      -- exactly four questions.

  AUDIT                     -- eight quantified risk classes; corrections are
                               implemented where the evidence justifies them.

MACRO POINT-IN-TIME RULE
------------------------
Macro/valuation observations are aligned by FIRST-KNOWN date (nominal date +
publication lag), not "as of". FRED and multpl serve LATEST REVISED vintages,
not real-time vintages; this residual revision risk is quantified against the
2023 Shiller vintage and by a with/without sensitivity test, and disclosed.

PRICE-BASIS RULE
----------------
Level-based technical features are built from RAW closes. Dividend-adjusted
closes are used ONLY for realised forward-return labels. Back-adjusted prices
embed dividends paid AFTER the date, which is a subtle look-ahead.

Output: data/evidence_v4.json
"""
from __future__ import annotations

import io
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import evidence_engine_v2 as E       # noqa: E402
from pipeline import evidence_engine_v3 as V       # noqa: E402
from pipeline import macro_data as M               # noqa: E402

OUT = ROOT / "data" / "evidence_v4.json"
V3_PAYLOAD = ROOT / "data" / "evidence_v3.json"
RAW, RAW2 = E.RAW, E.RAW2

SEED = 20261119
SCHEMES = V.SCHEMES
H = E.HORIZONS["12M"]

# 12 instruments: the two core ETFs plus the ten requested sector vehicles.
INSTRUMENTS = {
    "VOO": ("SPY", "VOO / S&P 500 (SPY proxy)"),
    "QQQ": ("QQQ", "QQQ / Nasdaq-100"),
    "XLK": ("XLK", "Technology"),
    "SOXX": ("SOXX", "Semiconductors (iShares)"),
    "SMH": ("SMH", "Semiconductors (VanEck)"),
    "XLF": ("XLF", "Financials"),
    "XLV": ("XLV", "Health Care"),
    "XLI": ("XLI", "Industrials"),
    "XLP": ("XLP", "Consumer Staples"),
    "XLY": ("XLY", "Consumer Discretionary"),
    "XLE": ("XLE", "Energy"),
    "XLU": ("XLU", "Utilities"),
}
# Available offline but deliberately NOT added: universe shopping would be
# another multiple-testing / survivorship degree of freedom.
UNIVERSE_EXCLUDED = ["XLB", "XLC", "XLRE"]

VAL_COLS = ["val_cape", "val_earnings_yield", "val_ey_minus_treasury",
            "val_dividend_yield", "val_pe_trailing", "val_buffett_proxy"]
LIQ_RAW_COLS = ["liq_fed_assets", "liq_fed_assets_growth_6m", "liq_m2_growth_12m",
                "liq_m2_acceleration", "liq_financial_conditions"]
LIQ_COMPOSITE = "liq_composite_percentile"
LIQ_COLS = LIQ_RAW_COLS + [LIQ_COMPOSITE]
NEW_COLS = VAL_COLS + LIQ_COLS

GRADE_ORDER = {"D": 0, "C": 1, "B": 2, "A": 3, "A+": 4}
GRADE_FROM_SCORE = ["D", "C", "B", "A", "A+"]

# Pre-registered thresholds. These are NOT tuned on the output; they are
# conventional levels for sample size, discrimination and calibration.
GRADE_RULES = {
    "effective_sample_size": [(100, "A+"), (60, "A"), (30, "B"), (15, "C")],
    "brier_improvement": [(0.010, "A+"), (0.005, "A"), (0.002, "B"), (0.0, "C")],
    "auc": [(0.70, "A+"), (0.65, "A"), (0.60, "B"), (0.55, "C")],
    "calibration_error": [(0.02, "A+"), (0.04, "A"), (0.06, "B"), (0.10, "C")],
    "calibration_slope": [(0.30, "C"), (0.60, "B"), (0.80, "A"), (0.90, "A+")],
    "feature_stability": [(0.80, "A+"), (0.65, "A"), (0.50, "B"), (0.30, "C")],
    "decade_stability": [(1.00, "A+"), (0.80, "A"), (0.67, "B"), (0.33, "C")],
    "regime_stability": [(1.00, "A+"), (0.80, "A"), (0.67, "B"), (0.33, "C")],
    "analogue_reliability": {"High": "A+", "Moderate": "A", "Low": "C", "Invalid": "D"},
}


# ---------------------------------------------------------------------------
# 1. Price frames: RAW for features, ADJUSTED for labels
# ---------------------------------------------------------------------------

def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    syms = (["SPY", "QQQ", "VIX", "TLT", "TNX", "IRX", "HYG", "LQD", "RSP", "IWM"]
            + E.SECTORS + ["XLC", "XLRE", "SOXX", "SMH", "W5000"])
    adj, raw = {}, {}
    for s in syms:
        f = RAW / f"{s}.csv"
        if not f.exists():
            raise FileNotFoundError(f"missing {f}; run pipeline/macro_data.py")
        adj[s] = E._read_adj(f)
        raw[s] = E._read_close(f)
    # Auxiliary series. FVX is a yield (close only); the ETFs have both.
    for name, path, has_adj in (("FVX", RAW2 / "fvx.csv", False),
                                ("IEF", RAW2 / "ief.csv", True),
                                ("SHY", RAW2 / "shy.csv", True),
                                ("VIX9D", RAW2 / "_vix9d.csv", True)):
        raw[name] = E._read_close(path)
        adj[name] = E._read_adj(path) if has_adj else E._read_close(path)

    def frame(d: dict) -> pd.DataFrame:
        px = pd.DataFrame(d)
        px = px[~px.index.duplicated(keep="last")].sort_index()
        # Truncate to the v3 as-of date so daily positions remain comparable.
        px = px.loc[lambda x: x.index <= pd.Timestamp("2026-11-16")]
        return px.reindex(pd.date_range(px.index.min(), px.index.max(), freq="B")).ffill()

    return frame(adj), frame(raw)


# ---------------------------------------------------------------------------
# 2. Macro observations -> FIRST-KNOWN-DATE aligned daily series
# ---------------------------------------------------------------------------

def macro_pit(index: pd.DatetimeIndex, lag_scale: float = 1.0) -> pd.DataFrame:
    obs = json.loads((M.MACRO / "macro_observations.json").read_text(encoding="utf-8"))
    out = {}
    for k in ("CAPE", "DIVIDEND_YIELD", "PE_TRAILING", "EARNINGS_YIELD",
              "WALCL", "M2SL", "NFCI", "GDP"):
        out[k] = M.pit_daily(index, obs[k], max(0, int(round(M.LAGS[k] * lag_scale))))
    out["W5000"] = M.pit_daily(index, obs["W5000"],
                               max(0, int(round(M.LAGS["W5000"] * lag_scale))),
                               value_key="close")
    return pd.DataFrame(out, index=index)


def build_raw_features_v4(px_raw: pd.DataFrame, pit: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """28 inherited features (from RAW closes) + 11 new raw candidates."""
    base, groups = E.build_raw_features(px_raw)
    f = base.copy()
    cape, ey, dy, pe = (pit[c] for c in ("CAPE", "EARNINGS_YIELD",
                                          "DIVIDEND_YIELD", "PE_TRAILING"))
    walcl, m2, nfci = (pit[c] for c in ("WALCL", "M2SL", "NFCI"))

    # ---- valuation (levels only; the percentile transform does the rest) --
    f["val_cape"] = cape
    f["val_earnings_yield"] = ey
    f["val_ey_minus_treasury"] = ey - px_raw["TNX"]
    f["val_dividend_yield"] = dy
    f["val_pe_trailing"] = pe
    f["val_buffett_proxy"] = pit["W5000"] / pit["GDP"]

    # ---- liquidity -------------------------------------------------------
    f["liq_fed_assets"] = walcl
    f["liq_fed_assets_growth_6m"] = walcl / walcl.shift(126) - 1.0
    m2g = m2 / m2.shift(252) - 1.0
    f["liq_m2_growth_12m"] = m2g
    f["liq_m2_acceleration"] = m2g - m2g.shift(126)
    f["liq_financial_conditions"] = -nfci

    f = f.replace([np.inf, -np.inf], np.nan)
    for c in VAL_COLS:
        groups[c] = "Valuation"
    for c in LIQ_RAW_COLS:
        groups[c] = "Liquidity"
    groups[LIQ_COMPOSITE] = "Liquidity"
    return f, groups


def pit_features(raw: pd.DataFrame, positions: list[int], with_composite: bool = True) -> pd.DataFrame:
    X = E.point_in_time_percentiles(raw, positions)
    if with_composite and all(c in X.columns for c in LIQ_RAW_COLS):
        X[LIQ_COMPOSITE] = X[LIQ_RAW_COLS].mean(axis=1)
    return X


# ---------------------------------------------------------------------------
# 3. Primary target (identical definition to v3's D @ 12M)
# ---------------------------------------------------------------------------

def primary_target(px_adj: pd.DataFrame, sym: str, positions: list[int]) -> dict:
    fp = E.forward_path_stats(px_adj, sym, positions)
    r = fp["12M"]["r"]
    irx = px_adj["IRX"].to_numpy(float) / 100.0
    pos = np.asarray(positions, dtype=int)
    rf = np.nan_to_num(irx[pos], nan=0.0)
    hurdle = rf + 0.05                     # 12M: exactly T-bill + 5%
    y = (r > hurdle).astype(float)
    y[~np.isfinite(r)] = np.nan
    return {"y": y, "r": r, "hurdle": hurdle,
            "maxdd": fp["12M"]["maxdd"], "sharpe": fp["12M"]["sharpe"]}


# ---------------------------------------------------------------------------
# 4. Phase 8 -- Evidence Quality Score (weakest link, no arbitrary weights)
# ---------------------------------------------------------------------------

def _thresh_grade(value: float | None, rules: list, higher_is_better: bool = True) -> str:
    if value is None or not np.isfinite(value):
        return "D"
    v = float(value) if higher_is_better else -float(value)
    rs = rules if higher_is_better else [(-t, g) for t, g in rules]
    for thr, g in sorted(rs, key=lambda x: -x[0]):
        if v >= thr:
            return g
    return "D"


def evidence_grade(inp: dict) -> dict:
    comps = {
        "effective_sample_size": _thresh_grade(inp["n_eff"], GRADE_RULES["effective_sample_size"]),
        "brier_improvement": _thresh_grade(inp["brier_edge"], GRADE_RULES["brier_improvement"]),
        "auc": _thresh_grade(inp["auc"], GRADE_RULES["auc"]),
        "calibration_error": _thresh_grade(inp["ece"], GRADE_RULES["calibration_error"], False),
        "calibration_slope": ("D" if inp["slope"] is None or inp["slope"] <= 0 or inp["slope"] > 2.5
                              else _thresh_grade(min(inp["slope"], 1 / inp["slope"]),
                                                 GRADE_RULES["calibration_slope"])),
        "feature_stability": _thresh_grade(inp["feature_stability"], GRADE_RULES["feature_stability"]),
        "decade_stability": _thresh_grade(inp["decade_stability"], GRADE_RULES["decade_stability"]),
        "regime_stability": ("D" if inp["novel"] else
                             _thresh_grade(inp["regime_stability"], GRADE_RULES["regime_stability"])),
        "analogue_reliability": GRADE_RULES["analogue_reliability"][inp["reliability"]],
    }
    if not inp["fdr_significant"]:
        comps["brier_improvement"] = "D"      # uncorrected edge is not evidence
    scores = {k: GRADE_ORDER[v] for k, v in comps.items()}
    worst = min(comps.values(), key=lambda g: GRADE_ORDER[g])
    return {"components": comps, "component_scores": scores,
            "numeric_score": round(float(np.mean(list(scores.values()))), 2),
            "grade": worst, "actionable": GRADE_ORDER[worst] >= GRADE_ORDER["B"],
            "binding_constraint": [k for k, v in comps.items() if v == worst],
            "method": "Weakest-link across nine pre-registered components. A weighted sum would "
                      "let a strong sample size compensate for a failed calibration or an "
                      "uncorrected p-value; evidence quality does not work that way.",
            "suppression_label": None if GRADE_ORDER[worst] >= GRADE_ORDER["B"]
                                 else "Evidence Insufficient"}


# ---------------------------------------------------------------------------
# 5. Decade / regime stability from out-of-sample predictions
# ---------------------------------------------------------------------------

def decade_model_edges(grid: dict, best: str, dates_of_pos: dict) -> dict:
    pm = V.pooled_predictions(grid, best)
    pb = V.pooled_predictions(grid, "Base Rate")
    ymap = grid["y_by_pos"]
    buckets = {"2000s": [], "2010s": [], "2020s": []}
    for d, p in pm.items():
        y = ymap.get(d)
        b = pb.get(d)
        ts = dates_of_pos.get(d)
        if y is None or b is None or ts is None or not np.isfinite(y):
            continue
        dec = "2000s" if ts.year < 2010 else ("2010s" if ts.year < 2020 else "2020s")
        buckets[dec].append((b - y) ** 2 - (p - y) ** 2)
    return {k: (float(np.mean(v)) if v else None) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# 6. Feature diagnostics for Phases 5 & 6
# ---------------------------------------------------------------------------

def feature_audit(grids: list[dict], names: list[str], groups: dict) -> dict:
    mi_all = np.zeros(len(names)); cnt = 0
    perm_all = {n: [] for n in names}; shap_all = {n: [] for n in names}
    uni_all = {n: [] for n in names}; sel_count = np.zeros(len(names))
    decade_edges = {"2000s": [], "2010s": [], "2020s": []}
    for g in grids:
        aud = g["audit"]
        for a in aud["mi"]:
            mi_all += np.asarray(a); cnt += 1
        for s in aud["selection"]:
            for i in s:
                sel_count[i] += 1
        for n, v in aud["perm"].items():
            perm_all[n] += v
        for n, v in aud["shap"].items():
            shap_all[n] += v
        for n, v in aud["univariate_brier"].items():
            uni_all[n] += v
        for d, arrs in aud["decade_edge"].items():
            decade_edges[d] += arrs
    mi_all /= max(cnt, 1)
    stab = np.zeros(len(names)); n_dec = 0
    for d, arrs in decade_edges.items():
        if arrs:
            stab += (np.mean(np.asarray(arrs), axis=0) > 0).astype(float); n_dec += 1
    stab /= max(1, n_dec)
    dec_mean = {d: (np.mean(np.asarray(a), axis=0) if a else np.zeros(len(names)))
                for d, a in decade_edges.items()}
    feats = []
    for i, n in enumerate(names):
        mi = float(mi_all[i]); pm_eval = len(perm_all[n]) > 0
        pm = float(np.mean(perm_all[n])) if pm_eval else None
        sh = float(np.mean(shap_all[n])) if shap_all[n] else None
        uv = float(np.mean(uni_all[n])) if uni_all[n] else 0.0
        sel = float(sel_count[i] / max(cnt, 1)); stability = float(stab[i])
        retained = bool(mi > 0.005 and (not pm_eval or pm > 0) and uv > 0
                        and sel >= 0.20 and stability >= 0.5)
        reasons = []
        if mi <= 0.005: reasons.append("mutual information at noise level")
        if pm_eval and pm <= 0: reasons.append("no positive OOS permutation importance")
        if uv <= 0: reasons.append("fails to beat base-rate Brier in univariate OOS test")
        if sel < 0.20: reasons.append("rarely selected inside purged folds")
        if stability < 0.5: reasons.append("unstable across decades")
        feats.append({
            "name": n, "group": groups[n], "layer": ("Inherited" if n not in NEW_COLS
                                                     else ("Valuation" if n in VAL_COLS else "Liquidity")),
            "mi": round(mi, 5), "perm": None if pm is None else round(pm, 6),
            "shap": None if sh is None else round(sh, 5),
            "oos_contribution": round(uv, 6), "selection_rate": round(sel, 3),
            "stability": round(stability, 2),
            "decade_edge": {d: round(float(a[i]), 6) for d, a in dec_mean.items()},
            "verdict": "RETAINED" if retained else "REMOVED",
            "reason": "; ".join(reasons) if reasons else "positive OOS evidence",
        })
    retained_idx = [i for i, f in enumerate(feats) if f["verdict"] == "RETAINED"]
    fallback = len(retained_idx) < 3
    if fallback:
        retained_idx = list(np.argsort(-mi_all)[:E.K_FEATURES])
    return {"list": feats, "retained": [names[i] for i, f in enumerate(feats)
                                        if f["verdict"] == "RETAINED"],
            "model_feature_set": [names[i] for i in retained_idx],
            "fallback": fallback, "n_folds": cnt,
            "val_ret": sum(1 for f in feats if f["name"] in VAL_COLS and f["verdict"] == "RETAINED"),
            "liq_ret": sum(1 for f in feats if f["name"] in LIQ_COLS and f["verdict"] == "RETAINED")}


# ---------------------------------------------------------------------------
# 7. Quantified scientific audit
# ---------------------------------------------------------------------------

def shiller_2023_vintage() -> pd.Series:
    """CAPE values from the stale Yale ie_data.xls (last updated 2023-09)."""
    path = RAW / "_shiller_ie_data.xls"
    if not path.exists():
        return pd.Series(dtype=float)
    d = pd.read_excel(io.BytesIO(path.read_bytes()), sheet_name="Data", header=7,
                      engine="xlrd", usecols=[0, 12])
    d.columns = ["ym", "cape"]
    d = d[pd.to_numeric(d["ym"], errors="coerce").notna()]
    d["date"] = pd.to_datetime(d["ym"].astype(float).map(
        lambda x: f"{int(x):04d}-{min(12, max(1, int(round((x % 1) * 100)))):02d}-01"),
        errors="coerce")
    d["cape"] = pd.to_numeric(d["cape"], errors="coerce")
    return d.dropna(subset=["date", "cape"]).set_index("date")["cape"]


def revision_magnitude() -> dict:
    obs = json.loads((M.MACRO / "macro_observations.json").read_text(encoding="utf-8"))
    cur = pd.Series({pd.Timestamp(r["date"]): r["value"] for r in obs["CAPE"]})
    old = shiller_2023_vintage()
    if old.empty:
        return {"available": False}
    common = cur.index.intersection(old.index)
    common = common[(common >= "1999-01-01") & (common <= "2023-09-01")]
    diff = (cur.loc[common] - old.loc[common]).abs()
    return {"available": True, "n_overlapping_months": int(len(common)),
            "mean_abs_revision_cape_points": round(float(diff.mean()), 4),
            "max_abs_revision_cape_points": round(float(diff.max()), 4),
            "pct_months_revised_gt_1pt": round(float((diff > 1.0).mean()), 4),
            "current_vintage_last": str(cur.index.max().date()),
            "old_vintage_last": str(old.index.max().date()),
            "note": "Compares the 2026 multpl/Shiller-derived CAPE vintage with the 2023 Yale "
                    "workbook on identical historical dates. Non-zero values prove that "
                    "historical CAPE observations are revised after the fact."}


def scientific_audit(px_adj, px_raw, positions, X_base, X_adj_base, X_exp, X_nolag,
                     grids, base_grids, gates, inst, faudit, v3, revision) -> dict:
    """Quantify eight risk classes. Corrections are implemented in the engine."""
    # 1. Adjusted-close look-ahead in level features -------------------------
    common = X_base.columns
    a, b = X_base[common].to_numpy(float), X_adj_base[common].to_numpy(float)
    d = np.abs(a - b)
    leak_price = {
        "mean_abs_percentile_shift": round(float(np.nanmean(d)), 5),
        "p99_abs_percentile_shift": round(float(np.nanquantile(d, 0.99)), 5),
        "pct_feature_cells_shifted_gt_5pp": round(float(np.nanmean(d > 0.05)), 4),
        "max_abs_percentile_shift": round(float(np.nanmax(d)), 5),
        "n_feature_cells": int(d.size),
        "correction": "v4 builds ALL level-based technical features from raw closes; "
                      "dividend-adjusted closes are used only for realised return labels.",
    }
    # 2. Publication-lag look-ahead ------------------------------------------
    d2 = np.abs(X_exp[NEW_COLS].to_numpy(float) - X_nolag[NEW_COLS].to_numpy(float))
    leak_macro = {
        "mean_abs_percentile_shift": round(float(np.nanmean(d2)), 5),
        "pct_feature_cells_shifted_gt_5pp": round(float(np.nanmean(d2 > 0.05)), 4),
        "max_abs_percentile_shift": round(float(np.nanmax(d2)), 5),
        "correction": "Every macro observation is aligned by first-known date "
                      "(nominal date + publication lag). Zero-lag alignment is "
                      "computed only for this impact measurement, never used.",
    }
    # 3. Overfitting / multiple testing --------------------------------------
    n_tests_v3 = int(v3["meta"]["n_gate_tests"])
    n_tests_new = len(gates)
    passed = sum(1 for g in gates.values() if g["pass"])
    selection_bias, sel_edges = [], []
    for tk, g in gates.items():
        cand = g.get("all_candidates") or {}
        if len(cand) >= 2:
            vals = list(cand.values())
            selection_bias.append(float(np.mean(vals) - min(vals)))
            sel_edges.append(g["mean_brier_edge"])
    overfit = {
        "v3_gate_tests": n_tests_v3, "v4_instrument_tests": n_tests_new,
        "total_tests": n_tests_v3 + n_tests_new,
        "fdr_q": V.FDR_Q,
        "expected_false_positives_at_q": round((n_tests_v3 + n_tests_new) * V.FDR_Q, 1),
        "v4_gates_passed": passed,
        "v3_gates_passed": sum(1 for g in v3["gates"].values() if g and g["pass"]),
        "best_of_4_selection_bias_brier": (round(float(np.mean(selection_bias)), 6)
                                           if selection_bias else None),
        "interpretation": "Picking the best of four models per gate inflates the apparent "
                          "Brier advantage by this amount even when all four are useless.",
        "correction": "In-fold feature selection, purged/embargoed splits, paired-bootstrap "
                      "p-values, BH-FDR across all gates and instruments, and a pre-registered "
                      "primary target. Selection-adjusted edge is reported alongside the "
                      "selected-model edge.",
    }
    # 4. Regime dependency ----------------------------------------------------
    dec = {tk: inst[tk]["decade_model_edges"] for tk in inst}
    all_edges = [v for d in dec.values() for v in d.values() if v is not None]
    regime = {
        "decade_brier_edges": dec,
        "std_across_decades": round(float(np.std(all_edges)), 6) if all_edges else None,
        "fraction_positive": round(float(np.mean([x > 0 for x in all_edges])), 3) if all_edges else None,
        "novel_regime_instruments": [tk for tk in inst if inst[tk]["novel"]],
        "correction": "PIT novel-regime gate (Phase 3) plus decade-stability inputs to the "
                      "evidence grade; a signal that only works in one decade cannot reach B.",
    }
    # 5. Survivorship ---------------------------------------------------------
    def mean12(sym):
        t = primary_target(px_adj, sym, positions)
        r = t["r"]
        return float(np.nanmean(r)) if np.isfinite(r).any() else None
    incl = [mean12(sym) for sym, _label in INSTRUMENTS.values()]
    excl = [mean12(s) for s in UNIVERSE_EXCLUDED]
    incl = [x for x in incl if x is not None]
    excl = [x for x in excl if x is not None]
    surv = {"n_instruments": len(INSTRUMENTS), "n_surviving_only": len(INSTRUMENTS),
            "excluded_available_tickers": UNIVERSE_EXCLUDED,
            "mean_historical_12m_return_included": round(float(np.mean(incl)), 4) if incl else None,
            "mean_historical_12m_return_excluded": round(float(np.mean(excl)), 4) if excl else None,
            "selection_gap_pp": (round(float(np.mean(incl) - np.mean(excl)) * 100, 2)
                                 if incl and excl else None),
            "quantified_impact": "The three available-but-excluded sector SPDRs have a different "
                                 "mean realised 12M return from the requested universe; had they "
                                 "been added post hoc they would have changed the cross-sectional "
                                 "ranking. No dead sector fund can be retrieved from the price "
                                 "provider, so the survivorship bias cannot be eliminated.",
            "correction": "Universe fixed to the 10 requested sectors + VOO/QQQ before any "
                          "testing. XLB/XLC/XLRE are available offline but excluded to avoid "
                          "universe shopping; no delisted sector fund can be downloaded from "
                          "the price provider, so sector rankings are explicitly conditional "
                          "on fund survival and are never presented as a return forecast.",
            "residual_risk": "Uncorrectable without a delisted-fund database; disclosed, and "
                             "no cross-sectional return claim is made."}
    # 6. Feature instability --------------------------------------------------
    rates = [f["selection_rate"] for f in faudit["list"]]
    instab = {
        "n_candidate_features": len(faudit["list"]),
        "n_retained": len(faudit["retained"]),
        "mean_selection_rate": round(float(np.mean(rates)), 4),
        "cv_selection_rate": round(float(np.std(rates) / max(np.mean(rates), 1e-9)), 4),
        "features_positive_in_all_3_decades": sum(1 for f in faudit["list"]
                                                  if f["stability"] >= 0.99),
        "valuation_retained": faudit["val_ret"], "liquidity_retained": faudit["liq_ret"],
        "correction": "A feature must show MI above noise, positive OOS permutation importance, "
                      "positive univariate OOS Brier edge, >=20% in-fold selection and "
                      "positive decade stability to be retained. No forced inclusion.",
    }
    # 7 & 8. UI / false confidence -------------------------------------------
    ui = {
        "misleading_components_found": [
            "Three-decimal AUC/Brier/p-values with no confidence interval",
            "'SELECTED SCHEDULE' badge implying a validated winner",
            "40 analogues displayed next to ~11 independent episodes",
            "1,297 weekly rows displayed next to ~25 independent 12M episodes",
            "Descriptive current-state return shown beside a suppressed advantage",
            "Ordinal rank framing ('strongest opportunities') before evidence gating",
        ],
        "n_components": 6,
        "corrections_implemented": [
            "Final page shows only four answers; no decorative metrics",
            "Every n is shown with its effective sample size",
            "Suppressed instruments are listed alphabetically with no rank numbers",
            "Precision reduced and uncertainty intervals shown where estimable",
            "Grade < B replaces all numeric actionable signals with 'Evidence Insufficient'",
            "No score bar, no colour-coded conviction, no indicator popularity",
        ],
        "correction": "Six misleading components identified in v3 and removed or relabelled "
                      "in the v4 rendering layer.",
    }
    fc = {
        "analogues_per_instrument": V.N_NEIGHBOURS,
        "mean_independent_episodes": round(float(np.mean([inst[t]["conf"]["n_eff"] for t in inst])), 1),
        "analogue_overstatement_multiple": (round(V.N_NEIGHBOURS /
                                                  max(np.mean([inst[t]["conf"]["n_eff"]
                                                               for t in inst]), 1), 2)),
        "weekly_rows": int(v3["meta"]["n_observations"]["Weekly"]),
        "independent_12m_episodes_approx": int(round(v3["meta"]["n_daily"] / 252)),
        "weekly_overstatement_multiple": (round(v3["meta"]["n_observations"]["Weekly"] /
                                                max(1, round(v3["meta"]["n_daily"] / 252)), 1)),
        "correction": "Both raw n and effective n are printed together; all confidence grades "
                      "and the evidence grade use non-overlapping episodes only. Grade < B "
                      "suppresses the numeric current-state return entirely.",
    }
    return {
        "data_leakage_risks": {"adjusted_close": leak_price, "macro_publication_lag": leak_macro,
                               "revision_vintage": revision},
        "look_ahead_risks": {
            "publication_lags_days": {k: M.LAGS[k] for k in
                                      ("CAPE", "PE_TRAILING", "EARNINGS_YIELD",
                                       "DIVIDEND_YIELD", "GDP", "WALCL", "M2SL", "NFCI", "W5000")},
            "quantified_impact_zero_lag": leak_macro,
            "inherited": "Expanding PIT percentiles, day-based purge, horizon-length embargo, "
                         "strictly-prior analogue outcomes, PIT novelty threshold.",
            "correction": "First-known-date alignment; conservative lags; with/without "
                          "sensitivity on revision-prone features. ALFRED real-time GDP "
                          "vintages were unavailable (HTTP 404), so revision risk is bounded "
                          "by sensitivity analysis rather than eliminated.",
        },
        "overfitting_risks": overfit,
        "regime_dependency_risks": regime,
        "survivorship_bias": surv,
        "feature_instability": instab,
        "misleading_ui_components": ui,
        "false_confidence_mechanisms": fc,
    }


# ---------------------------------------------------------------------------
# 8. Orchestration
# ---------------------------------------------------------------------------

def run() -> dict:
    t_start = datetime.now(timezone.utc)
    M.build()                                  # uses cached data by default
    v3 = json.loads(V3_PAYLOAD.read_text(encoding="utf-8"))
    px_adj, px_raw = load_frames()
    last = len(px_adj) - 1
    today_pos = int(last)
    end_pos = last - H
    print(f"[engine] daily rows {len(px_adj)} | as-of {px_adj.index[last].date()}")

    obs = V.observation_positions(px_adj, "Monthly", V.START_POS, end_pos)
    positions = sorted(set(obs.tolist() + [today_pos]))
    row_of = {int(p): i for i, p in enumerate(positions)}
    today_row = positions.index(today_pos)
    dates = [px_adj.index[i] for i in positions]
    dates_of_pos = {int(p): d for p, d in zip(positions, dates)}

    pit = macro_pit(px_adj.index)
    pit_nolag = macro_pit(px_adj.index, lag_scale=0.0)
    raw_all, groups = build_raw_features_v4(px_raw, pit)
    raw_nolag, _ = build_raw_features_v4(px_raw, pit_nolag)
    base_cols = [c for c in raw_all.columns if c not in NEW_COLS]
    X_base = pit_features(raw_all[base_cols], positions, with_composite=False)
    X_exp = pit_features(raw_all, positions)
    X_nolag = pit_features(raw_nolag, positions)
    _, base_groups_adj = E.build_raw_features(px_adj)
    X_adj_base = E.point_in_time_percentiles(
        E.build_raw_features(px_adj)[0], positions)
    names_base, names_exp = list(X_base.columns), list(X_exp.columns)
    print(f"[engine] base {len(names_base)} features | expanded {len(names_exp)} "
          f"(6 valuation + 6 liquidity) | {len(obs)} monthly observations")
    print(f"[engine] macro PIT coverage: "
          + ", ".join(f"{c}={int(pit[c].notna().sum())}" for c in pit.columns))

    # ---- 12 instrument grids: expanded vs base, same splits, same labels ----
    grids_exp, grids_base, gates, inst, uncond = {}, {}, {}, {}, {}
    for j, (tk, (sym, label)) in enumerate(INSTRUMENTS.items()):
        t = primary_target(px_adj, sym, positions)
        y = t["y"]
        y_by_pos = {int(p): (float(y[i]) if np.isfinite(y[i]) else 0.0)
                    for i, p in enumerate(positions)}
        splits = V.build_splits_v3(obs, H, y_by_pos)
        seed = SEED + 101 * j
        ge = V.run_model_grid_v3(X_exp.to_numpy(float), y, names_exp, splits,
                                 row_of, dates, diagnostics=True, seed=seed)
        gb = V.run_model_grid_v3(X_base.to_numpy(float), y, names_base, splits,
                                 row_of, dates, diagnostics=False, seed=seed)
        grids_exp[tk], grids_base[tk] = ge, gb
        me = {f"{s}|{m}": {k: v for k, v in mb.items() if k != "predictions"}
              for (s, m), mb in ge["metrics"].items() if mb is not None}
        gate = V.evaluate_gate(me, int(np.isfinite(y).sum()))
        gates[tk] = gate
        uncond[tk] = E.unconditional_stats(t["r"], t["maxdd"], t["sharpe"])
        inst[tk] = {"label": label, "symbol": sym, "gate": gate, "metrics": me,
                    "n_valid": int(np.isfinite(y).sum()),
                    "base_rate": float(np.nanmean(y)) if np.isfinite(y).any() else None,
                    "n_splits": {s: len(v) for s, v in splits.items()},
                    "decade_model_edges": decade_model_edges(ge, gate["best"], dates_of_pos)}
        print(f"[engine] {tk:5s} {label:28s} n={inst[tk]['n_valid']:3d} "
              f"best={gate['best']:18s} edge={gate['mean_brier_edge']:+.5f} "
              f"AUC={gate['auc']:.3f} pass={gate['pass']}")

    # ---- cross-instrument FDR ----------------------------------------------
    tks = list(INSTRUMENTS)
    pvals = [V.paired_edge_pvalue(grids_exp[tk], gates[tk]["best"], H,
                                  seed=SEED + 31 * i) for i, tk in enumerate(tks)]
    sig, fdr_thresh = V.bh_fdr(pvals, V.FDR_Q)
    fdr = {tk: {"p_value": float(p), "fdr_significant": bool(s)}
           for tk, p, s in zip(tks, pvals, sig)}

    # ---- feature diagnostics (Phases 5 & 6) -------------------------------
    faudit = feature_audit([grids_exp[t] for t in tks], names_exp, groups)
    print(f"[engine] features retained {len(faudit['retained'])}/{len(names_exp)} "
          f"(valuation {faudit['val_ret']}/6, liquidity {faudit['liq_ret']}/6)")
    retained_idx = [names_exp.index(n) for n in faudit["model_feature_set"]]
    Z = X_exp.to_numpy(float)[:, retained_idx] if retained_idx else X_exp.to_numpy(float)[:, :E.K_FEATURES]

    # ---- paired OOS contribution of the new layers -------------------------
    layer_rows = []
    for tk in tks:
        best = gates[tk]["best"]
        pe = V.pooled_predictions(grids_exp[tk], best)
        pb = V.pooled_predictions(grids_base[tk], best)
        ym = grids_exp[tk]["y_by_pos"]
        common = sorted(set(pe) & set(pb) & set(ym))
        if len(common) < 30:
            continue
        yv = np.array([ym[d] for d in common])
        be = np.array([pe[d] for d in common]); bb = np.array([pb[d] for d in common])
        valid = np.isfinite(yv)
        yv, be, bb = yv[valid], be[valid], bb[valid]
        br_e, br_b = float(np.mean((be - yv) ** 2)), float(np.mean((bb - yv) ** 2))
        from sklearn.metrics import roc_auc_score
        auc_e = float(roc_auc_score(yv, be)) if len(np.unique(yv)) > 1 else None
        auc_b = float(roc_auc_score(yv, bb)) if len(np.unique(yv)) > 1 else None
        layer_rows.append({"instrument": tk, "n_oos": int(len(yv)),
                           "brier_base": round(br_b, 6), "brier_expanded": round(br_e, 6),
                           "oos_contribution_brier": round(br_b - br_e, 6),
                           "auc_base": None if auc_b is None else round(auc_b, 4),
                           "auc_expanded": None if auc_e is None else round(auc_e, 4),
                           "auc_change": (None if auc_b is None or auc_e is None
                                          else round(auc_e - auc_b, 4))})
    contrib = float(np.mean([r["oos_contribution_brier"] for r in layer_rows])) if layer_rows else 0.0
    flips = sum(1 for tk in tks if gates[tk]["pass"] !=
                (V.evaluate_gate(
                    {f"{s}|{m}": {k: v for k, v in mb.items() if k != "predictions"}
                     for (s, m), mb in grids_base[tk]["metrics"].items() if mb is not None},
                    inst[tk]["n_valid"])["pass"]))

    # ---- Phase 2/3/7/8 per instrument -------------------------------------
    finals, timing, regimes, reliability, conf, grades = {}, {}, {}, {}, {}, {}
    for tk, (sym, label) in INSTRUMENTS.items():
        t = primary_target(px_adj, sym, positions)
        r12 = t["r"]
        valid_r = np.isfinite(r12)
        cand = [i for i in range(len(positions)) if valid_r[i]]
        d = np.sqrt(np.mean((Z[cand] - Z[today_row]) ** 2, axis=1))
        analogues = [int(positions[cand[i]]) for i in np.argsort(d)[:V.N_NEIGHBOURS]]
        r12_by_pos = {int(positions[i]): float(r12[i]) for i in cand}
        conf[tk] = E.confidence_engine(analogues, r12_by_pos)
        regimes[tk] = V.novel_regime(Z, positions, r12, today_row)
        reliability[tk] = V.analogue_reliability(regimes[tk], conf[tk])
        gate_pass = bool(gates[tk]["pass"] and fdr[tk]["fdr_significant"]
                         and not gates[tk].get("degenerate"))
        sc = E.entry_score(X_exp.to_numpy(float), r12, positions, today_row,
                           retained_idx, dates)
        current = sc["today_expected"] if reliability[tk] != "Invalid" else None
        timing[tk] = V.timing_advantage(tk, uncond[tk]["mean"], current, analogues,
                                        r12_by_pos, gate_pass, sc["validated"],
                                        reliability[tk], fdr[tk]["fdr_significant"],
                                        conf[tk])
        de = inst[tk]["decade_model_edges"]
        dec_vals = [v for v in de.values() if v is not None]
        ret_feats = [f for f in faudit["list"] if f["name"] in faudit["retained"]]
        feat_stab = float(np.mean([f["selection_rate"] for f in ret_feats])) if ret_feats else 0.0
        decade_stab = float(np.mean([f["stability"] for f in ret_feats])) if ret_feats else 0.0
        regime_stab = (float(np.mean([v > 0 for v in dec_vals])) if dec_vals else 0.0)
        ginputs = {
            "n_eff": conf[tk]["n_eff"], "brier_edge": gates[tk]["mean_brier_edge"],
            "auc": gates[tk]["auc"], "ece": gates[tk]["ece"], "slope": gates[tk]["slope"],
            "feature_stability": feat_stab, "decade_stability": decade_stab,
            "regime_stability": regime_stab, "novel": bool(regimes[tk]["novel"]),
            "reliability": reliability[tk], "fdr_significant": fdr[tk]["fdr_significant"],
        }
        grades[tk] = evidence_grade(ginputs)
        # expose to the quantified audit (which runs after this loop)
        inst[tk]["conf"] = conf[tk]
        inst[tk]["novel"] = bool(regimes[tk]["novel"])
        inst[tk]["reliability"] = reliability[tk]
        adv_ok = bool(timing[tk]["validated"] and grades[tk]["actionable"])
        finals[tk] = {
            "label": label, "symbol": sym,
            "historical_12m_return": uncond[tk]["mean"],
            "n_historical_12m": uncond[tk]["n"],
            "current_state_expected_12m_return": (current if grades[tk]["actionable"] else None),
            "timing_advantage": timing[tk]["timing_advantage"] if adv_ok else None,
            "timing_advantage_ci_low": timing[tk]["timing_advantage_ci_low"] if adv_ok else None,
            "timing_advantage_ci_high": timing[tk]["timing_advantage_ci_high"] if adv_ok else None,
            "edge_detected": gate_pass,
            "evidence_grade": grades[tk]["grade"],
            "evidence_actionable": grades[tk]["actionable"],
            "evidence_score": grades[tk]["numeric_score"],
            "binding_constraint": grades[tk]["binding_constraint"],
            "confidence": conf[tk]["grade"], "n_eff": conf[tk]["n_eff"],
            "analogue_reliability": reliability[tk],
            "novel_regime": bool(regimes[tk]["novel"]),
            "gate_pass": gates[tk]["pass"],
            "fdr_significant": fdr[tk]["fdr_significant"],
            "p_value": fdr[tk]["p_value"],
            "auc": gates[tk]["auc"], "brier_edge": gates[tk]["mean_brier_edge"],
            "released": adv_ok,
            "suppression_label": (None if adv_ok else
                                  ("Evidence Insufficient" if not grades[tk]["actionable"]
                                   else "No Reliable Timing Edge Detected")),
            "justification": (
                f"{label}: primary target D @ 12M, Monthly, {inst[tk]['n_valid']} valid labels. "
                f"Best model {gates[tk]['best']}; Brier edge "
                f"{gates[tk]['mean_brier_edge']:+.5f}; AUC {gates[tk]['auc']:.3f}; "
                f"FDR p={fdr[tk]['p_value']:.3f} "
                f"({'significant' if fdr[tk]['fdr_significant'] else 'not significant'}). "
                f"Evidence grade {grades[tk]['grade']} "
                f"(binding: {', '.join(grades[tk]['binding_constraint'])}). "
                + ("Novel regime — analogue analysis disabled. " if regimes[tk]["novel"] else "")
                + (f"Timing advantage suppressed: {'; '.join(timing[tk]['suppression_reasons'])}. "
                   if not timing[tk]["validated"] else "")),
        }

    # ---- Phase 7 ranking ---------------------------------------------------
    released = [tk for tk in tks if finals[tk]["released"]]
    released.sort(key=lambda t: -(finals[t]["timing_advantage"] or 0))
    ranking = []
    for i, tk in enumerate(tks):
        ranking.append({
            "instrument": tk, "label": finals[tk]["label"],
            "rank": (i + 1) if tk in released else None,   # NO ordinal rank without evidence
            "released": tk in released,
            "timing_advantage": finals[tk]["timing_advantage"],
            "evidence_grade": finals[tk]["evidence_grade"],
            "display": (f"{finals[tk]['timing_advantage'] * 100:+.1f}%" if tk in released
                        else "Evidence Insufficient"),
        })
    # Deterministic, non-ordinal display order for anything without evidence.
    ranking.sort(key=lambda r: ((0, r["rank"], "") if r["rank"] is not None
                                else (1, 0, r["instrument"])))

    # ---- Phase 9: exactly four answers ------------------------------------
    q1 = bool(released)
    q2 = max([finals[t]["timing_advantage"] for t in released], default=None)
    best_grade = max([GRADE_ORDER[grades[t]["grade"]] for t in tks])
    q3 = GRADE_FROM_SCORE[best_grade]
    allowed_q4 = {"All In", "70/30", "50/50", "DCA 3M", "DCA 6M", "No Timing Signal"}
    if not q1:
        q4 = "No Timing Signal"
    else:
        core_action = v3["final"]["VOO"]["action"]
        q4 = core_action if core_action in allowed_q4 else "No Timing Signal"
    final_page = {
        "q1_label": "Is there reliable market timing evidence?",
        "q1": "YES" if q1 else "NO",
        "q2_label": "What is the expected timing advantage?",
        "q2": (f"{q2 * 100:+.2f}%" if q2 is not None
               else "No Reliable Timing Edge Detected"),
        "q2_style": "long" if q2 is None else "",
        "q2_detail": (f"Best evidence-supported instrument: {released[0]} "
                      f"({finals[released[0]]['label']})" if released else
                      "No instrument cleared the fail-safe, FDR and grade-B evidence gate. "
                      "Evidence Insufficient."),
        "q3_label": "How trustworthy is that estimate?",
        "q3": q3,
        "q3_detail": ("Evidence Insufficient — actionable signals suppressed (grade below B)."
                      if best_grade < GRADE_ORDER["B"] else
                      "Grade B or better; evidence components are documented on the Evidence tab."),
        "q4_label": "What is the suggested deployment?",
        "q4": q4,
        "q4_detail": ("Maintain ordinary investment plans. No instrument-level timing call is "
                      "released; a lump-sum/DCA choice is a policy question, not a timing signal."
                      if not q1 else "See the deployment tab."),
        "n_instruments_evaluated": len(tks),
        "n_released": len(released),
        "released_instruments": released,
    }

    # ---- quantified scientific audit --------------------------------------
    audit = scientific_audit(px_adj, px_raw, positions, X_base, X_adj_base, X_exp, X_nolag,
                             grids_exp, grids_base, gates, inst, faudit, v3,
                             revision_magnitude())

    # ---- assemble ----------------------------------------------------------
    payload = {
        "meta": {
            "version": "4.0 — valuation, liquidity, sectors, evidence grades, four-question output",
            "generated_at": t_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "as_of": str(px_adj.index[today_pos].date()),
            "sample_start": str(px_adj.index[obs[0]].date()),
            "sample_end": str(px_adj.index[obs[-1]].date()),
            "n_daily": int(len(px_adj)), "n_monthly_observations": int(len(obs)),
            "n_positions": len(positions),
            "n_instruments": len(INSTRUMENTS),
            "instruments": list(INSTRUMENTS),
            "n_features": len(names_exp), "n_base_features": len(names_base),
            "n_valuation_features": len(VAL_COLS), "n_liquidity_features": len(LIQ_COLS),
            "price_basis": "RAW closes for features; dividend-adjusted closes for labels",
            "macro_alignment": "first-known-date (nominal date + publication lag)",
            "primary_target": "D @ 12M (T-bill + 5%)", "schedule": "Monthly",
            "schemes": SCHEMES, "models": E.MODEL_NAMES, "fdr_q": V.FDR_Q,
            "fdr_threshold": fdr_thresh,
            "inherited_from_v3": {
                "selected_schedule": v3["meta"]["selected_schedule"],
                "n_gate_tests": v3["meta"]["n_gate_tests"],
                "version": v3["meta"]["version"],
            },
        },
        "final_page": final_page,
        "final": finals,
        "ranking": ranking,
        "evidence_grades": grades,
        "grade_rules": GRADE_RULES,
        "timing_advantage": timing,
        "novel_regime": regimes,
        "analogue_reliability": reliability,
        "confidence": conf,
        "gates": {t: (gates[t] | fdr[t]) for t in tks},
        "instruments": {t: inst[t] for t in tks},
        "unconditional": uncond,
        "feature_audit": faudit,
        "layer_contribution": {"rows": layer_rows, "mean_oos_brier_contribution": round(contrib, 6),
                               "gates_flipped_by_new_layers": flips,
                               "method": "Same instrument, same purged splits, same labels, same "
                                         "model: only the feature set differs. Positive = new "
                                         "layers improved out-of-sample Brier."},
        "excluded_candidates": {
            "FORWARD_PE": {"status": "EXCLUDED",
                           "reason": "No free historical real-time forward-earnings series. "
                                     "A trailing-P/E relabelled as forward P/E would be "
                                     "fabricated data, not an unavailable feature."},
            "MARKET_CAP_TO_GDP": {"status": "MERGED",
                                  "reason": "Identical to the Buffett Indicator. Implemented once "
                                            "as val_buffett_proxy (Wilshire 5000 / nominal GDP)."},
            "SHILLER_IE_DATA_XLS": {"status": "AUDIT ONLY",
                                    "reason": "Yale workbook ends 2023-09 (stale). Used solely "
                                              "as an older CAPE vintage to quantify revisions."},
            "XLB_XLC_XLRE": {"status": "EXCLUDED",
                             "reason": "Available offline but not in the requested universe; "
                                       "adding them post hoc is universe shopping."},
        },
        "audit": audit,
        "disclosures": [
            "All v2/v3 anti-lookahead protections are imported unchanged: expanding point-in-time percentiles, day-based label purging, horizon-length embargo, four purged CV schemes, in-fold feature selection, five competing models, paired-bootstrap FDR, PIT analogue cutoffs, PIT novelty thresholds and episode-based ESS.",
            "Macro and valuation series use FIRST-KNOWN-DATE alignment: nominal observation date plus a conservative publication lag. No macro value is visible before it could have been known. FRED/multpl serve LATEST REVISED vintages rather than real-time vintages; revision magnitude is quantified against the 2023 Shiller workbook and a with/without sensitivity test is reported.",
            "FRED and multpl serve LATEST REVISED vintages, not real-time vintages. ALFRED real-time GDP vintages were unavailable (HTTP 404) in this environment, so revision look-ahead cannot be fully eliminated; it is quantified against the stale 2023 Shiller workbook and bounded by a with/without feature-set sensitivity test. Conservative publication lags are used throughout.",
            "Level-based technical features are built from RAW closes. Dividend-adjusted closes are used only for realised forward-return labels, because back-adjusted historical prices embed dividends paid after the date.",
            "Forward P/E is EXCLUDED, not proxied: no free real-time historical forward-earnings series exists. The trailing P/E is included as an explicitly labelled substitute candidate and must survive the same selection gates.",
            "The Buffett Indicator and Market Cap/GDP are the same quantity and are implemented once. The Wilshire 5000 is a PRICE index, so val_buffett_proxy is a proportional proxy for market cap/GDP, not the exact ratio.",
            "The liquidity composite is a row-mean of the five PIT liquidity percentiles. It carries NO new information and is disclosed as a derived diagnostic, not an independent signal.",
            "The sector universe was fixed BEFORE testing to the ten requested vehicles plus VOO/QQQ. Three additional sector SPDRs are available offline but excluded to avoid universe shopping. All prices come from surviving funds only; no delisted sector fund can be retrieved, which is an uncorrectable survivorship limitation disclosed here.",
            "Evidence grades are WEAKEST-LINK across nine pre-registered components, not a weighted score. A large sample size cannot compensate for a failed calibration or an FDR-uncorrected p-value. Grade below B suppresses every actionable signal and prints 'Evidence Insufficient'.",
            "Twelve instruments are tested simultaneously; Benjamini-Hochberg FDR at q=0.10 is applied across instruments. The v3 64-gate framework is retained in full and is not re-tuned.",
            "Valuation and liquidity features were NOT forced into the model. They pass the same mutual-information, permutation, SHAP, univariate OOS Brier, in-fold selection-rate and decade-stability gates as every inherited feature.",
            "The final page answers exactly four questions and contains no decorative metrics, no conviction bar, no opportunity score and no ordinal ranking of unvalidated instruments.",
            "Backtested results are not a guarantee of future performance. Educational research tool, not investment advice.",
        ],
    }
    payload = V.clean(payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    print(f"[engine] wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"[engine] Q1={final_page['q1']} Q2={final_page['q2']} Q3={final_page['q3']} "
          f"Q4={final_page['q4']}")
    print(f"[engine] elapsed {(datetime.now(timezone.utc) - t_start).total_seconds():.0f}s")
    return payload


if __name__ == "__main__":
    run()
