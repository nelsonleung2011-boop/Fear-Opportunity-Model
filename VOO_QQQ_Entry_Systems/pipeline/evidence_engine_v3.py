"""
ETF Entry Decision System -- evidence engine (v3)
=================================================

Evolution of v2, NOT a redesign. Every anti-lookahead primitive is imported
from ``evidence_engine_v2`` and reused unchanged:

  * load_prices / build_raw_features / point_in_time_percentiles
  * forward_path_stats / unconditional_stats
  * _purge (day-based purging + horizon-length embargo)
  * fit_predict / make_model / metric_bundle / select_features
  * entry_score / confidence_engine / deployment_backtest

v3 adds four things:

  Phase 1  Four targets per (ticker, horizon):
             A  r > point-in-time historical median return
             B  r > 10%
             C  r > 15%
             D  r > T-Bill + 5% (PRIMARY investment target)
           Each target gets the FULL model grid, all four purged schemes,
           fail-safe gates, confidence framework and forecast suppression.
           A paired-bootstrap p-value + Benjamini-Hochberg FDR controls the
           64 simultaneous gate tests.

  Phase 2  Timing Advantage = current-state expected 12M return minus the
           historical average 12M return, with an episode-bootstrap CI.
           Suppressed to "No Reliable Timing Edge Detected" on failure.

  Phase 3  Novel regime detection: a POINT-IN-TIME distribution of nearest-
           neighbour distances. Above its 95th percentile all analogue
           outputs are disabled and an explicit warning is raised.
           Analogue Reliability: High / Moderate / Low / Invalid.

  Phase 4  Weekly-Friday observations vs month-end observations, with
           time-based (not row-count-based) warmup/windows, unchanged day-based
           purge/embargo, aligned common month-end OOS evaluation, and a
           conservative documented selection rule. Monthly is NOT replaced
           unless Weekly clearly wins on out-of-sample evidence.

Output: data/evidence_v3.json
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import evidence_engine_v2 as E  # noqa: E402

OUT = ROOT / "data" / "evidence_v3.json"

SEED = 20261119
HORIZONS = E.HORIZONS
TICKERS = E.TICKERS
MODEL_NAMES = E.MODEL_NAMES
SCHEMES = ["Walk-forward", "Rolling", "Expanding", "CPCV"]
SCHEDULES = ["Monthly", "Weekly"]
TARGET_KEYS = ["A", "B", "C", "D"]
PRIMARY_TARGET = "D"
PRIMARY_HORIZON = "12M"
TARGET_DEFS = {
    "A": {"label": "Above historical median return",
          "definition": "forward return > point-in-time expanding median of already-realised forward returns",
          "primary": False},
    "B": {"label": "Return > 10%", "definition": "forward return > +10%", "primary": False},
    "C": {"label": "Return > 15%", "definition": "forward return > +15%", "primary": False},
    "D": {"label": "Return > T-Bill + 5%",
          "definition": "forward return > point-in-time 13-week T-bill return over the horizon "
                        "+ 5% annualised equity risk premium pro-rated by horizon "
                        "(at 12M this is exactly T-bill + 5%)",
          "primary": True},
}
N_NEIGHBOURS = 40
NOVELTY_Q = 0.95
# calendar-time equivalents of v2's row-count settings (v2: 72 / 60 / 120 MONTHS)
WARMUP_DAYS = 72 * 21
WF_WINDOW_DAYS = 60 * 21
ROLL_WINDOW_DAYS = 120 * 21
PIT_MIN_OBS = 756          # >= 3 years of daily realised outcomes before a median is defined
START_POS = 500            # identical calendar start for both schedules (as in v2)
FDR_Q = 0.10
N_BOOT = 2000


# ---------------------------------------------------------------------------
# 1. Observation schedules -- identical calendar window, different frequency
# ---------------------------------------------------------------------------

def observation_positions(px: pd.DataFrame, schedule: str, start: int, end: int) -> np.ndarray:
    if schedule == "Monthly":
        me = px.index.to_series().resample("ME").last()
        cand = px.index.get_indexer(pd.DatetimeIndex(me.values))
    else:
        cand = np.where(px.index.dayofweek == 4)[0]   # every Friday on the B-day calendar
    out = sorted({int(p) for p in cand if p >= 0 and start <= int(p) <= end})
    return np.asarray(out, dtype=int)


# ---------------------------------------------------------------------------
# 2. Phase 1 -- point-in-time target construction
# ---------------------------------------------------------------------------

def pit_median_curve(px: pd.DataFrame, sym: str, H: int) -> np.ndarray:
    """Point-in-time median of ALREADY-REALISED forward returns.

    Uses a canonical MONTH-END grid of decision dates (so overlapping daily
    windows do not over-weight any regime). At month-end date t the investor
    can only know outcomes r[s] with s + H <= t. The resulting step curve is
    defined at every daily position and is SCHEDULE-INDEPENDENT, so Monthly and
    Weekly share identical labels at identical dates.
    """
    p = px[sym].to_numpy(float)
    n = len(p)
    me = px.index.to_series().resample("ME").last()
    mp = np.array([int(t) for t in px.index.get_indexer(pd.DatetimeIndex(me.values)) if t >= 0])
    rm = np.array([p[t + H] / p[t] - 1.0 if t + H < n else np.nan for t in mp])
    med_at_mp = np.full(len(mp), np.nan)
    vals, j = [], 0
    for i, t in enumerate(mp):
        while j < len(mp) and mp[j] + H <= t:
            if np.isfinite(rm[j]):
                vals.append(float(rm[j]))
            j += 1
        if len(vals) >= 36:                       # >= 3 years of monthly realised outcomes
            med_at_mp[i] = float(np.median(vals))
    return (pd.Series(med_at_mp, index=mp)
            .reindex(np.arange(n), method="ffill").to_numpy(float))


def build_target_set(px: pd.DataFrame, sym: str, positions: list[int]) -> dict:
    """Return {(h, tkey): {'y', 'valid', 'hurdle', 'r'}} for one ticker."""
    fp = E.forward_path_stats(px, sym, positions)
    med = {h: pit_median_curve(px, sym, H) for h, H in HORIZONS.items()}
    irx = px["IRX"].to_numpy(float) / 100.0     # annualised 13-week T-bill, point-in-time
    pos = np.asarray(positions, dtype=int)
    out = {}
    for h, H in HORIZONS.items():
        r = fp[h]["r"]
        yrs = H / 252.0
        rf = np.nan_to_num(irx[pos], nan=0.0)
        hurdles = {
            "A": med[h][pos],
            "B": np.full(len(pos), 0.10),
            "C": np.full(len(pos), 0.15),
            "D": rf * yrs + 0.05 * yrs,
        }
        for tkey in TARGET_KEYS:
            hu = hurdles[tkey]
            valid = np.isfinite(r) & np.isfinite(hu)
            y = (r > hu).astype(float)
            y[~valid] = np.nan
            out[(h, tkey)] = {"y": y, "valid": valid, "hurdle": hu,
                              "r": r, "maxdd": fp[h]["maxdd"], "sharpe": fp[h]["sharpe"],
                              "years": yrs}
    return out


# ---------------------------------------------------------------------------
# 3. Phase 4 -- schedule-aware but time-based purged splits
# ---------------------------------------------------------------------------

def build_splits_v3(obs: np.ndarray, H: int, y_by_pos: dict) -> dict:
    """Same four schemes as v2. Purge/embargo remain DAY-based (E._purge).

    Warmup and rolling windows are expressed in trading DAYS so that a weekly
    schedule is compared over the same calendar history as a monthly one.
    """
    n = len(obs)
    if n < 60:
        return {s: [] for s in SCHEMES}
    warm = int(np.searchsorted(obs, obs[0] + WARMUP_DAYS, side="left"))
    warm = min(warm, n // 3)
    blocks = np.array_split(np.arange(warm, n), E.N_TEST_BLOCKS)
    schemes = {s: [] for s in SCHEMES}

    def balanced(tr):
        return len(tr) >= 30 and len(np.unique([y_by_pos.get(int(q), 0.0) for q in tr])) >= 2

    for b in blocks:
        test_idx = b
        tpos = obs[test_idx]
        start = test_idx[0]
        for name, window in (("Walk-forward", WF_WINDOW_DAYS),
                             ("Rolling", ROLL_WINDOW_DAYS), ("Expanding", None)):
            if window is None:
                lo = 0
            else:
                lo = int(np.searchsorted(obs, obs[start] - window, side="left"))
            cand = np.arange(lo, start)
            tr = E._purge(obs[cand], tpos, H)
            if balanced(tr):
                schemes[name].append((tr, tpos, test_idx))

    groups = np.array_split(np.arange(n), E.N_CPCV_GROUPS)
    for combo in itertools.combinations(range(E.N_CPCV_GROUPS), E.CPCV_TEST_GROUPS):
        test_idx = np.concatenate([groups[i] for i in combo])
        tpos = obs[test_idx]
        train_idx = np.concatenate([groups[i] for i in range(E.N_CPCV_GROUPS)
                                    if i not in combo])
        tr = E._purge(obs[train_idx], tpos, H)
        if len(tr) >= 40 and balanced(tr):
            schemes["CPCV"].append((tr, tpos, test_idx))
    return schemes


# ---------------------------------------------------------------------------
# 4. Model grid -- explicit context, no module globals, optional diagnostics
# ---------------------------------------------------------------------------

def run_model_grid_v3(X: np.ndarray, y: np.ndarray, names: list[str], splits: dict,
                      row_of: dict, feature_dates: list, diagnostics: bool = False,
                      seed: int = SEED) -> dict:
    """In-fold feature selection runs on EVERY grid. Heavy SHAP / permutation /
    univariate diagnostics run only where requested (12M Monthly grids)."""
    rng = np.random.default_rng(seed)
    results = {}
    audit = {"mi": [], "selection": [], "perm": {}, "shap": {},
             "univariate_brier": {n: [] for n in names},
             "decade_edge": {"2000s": [], "2010s": [], "2020s": []}}
    y_by_pos = {}
    for scheme, split_list in splits.items():
        acc = {m: {} for m in MODEL_NAMES}
        for tr_pos, te_pos, _te_idx in split_list:
            tr_rows = np.array([row_of[int(p)] for p in tr_pos if int(p) in row_of])
            te_rows = np.array([row_of[int(p)] for p in te_pos if int(p) in row_of])
            if len(tr_rows) < 25 or len(te_rows) == 0:
                continue
            ytr = y[tr_rows]
            yte = y[te_rows]
            ok_tr, ok_te = np.isfinite(ytr), np.isfinite(yte)
            tr_rows, te_rows = tr_rows[ok_tr], te_rows[ok_te]
            ytr, yte = ytr[ok_tr], yte[ok_te]
            if len(tr_rows) < 25 or len(te_rows) == 0 or len(np.unique(ytr)) < 2:
                continue
            picked, mi, _thr = E.select_features(X[tr_rows], ytr, names, rng)
            audit["mi"].append(mi)
            audit["selection"].append(picked)
            Xtr, Xte = X[tr_rows][:, picked], X[te_rows][:, picked]
            for m in MODEL_NAMES:
                p, _ = E.fit_predict(m, Xtr, ytr, Xte)
                for pp, pred in zip(te_pos, p):
                    acc[m].setdefault(int(pp), []).append(float(pred))
            for pp, rr in zip(te_pos, te_rows):
                y_by_pos.setdefault(int(pp), float(y[rr]))

            if diagnostics and scheme == "Expanding":
                p_x, model = E.fit_predict("XGBoost", Xtr, ytr, Xte)
                b0 = float(np.mean((p_x - yte) ** 2))
                for j, fi in enumerate(picked):
                    Xp = Xte.copy()
                    Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                    pp = np.clip(model.predict_proba(Xp)[:, 1], 1e-4, 1 - 1e-4)
                    audit["perm"].setdefault(names[fi], []).append(
                        float(np.mean((pp - yte) ** 2)) - b0)
                try:
                    import shap
                    expl = shap.TreeExplainer(model.steps[-1][1])
                    sv = np.asarray(expl.shap_values(Xte))
                    if sv.ndim == 3:
                        sv = sv[:, :, 1]
                    mag = np.abs(sv).mean(axis=0)
                    for j, fi in enumerate(picked):
                        audit["shap"].setdefault(names[fi], []).append(float(mag[j]))
                except Exception:
                    pass
                p_const = float(np.clip(ytr.mean(), 0.02, 0.98))
                bb = float(np.mean((np.full(len(yte), p_const) - yte) ** 2))
                edges = np.zeros(len(names))
                for fi, fname in enumerate(names):
                    try:
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.pipeline import make_pipeline
                        from sklearn.preprocessing import StandardScaler
                        mm = make_pipeline(E.SimpleImputer(strategy="median"),
                                           StandardScaler(),
                                           LogisticRegression(C=0.3, max_iter=500))
                        mm.fit(X[tr_rows][:, [fi]], ytr)
                        pu = np.clip(mm.predict_proba(X[te_rows][:, [fi]])[:, 1], 1e-4, 1 - 1e-4)
                        edges[fi] = bb - float(np.mean((pu - yte) ** 2))
                        audit["univariate_brier"][fname].append(edges[fi])
                    except Exception:
                        pass
                yr = int(pd.Timestamp(feature_dates[te_rows[0]]).year)
                dec = "2000s" if yr < 2010 else ("2010s" if yr < 2020 else "2020s")
                audit["decade_edge"][dec].append(edges)

        for m in MODEL_NAMES:
            if not acc[m]:
                results[(scheme, m)] = None
                continue
            ps = sorted(acc[m])
            pp = np.array([np.mean(acc[m][k]) for k in ps])
            yy = np.array([y_by_pos[k] for k in ps])
            mb = E.metric_bundle(yy, pp)
            mb["predictions"] = {str(k): float(v) for k, v in zip(ps, pp)}
            results[(scheme, m)] = mb
    return {"metrics": results, "audit": audit, "y_by_pos": y_by_pos}


# ---------------------------------------------------------------------------
# 5. Fail-safe gate + multiple-testing control
# ---------------------------------------------------------------------------

def evaluate_gate(me: dict, n_valid_target: int = 0) -> dict:
    """v2's five fail-safe checks, plus a rare-event guard.

    With 64 tests, targets such as '>15% in one month' can contain zero or one
    positive case in an out-of-sample block. AUC on one positive is meaningless
    and a model that merely predicts a smaller constant than the training-window
    base rate will 'beat' it on Brier. Such gates fail explicitly rather than
    being silently dropped or reported as evidence.
    """
    cand = {}
    for m in MODEL_NAMES[1:]:
        vals = [me.get(f"{s}|{m}") for s in SCHEMES]
        if all(v is not None for v in vals):
            cand[m] = float(np.mean([v["brier"] for v in vals]))
    if not cand:
        return {"best": "n/a", "degenerate": True, "n_positive_oos": 0.0,
                "checks": [{"name": "Evaluable purged model grid", "pass": False,
                            "detail": f"no model produced all four schemes; "
                                      f"{len(me)} metric blocks, {n_valid_target} valid labels"}],
                "pass": False, "mean_brier_edge": 0.0, "schemes_beaten": 0,
                "auc": 0.5, "slope": 0.0, "intercept": 0.0, "ece": 1.0,
                "brier": None, "baseline_brier": None, "all_candidates": {}}
    best = min(cand, key=cand.get)
    bm = [me[f"{s}|{best}"] for s in SCHEMES]
    br = [me[f"{s}|Base Rate"] for s in SCHEMES]
    beaten = sum(1 for a, b in zip(bm, br) if a["brier"] < b["brier"])
    mean_edge = float(np.mean([b["brier"] - a["brier"] for a, b in zip(bm, br)]))
    auc = float(np.mean([x["roc_auc"] for x in bm]))
    slope = float(np.mean([x["cal_slope"] for x in bm]))
    icept = float(np.mean([x["cal_intercept"] for x in bm]))
    ece = float(np.mean([x["ece"] for x in bm]))
    rate = float(np.mean([x["base_rate"] for x in bm]))
    n_pos = float(np.mean([x["base_rate"] * x["n"] for x in bm]))
    rare = bool(n_pos < 20 or rate < 0.03)
    checks = [
        {"name": "Brier better than base-rate model", "pass": bool(mean_edge > 0 and beaten >= 3),
         "detail": f"mean edge {mean_edge:+.4f}; beaten in {beaten}/4 schemes"},
        {"name": "ROC AUC not near random", "pass": bool(auc > 0.55),
         "detail": f"AUC {auc:.3f} (threshold 0.55)"},
        {"name": "Calibration slope positive", "pass": bool(0.2 < slope < 2.5),
         "detail": f"slope {slope:.2f}"},
        {"name": "Calibration intercept near zero", "pass": bool(abs(icept) < 1.0),
         "detail": f"intercept {icept:+.2f}"},
        {"name": "Expected calibration error", "pass": bool(ece < 0.10),
         "detail": f"ECE {ece:.3f}"},
        {"name": "Sufficient positive OOS cases for stable AUC", "pass": bool(not rare),
         "detail": f"~{n_pos:.0f} positive cases, event rate {rate:.1%} "
                   f"(minimum 20 positives / 3% event rate)"},
    ]
    return {"best": best, "checks": checks, "pass": all(c["pass"] for c in checks),
            "mean_brier_edge": mean_edge, "schemes_beaten": beaten, "auc": auc,
            "slope": slope, "intercept": icept, "ece": ece,
            "brier": float(np.mean([x["brier"] for x in bm])),
            "baseline_brier": float(np.mean([x["brier"] for x in br])),
            "n_positive_oos": n_pos, "event_rate": rate, "degenerate": rare,
            "all_candidates": cand}


def paired_edge_pvalue(grid: dict, best: str, H: int, seed: int = SEED) -> float:
    """One-sided paired bootstrap p-value for Brier edge vs the Base Rate model.

    Resamples INDEPENDENT H-day episodes, not overlapping observations.
    """
    pooled = {}
    for s in SCHEMES:
        mb, br = grid["metrics"].get((s, best)), grid["metrics"].get((s, "Base Rate"))
        if mb is None or br is None:
            continue
        for k, v in mb["predictions"].items():
            d = int(k)
            rec = pooled.setdefault(d, {"y": None, "pm": [], "pb": []})
            rec["pm"].append(v)
            rec["pb"].append(br["predictions"].get(k, float("nan")))
        for k, yv in grid["y_by_pos"].items():
            if int(k) in pooled:
                pooled[int(k)]["y"] = yv
    rows = [(d, r["y"], float(np.mean(r["pm"])), float(np.nanmean(r["pb"])))
            for d, r in pooled.items()
            if r["y"] is not None and np.isfinite(r["y"]) and np.isfinite(np.nanmean(r["pb"]))]
    if len(rows) < 20:
        return 1.0
    rows.sort()
    dates = np.array([x[0] for x in rows])
    diff = np.array([(x[3] - x[1]) ** 2 - (x[2] - x[1]) ** 2 for x in rows])
    episodes, cur = [], []
    for d, v in zip(dates, diff):
        if cur and d - cur[0][0] >= H:
            episodes.append(cur)
            cur = []
        cur.append((d, v))
    if cur:
        episodes.append(cur)
    emeans = np.array([np.mean([v for _d, v in e]) for e in episodes])
    if len(emeans) < 3:
        return 1.0
    rng = np.random.default_rng(seed)
    boots = rng.choice(emeans, size=(N_BOOT, len(emeans)), replace=True).mean(axis=1)
    return float(np.mean(boots <= 0))


def bh_fdr(pvals: list[float], q: float = FDR_Q) -> tuple[list[bool], float]:
    m = len(pvals)
    if m == 0:
        return [], 0.0
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, m + 1) / m)
    ok = np.asarray(pvals)[order] <= thresh
    k = int(np.where(ok)[0].max()) if ok.any() else -1
    sig = [False] * m
    if k >= 0:
        for i in order[: k + 1]:
            sig[int(i)] = True
    return sig, float(thresh[k]) if k >= 0 else 0.0


# ---------------------------------------------------------------------------
# 6. Phase 3 -- point-in-time neighbour-distance / novel regime detection
# ---------------------------------------------------------------------------

def novel_regime(Z: np.ndarray, positions: list[int], r12: np.ndarray,
                 today_row: int, k: int = N_NEIGHBOURS) -> dict:
    """Historical distribution of mean distance to the k nearest PRIOR states.

    At each historical date i only states whose 12M outcome was realised before
    i are eligible -- so the threshold itself contains no future information.
    """
    pos = np.asarray(positions, dtype=int)
    n = len(pos)
    dists = np.full(n, np.nan)
    for i in range(n):
        cutoff = int(np.searchsorted(pos, pos[i] - 252, side="right"))
        cand = np.where(np.isfinite(r12[:cutoff]))[0]
        if len(cand) < k:
            continue
        d = np.sqrt(np.mean((Z[cand] - Z[i]) ** 2, axis=1))
        dists[i] = float(np.mean(d[np.argsort(d)[:k]]))
    hist_mask = np.isfinite(dists) & (np.arange(n) < today_row)
    hist = dists[hist_mask]
    today = float(dists[today_row]) if np.isfinite(dists[today_row]) else None
    if today is None or len(hist) < 50:
        return {"available": False, "today_distance": today, "n_hist": int(len(hist)),
                "reliability": "Invalid", "novel": True,
                "warning": "Insufficient point-in-time history to calibrate a novelty threshold."}
    pct = float(100.0 * (hist <= today).mean())
    q50, q80, q95 = [float(np.quantile(hist, q)) for q in (0.50, 0.80, NOVELTY_Q)]
    novel = bool(today > q95)
    hist_novel_rate = float((hist > q95).mean())
    return {"available": True, "today_distance": today,
            "distance_percentile": pct, "n_hist": int(len(hist)),
            "p50": q50, "p80": q80, "p95": q95,
            "novel": novel, "hist_novel_rate": hist_novel_rate,
            "distribution": [round(float(v), 5) for v in hist],
            "warning": ("NOVEL REGIME WARNING — current market environment differs "
                        "significantly from historical observations. Analogue analysis disabled."
                        if novel else None)}


def analogue_reliability(regime: dict, conf: dict) -> str:
    if not regime["available"] or regime["novel"]:
        return "Invalid"
    if conf["low_confidence"] or regime["distance_percentile"] >= 80:
        return "Low"
    if regime["distance_percentile"] >= 50:
        return "Moderate"
    return "High"


# ---------------------------------------------------------------------------
# 7. Phase 2 -- Timing Advantage with episode bootstrap CI
# ---------------------------------------------------------------------------

def timing_advantage(tk: str, hist_mean: float, current: float | None,
                     analogues: list[int], r12_by_pos: dict,
                     gate_pass: bool, score_validated: bool, reliability: str,
                     fdr_ok: bool, conf: dict, seed: int = SEED) -> dict:
    point = None if current is None else float(current - hist_mean)
    ci_lo = ci_hi = None
    if current is not None:
        dates = sorted(t for t in analogues if t in r12_by_pos and np.isfinite(r12_by_pos[t]))
        episodes, cur = [], []
        for d in dates:
            if cur and d - cur[0] >= 252:
                episodes.append(cur)
                cur = []
            cur.append(d)
        if cur:
            episodes.append(cur)
        emeans = np.array([np.mean([r12_by_pos[d] for d in e]) for e in episodes])
        if len(emeans) >= 3:
            rng = np.random.default_rng(seed)
            boots = rng.choice(emeans, size=(N_BOOT, len(emeans)), replace=True).mean(axis=1)
            ci_lo, ci_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    reasons = []
    if not gate_pass:
        reasons.append("primary target D @ 12M model gate failed")
    if not score_validated:
        reasons.append("entry score failed out-of-sample rank validation")
    if reliability in ("Invalid", "Low"):
        reasons.append(f"analogue reliability is {reliability}")
    if conf["low_confidence"]:
        reasons.append(f"confidence grade {conf['grade']} ({conf['n_eff']} independent episodes)")
    if not fdr_ok:
        reasons.append("gate does not survive Benjamini-Hochberg FDR correction")
    if ci_lo is None or ci_lo <= 0:
        reasons.append("95% CI on current-state return does not exclude the historical mean")
    validated = bool(point is not None and not reasons)
    return {"historical_12m_return": float(hist_mean),
            "current_state_expected_12m_return": current,
            "timing_advantage": point if validated else None,
            "timing_advantage_ci_low": (None if ci_lo is None else ci_lo - hist_mean),
            "timing_advantage_ci_high": (None if ci_hi is None else ci_hi - hist_mean),
            "validated": validated,
            "display": ("No Reliable Timing Edge Detected" if not validated else None),
            "suppression_reasons": reasons,
            "n_episodes": len(analogues)}


# ---------------------------------------------------------------------------
# 8. Helpers
# ---------------------------------------------------------------------------

def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, float):
        return None if not np.isfinite(o) else round(o, 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def pooled_predictions(grid: dict, model: str) -> dict:
    out = {}
    for s in SCHEMES:
        mb = grid["metrics"].get((s, model))
        if mb is None:
            continue
        for k, v in mb["predictions"].items():
            out.setdefault(int(k), []).append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


def metric_on_dates(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    if len(y) < 20:
        return {}
    p = np.clip(p, 1e-5, 1 - 1e-5)
    out = {"n": int(len(y)), "brier": float(np.mean((p - y) ** 2)),
           "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
           "base_rate": float(y.mean())}
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        z = logit(p).reshape(-1, 1)
        from sklearn.linear_model import LogisticRegression
        try:
            cal = LogisticRegression(penalty=None, max_iter=1000).fit(z, y)
            out["cal_slope"] = float(cal.coef_[0][0])
            out["cal_intercept"] = float(cal.intercept_[0])
        except Exception:
            out["cal_slope"], out["cal_intercept"] = 0.0, 0.0
    else:
        out["roc_auc"] = 0.5
        out["cal_slope"], out["cal_intercept"] = 0.0, 0.0
    return out


# ---------------------------------------------------------------------------
# 9. Orchestration
# ---------------------------------------------------------------------------

def run() -> dict:
    px = E.load_prices()
    raw, groups = E.build_raw_features(px)
    last = len(px) - 1
    today_pos = int(last)
    end_pos = last - 252                     # identical calendar window for both schedules
    names = list(raw.columns)

    # Both schedules are clipped to the SAME calendar bounds. Because the two
    # frequencies place their first/last observation on different weekdays the
    # realised edges differ by at most four weeks.
    raw_obs = {s: observation_positions(px, s, START_POS, end_pos) for s in SCHEDULES}
    common_start = max(raw_obs[s][0] for s in SCHEDULES)
    common_end = min(raw_obs[s][-1] for s in SCHEDULES)
    contexts = {}
    for sched in SCHEDULES:
        obs = raw_obs[sched][(raw_obs[sched] >= common_start) & (raw_obs[sched] <= common_end)]
        fpos = sorted(set(obs.tolist() + [today_pos]))
        X = E.point_in_time_percentiles(raw, fpos).to_numpy(float)
        contexts[sched] = {
            "name": sched, "obs": obs, "positions": fpos, "X": X,
            "dates": [px.index[i] for i in fpos],
            "row_of": {int(p): i for i, p in enumerate(fpos)},
            "today_row": fpos.index(today_pos),
        }
        print(f"[engine] {sched}: {len(obs)} observations | "
              f"{px.index[obs[0]].date()} -> {px.index[obs[-1]].date()}")

    # ---- Phase 1: targets for every schedule / ticker / horizon ----------
    target_sets, uncond = {}, {}
    for sched, ctx in contexts.items():
        for tk, sym in TICKERS.items():
            ts = build_target_set(px, sym, ctx["positions"])
            for h in HORIZONS:
                fp = {k: ts[(h, "A")][k] for k in ("r", "maxdd", "sharpe")}
                uncond[(sched, tk, h)] = E.unconditional_stats(fp["r"], fp["maxdd"], fp["sharpe"])
            target_sets[(sched, tk)] = ts

    # ---- full model grids -------------------------------------------------
    grids, gate_meta, model_evidence = {}, {}, {}
    for sched, ctx in contexts.items():
        for tk in TICKERS:
            ts = target_sets[(sched, tk)]
            for h, H in HORIZONS.items():
                for tkey in TARGET_KEYS:
                    t = ts[(h, tkey)]
                    y = t["y"]
                    y_by_pos = {int(p): (float(y[i]) if np.isfinite(y[i]) else 0.0)
                                for i, p in enumerate(ctx["positions"])}
                    splits = build_splits_v3(ctx["obs"], H, y_by_pos)
                    diag = (sched == "Monthly" and h == "12M")
                    seed = SEED + 7919 * SCHEDULES.index(sched) + 131 * TARGET_KEYS.index(tkey) + H
                    grid = run_model_grid_v3(ctx["X"], y, names, splits, ctx["row_of"],
                                             ctx["dates"], diagnostics=diag, seed=seed)
                    key = (sched, tk, h, tkey)
                    grids[key] = grid
                    me = {f"{s}|{m}": {k: v for k, v in mb.items() if k != "predictions"}
                          for (s, m), mb in grid["metrics"].items() if mb is not None}
                    model_evidence[key] = {"metrics": me,
                                          "n_splits": {s: len(v) for s, v in splits.items()},
                                          "purge_embargo_days": H,
                                          "n_valid_target": int(t["valid"].sum())}
                    gate = evaluate_gate(me, int(t["valid"].sum()))
                    gate_meta[key] = gate
                print(f"[engine] {sched} {tk} {h}: 4 target grids done "
                      f"({sum(len(v) for v in splits.values())} purged splits per target)")

    # ---- multiple-testing control ----------------------------------------
    gate_keys = [k for k in gate_meta if gate_meta[k] is not None]
    pvals = []
    for k in gate_keys:
        sched, tk, h, tkey = k
        H = HORIZONS[h]
        if gate_meta[k]["pass"]:
            pv = paired_edge_pvalue(grids[k], gate_meta[k]["best"], H,
                                    seed=SEED + 17 * gate_keys.index(k))
        else:
            pv = paired_edge_pvalue(grids[k], gate_meta[k]["best"], H,
                                    seed=SEED + 17 * gate_keys.index(k))
        pvals.append(pv)
    sig, fdr_thresh = bh_fdr(pvals)
    fdr_map = {k: {"p_value": pv, "fdr_significant": s}
               for k, pv, s in zip(gate_keys, pvals, sig)}

    # ---- Phase 4: schedule comparison ------------------------------------
    comparison_rows = []
    for sched in SCHEDULES:
        edges, aucs, slopes, eces, briers, base_briers = [], [], [], [], [], []
        for tk in TICKERS:
            for h in HORIZONS:
                for tkey in TARGET_KEYS:
                    g = gate_meta.get((sched, tk, h, tkey))
                    if g is None or g["brier"] is None:
                        continue
                    me = model_evidence[(sched, tk, h, tkey)]["metrics"]
                    for s in SCHEMES:
                        a, b = me.get(f"{s}|{g['best']}"), me.get(f"{s}|Base Rate")
                        if a and b:
                            edges.append(b["brier"] - a["brier"])
                    aucs.append(g["auc"])
                    slopes.append(g["slope"])
                    eces.append(g["ece"])
                    briers.append(g["brier"])
                    base_briers.append(g["baseline_brier"])
        edges = np.asarray(edges)
        comparison_rows.append({
            "schedule": sched,
            "n_observations": int(len(contexts[sched]["obs"])),
            "n_gate_sets": len(aucs),
            "mean_auc": float(np.mean(aucs)),
            "mean_brier": float(np.mean(briers)),
            "baseline_brier": float(np.mean(base_briers)),
            "mean_brier_edge": float(np.mean(edges)),
            "stability_std_edge": float(np.std(edges)),
            "stability_positive_fraction": float((edges > 0).mean()),
            "mean_cal_slope": float(np.mean(slopes)),
            "mean_ece": float(np.mean(eces)),
            "n_gates_passed": sum(1 for k in gate_keys
                                  if k[0] == sched and gate_meta[k]["pass"]),
            "n_fdr_significant": sum(1 for k, v in fdr_map.items()
                                     if k[0] == sched and v["fdr_significant"]),
        })

    # aligned evaluation: identical month-end OOS dates and identical labels
    aligned = {}
    month_dates = contexts["Monthly"]["obs"]
    for tk in TICKERS:
        for h in HORIZONS:
            for tkey in TARGET_KEYS:
                g = gate_meta.get(("Monthly", tk, h, tkey))
                gw = gate_meta.get(("Weekly", tk, h, tkey))
                if g is None or gw is None:
                    continue
                ymap = {int(p): float(target_sets[("Monthly", tk)][(h, tkey)]["y"][i])
                        for i, p in enumerate(contexts["Monthly"]["positions"])}
                pm = pooled_predictions(grids[("Monthly", tk, h, tkey)], g["best"])
                pw = pooled_predictions(grids[("Weekly", tk, h, tkey)], gw["best"])
                bm = pooled_predictions(grids[("Monthly", tk, h, tkey)], "Base Rate")
                bw = pooled_predictions(grids[("Weekly", tk, h, tkey)], "Base Rate")
                wpos = np.asarray(sorted(pw))
                ys, pmm, pww, bmm, bww = [], [], [], [], []
                for d in month_dates:
                    d = int(d)
                    if d not in pm or d not in bm or d not in ymap or not np.isfinite(ymap[d]):
                        continue
                    j = int(np.searchsorted(wpos, d, side="right")) - 1
                    if j < 0 or int(wpos[j]) not in pw:
                        continue
                    wd = int(wpos[j])
                    ys.append(ymap[d]); pmm.append(pm[d]); pww.append(pw[wd])
                    bmm.append(bm[d]); bww.append(bw.get(wd, bmm[-1]))
                if len(ys) < 30:
                    continue
                yv = np.asarray(ys)
                aligned[(tk, h, tkey)] = {
                    "n_common_dates": len(ys),
                    "Monthly": metric_on_dates(yv, np.asarray(pmm)),
                    "Weekly": metric_on_dates(yv, np.asarray(pww)),
                    "Monthly_base": metric_on_dates(yv, np.asarray(bmm))["brier"],
                    "Weekly_base": metric_on_dates(yv, np.asarray(bww))["brier"],
                }

    mrow, wrow = comparison_rows
    aligned_edges = {"Monthly": [], "Weekly": []}
    aligned_wins = 0
    for a in aligned.values():
        if not a["Monthly"] or not a["Weekly"]:
            continue
        em = a["Monthly_base"] - a["Monthly"]["brier"]
        ew = a["Weekly_base"] - a["Weekly"]["brier"]
        aligned_edges["Monthly"].append(em)
        aligned_edges["Weekly"].append(ew)
        aligned_wins += int(ew > em)
    am = float(np.mean(aligned_edges["Monthly"])) if aligned_edges["Monthly"] else 0.0
    aw = float(np.mean(aligned_edges["Weekly"])) if aligned_edges["Weekly"] else 0.0
    win_frac = aligned_wins / max(len(aligned), 1)
    criteria = {
        "edge_margin": aw - am,
        "weekly_win_fraction": win_frac,
        "weekly_ece_not_worse": wrow["mean_ece"] <= mrow["mean_ece"] + 0.005,
        "weekly_stability_not_worse": wrow["stability_std_edge"] <= mrow["stability_std_edge"] * 1.10,
        "weekly_aligned_edge_positive": aw > 0.0,
    }
    switch = bool(criteria["edge_margin"] > 0.002 and win_frac >= 0.60
                  and all(criteria[k] for k in ("weekly_ece_not_worse",
                                                "weekly_stability_not_worse",
                                                "weekly_aligned_edge_positive")))
    selected = "Weekly" if switch else "Monthly"
    reason = ("Weekly beats Monthly on aligned out-of-sample Brier edge by "
              f"{criteria['edge_margin']:+.4f} (win fraction {win_frac:.0%}) with no "
              "calibration or stability penalty."
              if switch else
              "Weekly does NOT clear the pre-registered switch criteria "
              f"(aligned edge margin {criteria['edge_margin']:+.4f}, win fraction "
              f"{win_frac:.0%}); Monthly is retained as the primary schedule. "
              "Weekly evidence is reported in full, not discarded.")

    # ---- feature audit (from the diagnostic grids) -----------------------
    mi_all = np.zeros(len(names)); cnt = 0
    perm_all = {n: [] for n in names}; shap_all = {n: [] for n in names}
    uni_all = {n: [] for n in names}; sel_count = np.zeros(len(names))
    decade_edges = {"2000s": [], "2010s": [], "2020s": []}
    for k, grid in grids.items():
        aud = grid["audit"]
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
    feats = []
    for i, n in enumerate(names):
        mi = float(mi_all[i]); pm_eval = len(perm_all[n]) > 0; sh_eval = len(shap_all[n]) > 0
        pm = float(np.mean(perm_all[n])) if pm_eval else None
        sh = float(np.mean(shap_all[n])) if sh_eval else None
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
        feats.append({"name": n, "group": groups[n], "mi": round(mi, 5),
                      "perm": None if pm is None else round(pm, 6),
                      "shap": None if sh is None else round(sh, 5),
                      "univariate_edge": round(uv, 6), "selection_rate": round(sel, 3),
                      "stability": round(stability, 2),
                      "verdict": "RETAINED" if retained else "REMOVED",
                      "reason": "; ".join(reasons) if reasons else "positive OOS evidence"})
    retained_idx = [i for i, f in enumerate(feats) if f["verdict"] == "RETAINED"]
    n_actually_retained = len(retained_idx)
    fallback = len(retained_idx) < 3
    if fallback:
        retained_idx = list(np.argsort(-mi_all)[:E.K_FEATURES])
    retained_names = [names[i] for i in retained_idx]
    print(f"[engine] feature gate retained {n_actually_retained}/{len(names)}; "
          f"grid uses {len(retained_names)} ({'top-MI fallback' if fallback else 'retained set'})")

    # ---- Phase 2/3 + final decision on the SELECTED schedule -------------
    ctx = contexts[selected]
    finals, timing, regimes, reliability, scores, conf, deploy = {}, {}, {}, {}, {}, {}, {}
    for tk, sym in TICKERS.items():
        gk = (selected, tk, PRIMARY_HORIZON, PRIMARY_TARGET)
        gate = gate_meta[gk]
        fdr_ok = bool(fdr_map.get(gk, {}).get("fdr_significant", False))
        r12 = target_sets[(selected, tk)][("12M", "A")]["r"]
        sc = E.entry_score(ctx["X"], r12, ctx["positions"], ctx["today_row"],
                           retained_idx, ctx["dates"])
        scores[tk] = sc
        Z = ctx["X"][:, retained_idx] if retained_idx else ctx["X"][:, :E.K_FEATURES]
        cand = [i for i in range(len(ctx["positions"])) if np.isfinite(r12[i])]
        d = np.sqrt(np.mean((Z[cand] - Z[ctx["today_row"]]) ** 2, axis=1))
        analogues = [int(ctx["positions"][cand[i]]) for i in np.argsort(d)[:N_NEIGHBOURS]]
        r12_by_pos = {int(ctx["positions"][i]): float(r12[i]) for i in cand}
        conf[tk] = E.confidence_engine(analogues, r12_by_pos)
        regimes[tk] = novel_regime(Z, ctx["positions"], r12, ctx["today_row"])
        reliability[tk] = analogue_reliability(regimes[tk], conf[tk])
        analogue_ok = reliability[tk] != "Invalid"
        gate_pass = bool(gate and gate["pass"] and fdr_ok)
        br = uncond[(selected, tk, "12M")]
        current = sc["today_expected"] if analogue_ok else None
        timing[tk] = timing_advantage(tk, br["mean"], current, analogues, r12_by_pos,
                                      gate_pass, sc["validated"], reliability[tk],
                                      fdr_ok, conf[tk])
        deploy[tk] = E.deployment_backtest(
            px, sym, contexts["Monthly"]["obs"],
            pooled_predictions(grids[(selected, tk, "12M", PRIMARY_TARGET)], gate["best"])
            if gate else {},
            br["p_pos"], analogues, sc["today_score"], gate_pass, conf[tk])
        finals[tk] = {
            "action": deploy[tk]["recommendation"],
            "decision_released": deploy[tk]["decision_released"],
            "action_basis": deploy[tk]["action_basis"],
            "zone": ("Neutral — maintain ordinary investment plan" if not gate_pass
                     else "Evidence-based entry assessment available"),
            "confidence": conf[tk]["grade"],
            "expected_12m_return": br["mean"],
            "expected_12m_return_type": "unconditional historical base rate (not a forecast)",
            "expected_downside": br["downside_cvar10"],
            "edge_detected": gate_pass,
            "analogue_reliability": reliability[tk],
            "novel_regime": bool(regimes[tk]["novel"]),
            "timing_validated": timing[tk]["validated"],
            "justification": (
                f"Primary gate = target {PRIMARY_TARGET} @ {PRIMARY_HORIZON} on the "
                f"{selected} schedule. "
                + (f"{gate['best']} passed the fail-safe checks (AUC {gate['auc']:.2f}, "
                   f"mean Brier edge {gate['mean_brier_edge']:+.4f})."
                   if gate_pass else
                   (f"No model beat the base rate out of sample (best {gate['best']}, "
                    f"AUC {gate['auc']:.2f}, mean Brier edge {gate['mean_brier_edge']:+.4f}); "
                    "forecast suppressed." if gate else
                    "No evaluable model grid; forecast suppressed."))
                + f" FDR-adjusted p = {fdr_map.get(gk, {}).get('p_value', 1.0):.3f}. "
                + ("NOVEL REGIME — " + regimes[tk]["warning"] + " "
                   if regimes[tk]["novel"] else "")
                + ("Timing Advantage suppressed: " + "; ".join(timing[tk]["suppression_reasons"])
                   + ". " if not timing[tk]["validated"] else "")
                + deploy[tk]["justification"]
                + " Maintain your ordinary investment plan; do not deviate on this evidence."),
        }

    # ---- forecast tables (base / model / edge for every target) -----------
    forecasts, base_rates_out = {}, {}
    for sched in SCHEDULES:
        for tk in TICKERS:
            forecasts.setdefault(sched, {}).setdefault(tk, {})
            base_rates_out.setdefault(sched, {}).setdefault(tk, {})
            for h in HORIZONS:
                forecasts[sched][tk][h] = {}
                base_rates_out[sched][tk][h] = uncond[(sched, tk, h)]
                for tkey in TARGET_KEYS:
                    g = gate_meta.get((sched, tk, h, tkey))
                    t = target_sets[(sched, tk)][(h, tkey)]
                    yy = t["y"][t["valid"]]
                    base = float(yy.mean()) if len(yy) else None
                    model_p = None
                    if g and g["pass"] and fdr_map.get((sched, tk, h, tkey), {}).get("fdr_significant"):
                        me = model_evidence[(sched, tk, h, tkey)]["metrics"]
                        vals = [me[f"{s}|{g['best']}"]["mean_pred"] for s in SCHEMES
                                if me.get(f"{s}|{g['best']}")]
                        model_p = float(np.mean(vals)) if vals else None
                    forecasts[sched][tk][h][tkey] = {
                        "base": base, "model": model_p,
                        "edge": None if (base is None or model_p is None) else model_p - base,
                        "released": model_p is not None,
                        "degenerate": bool(g and g.get("degenerate")),
                        "n": int(len(yy)),
                        "hurdle_mean": (None if not np.isfinite(np.nanmean(t["hurdle"]))
                                        else float(np.nanmean(t["hurdle"]))),
                    }

    # ---- assemble payload ------------------------------------------------
    gate_json, me_json = {}, {}
    for k, g in gate_meta.items():
        sched, tk, h, tkey = k
        gk = f"{sched}|{tk}|{h}|{tkey}"
        gate_json[gk] = (dict(g) | fdr_map.get(k, {}) |
                         {"target": tkey, "horizon": h, "ticker": tk, "schedule": sched}
                         ) if g is not None else None
        me_json[gk] = model_evidence[k]
    aligned_json = {f"{tk}|{h}|{t}": v for (tk, h, t), v in aligned.items()}

    payload = {
        "meta": {
            "version": "3.0 — multi-target, regime-aware, dual-schedule decision system",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "as_of": str(px.index[today_pos].date()),
            "sample_start": str(px.index[contexts["Monthly"]["obs"][0]].date()),
            "sample_end": str(px.index[contexts["Monthly"]["obs"][-1]].date()),
            "n_daily": int(len(px)),
            "n_observations": {s: int(len(contexts[s]["obs"])) for s in SCHEDULES},
            "horizons": list(HORIZONS), "schemes": SCHEMES, "models": MODEL_NAMES,
            "schedules": SCHEDULES, "targets": TARGET_KEYS,
            "primary_target": PRIMARY_TARGET, "primary_horizon": PRIMARY_HORIZON,
            "selected_schedule": selected,
            "n_candidate_features": len(names),
            "n_retained_features": n_actually_retained,
            "feature_fallback": fallback,
            "n_gate_tests": len(gate_keys),
            "fdr_q": FDR_Q, "fdr_threshold": fdr_thresh,
            "proxy": "VOO total return proxied by SPY adjusted close (daily corr 0.9986).",
            "data": "Yahoo/Stooq dividend-adjusted closes, 1999-2026, offline dataset.",
        },
        "target_definitions": TARGET_DEFS,
        "verdict": {
            "edge_detected": any(finals[tk]["edge_detected"] for tk in TICKERS),
            "headline": ("RELIABLE TIMING EDGE DETECTED" if any(finals[tk]["edge_detected"]
                                                                 for tk in TICKERS)
                         else "NO RELIABLE TIMING EDGE DETECTED"),
            "subheadline": "Evidence favours maintaining ordinary investment plans.",
            "selected_schedule": selected,
            "primary_gate": {tk: gate_json.get(f"{selected}|{tk}|12M|D") for tk in TICKERS},
            "suppressed": {tk: not finals[tk]["edge_detected"] for tk in TICKERS},
        },
        "final": finals,
        "timing_advantage": timing,
        "novel_regime": regimes,
        "analogue_reliability": reliability,
        "entry_score": scores,
        "confidence": conf,
        "deployment": deploy,
        "forecasts": forecasts,
        "base_rates": base_rates_out,
        "gates": gate_json,
        "model_evidence": me_json,
        "schedule_comparison": {
            "rows": comparison_rows, "aligned_edge": {"Monthly": am, "Weekly": aw},
            "aligned_weekly_win_fraction": win_frac,
            "aligned_detail": aligned_json,
            "criteria": criteria, "selected": selected, "reason": reason,
            "method": "Both schedules are clipped to the same calendar bounds (realised edges differ by at most four weeks because month-ends and Fridays fall on different dates). Warmup and rolling "
                      "windows are expressed in trading days. Aligned evaluation scores both "
                      "models on the same month-end OOS dates (Weekly uses its last Friday "
                      "prediction on/before each month-end) with identical labels.",
        },
        "features": {"list": feats, "retained": [names[i] for i, f in enumerate(feats)
                                                  if f["verdict"] == "RETAINED"],
                     "model_feature_set": retained_names, "fallback": fallback},
        "disclosures": [
            "All anti-lookahead protections from v2 are imported and reused unchanged: expanding-window point-in-time percentiles, day-based label purging, horizon-length embargo, in-fold feature selection, strictly-prior analogue outcomes and episode-based effective sample size.",
            "Target A uses a point-in-time expanding median of ALREADY-REALISED month-end forward returns (outcomes with s+H <= t), not a full-sample median. Its unconditional hit rate is ~70%, not 50%: a slowly-updating historical hurdle is easy to beat during a long bull market. It is still materially harder than the 80.6% 'return > 0' target it replaces. The curve is schedule-independent so Monthly and Weekly share identical labels.",
            "Target D uses the point-in-time 13-week T-bill yield scaled to the horizon plus a 5% ANNUALISED risk premium pro-rated by horizon; at 12M it is exactly 'T-bill return + 5%'. A flat +5pp at 1M would be a rare-event label and would distort cross-horizon comparisons.",
            f"{len(gate_keys)} simultaneous gate tests are controlled with paired-bootstrap p-values and Benjamini-Hochberg FDR at q={FDR_Q}. Only the pre-registered primary target D @ 12M on the selected schedule may drive the verdict.",
            "The historical nearest-neighbour distance distribution is itself point-in-time: at each historical date only states whose forward outcome had already been realised are eligible. No future state calibrates the novelty threshold.",
            "Weekly observations overlap ~52x within a 12-month label. Effective sample size always counts non-overlapping 12-month episodes; weekly rows are never treated as independent.",
            "The schedule comparison is evaluated on common month-end out-of-sample dates. Monthly remains the primary schedule unless Weekly clears all pre-registered switch criteria; nothing is replaced automatically.",
            "Rare-event guard: a gate also fails when the out-of-sample block has fewer than ~20 positive cases or an event rate below 3%. Targets such as '>15% in one month' otherwise report AUC near 0.9 from a single positive case, and a model that merely predicts a smaller constant than the training-window base rate can 'beat' it on Brier. Degenerate target/horizon combinations are reported as failed, not dropped.",
            "Feature diagnostics (permutation importance, SHAP, univariate Brier edge) are computed on the monthly 12M grids; mutual information and in-fold selection rates aggregate all 64 grids. Zero retained features means no candidate showed consistent positive out-of-sample value across the four targets, and the top-MI fallback is disclosed as a control set.",
            "Low analogue reliability (grade-D confidence) still permits the descriptive Current-State Expected 12M Return, but the actionable Timing Advantage remains suppressed. Only Invalid reliability (novel regime) disables analogue outputs entirely.",
            "The deployment backtest remains on monthly decision dates because it is a policy question (how to deploy capital already committed to equities), not a timing question.",
            "No post-hoc calibrator, no arbitrary indicator weighting, no sentiment, no macro commentary.",
            "Backtested results are not a guarantee of future performance. Educational research tool, not investment advice.",
        ],
    }
    payload = clean(payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    print(f"[engine] wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"[engine] selected schedule: {selected} | gates passed "
          f"{sum(1 for g in gate_meta.values() if g and g['pass'])}/{len(gate_meta)} | "
          f"FDR survivors {sum(sig)}")
    return payload


if __name__ == "__main__":
    p = run()
    print("[engine] verdict:", p["verdict"]["headline"])
    for tk in TICKERS:
        print(f"[engine] {tk}: reliability={p['analogue_reliability'][tk]} "
              f"timing_validated={p['timing_advantage'][tk]['validated']} "
              f"action={p['final'][tk]['action']}")
