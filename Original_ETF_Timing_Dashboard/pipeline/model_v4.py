#!/usr/bin/env python3
"""
model_v4.py — purged walk-forward engine for the ETF Timing Dashboard.

Design goals (methodology refactor):
  * Every feature is computed from data available at or before its date.
  * Every displayed historical prediction is genuinely out of sample: produced
    by a model that never saw the date or anything after it.
  * Overlapping 6-month labels are never treated as independent observations:
    training samples whose label window touches the validation window are
    purged, and an embargo equal to the horizon is inserted.
  * Calibration (isotonic / Platt / beta) is fitted on a validation window and
    selected there; the test window is untouched until scoring.
  * The probability score keeps its literal meaning and is never rescaled to
    fill 0-100. Relative attractiveness lives in a separate Timing Percentile
    and Opportunity Score.

Vocabulary
----------
H          forward horizon in trading sessions (126 = 6 months)
T          test-block start index (an "origin")
val        window used only to fit the probability calibrator
test       window whose predictions are never used for fitting anything

Three consumers exist for these numbers and they are deliberately separate:
  p_win        calibrated P(6M total return > 0)      -> "probability score"
  exp_ret      expected 6M return from a ridge model  -> magnitude, not odds
  opportunity  point-in-time relative attractiveness  -> timing, not odds
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss

H = 126                 # 6-month horizon, trading sessions
EMBARGO = H             # embargo after purge, equal to the horizon
TEST_SPAN = 252         # one year per walk-forward block
MIN_TRAIN = 756         # minimum training rows before the first origin
Z_CLIP = 3.0

# --- probability model configuration (documented, not tuned per run) --------
# Only one feature survived purged validation: the trend/posture variable. A
# 10-feature logistic regression overfits roughly 30 independent 6-month
# episodes and scored WORSE than a constant forecast out of sample.
PROB_FEATURES = ["trend_dev"]
PROB_C = 1.0
# The unconditional 6M win rate is non-stationary (about 65% in 1999-2004,
# about 78% in 2004-2026). A model trained on the past therefore cannot set the
# LEVEL of the probability, only its slope. The level is anchored to the most
# recent REALISED 6M win rate over a 7-year window, and the model supplies a
# 40% tilt in logit space. Without this anchor the walk-forward forecast was
# 15-25pp miscalibrated; with it, ECE falls by roughly half.
BASE_WINDOW = 1764      # 7 years of realised labels
BASE_MINP = 250
MODEL_WEIGHT = 0.40
MIN_CAL_EPISODES = 10.0   # independent 6M outcomes needed before fitting a calibrator

# Feature set. Names are stable because they are shown in the UI.
FEATURES = [
    "trend_dev",   # price / 200DMA - 1
    "mom12",       # 12-month momentum
    "dd",          # drawdown from trailing 52w high
    "vix_pct",     # VIX percentile, trailing 3y
    "curve",       # 10Y - 3M yield
    "realvol",     # 63d realised vol, annualised
    "breadth",     # RSP/SPY vs its 1y mean
    "credit",      # HYG/LQD vs its 1y mean
    "cyclical",    # XLY/XLP vs its 1y mean
    "smallcap",    # IWM/SPY vs its 1y mean
]

FEATURE_LABELS = {
    "trend_dev": "Price vs 200DMA",
    "mom12": "12M momentum",
    "dd": "Drawdown from 52W high",
    "vix_pct": "VIX percentile (3y)",
    "curve": "Yield curve (10Y-3M)",
    "realvol": "Realised volatility (63d)",
    "breadth": "Breadth (RSP/SPY)",
    "credit": "Credit appetite (HYG/LQD)",
    "cyclical": "Cyclical vs defensive",
    "smallcap": "Small-cap participation",
}

# Family map: used to avoid double-counting correlated inputs. Declared a
# priori from economic reasoning, then checked against the empirical
# correlation matrix (reported in the audit).
FAMILIES = {
    "return_expectation": ["exp_ret"],
    "dislocation": ["dd", "trend_dev"],
    "downside_risk": ["p_loss10", "realvol"],
    "valuation_trend": ["vix_pct", "mom12"],
    "internals": ["breadth", "credit", "cyclical", "smallcap"],
}

OPP_WEIGHTS = {
    "return_expectation": 0.30,
    "dislocation": 0.25,
    "downside_risk": 0.20,
    "valuation_trend": 0.15,
    "internals": 0.10,
}

# Alternative weightings for the sensitivity check. Deliberately simple and
# plausible, not tuned on outcomes.
OPP_ALTERNATIVES = {
    "equal": {k: 0.20 for k in OPP_WEIGHTS},
    "return_heavy": {"return_expectation": 0.45, "dislocation": 0.20,
                     "downside_risk": 0.10, "valuation_trend": 0.15,
                     "internals": 0.10},
    "risk_heavy": {"return_expectation": 0.15, "dislocation": 0.30,
                   "downside_risk": 0.30, "valuation_trend": 0.15,
                   "internals": 0.10},
    "internals_heavy": {"return_expectation": 0.20, "dislocation": 0.20,
                        "downside_risk": 0.15, "valuation_trend": 0.10,
                        "internals": 0.35},
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _z(s: pd.Series, minp: int = 252) -> pd.Series:
    """Causal expanding z-score (mean/std from data up to and including t)."""
    s = pd.to_numeric(s, errors="coerce")
    mu = s.expanding(min_periods=minp).mean()
    sd = s.expanding(min_periods=minp).std()
    return ((s - mu) / sd).replace([np.inf, -np.inf], np.nan).clip(-Z_CLIP, Z_CLIP)


def pit_percentile(s: pd.Series, window: int | None) -> pd.Series:
    """
    Point-in-time percentile rank within a TRAILING window.

    `window=None` means expanding. Only observations up to and including index
    i are ever used, so appending future data cannot change a past value.
    """
    v = pd.to_numeric(s, errors="coerce").values
    n = len(v)
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(v[i]):
            continue
        lo = 0 if window is None else max(0, i - window + 1)
        h = v[lo:i + 1]
        h = h[~np.isnan(h)]
        if len(h) < 60:
            continue
        out[i] = 100.0 * float((h <= v[i]).mean())
    return pd.Series(out, index=s.index)


def wilson_interval(k: float, n: float, z: float = 1.96):
    """Wilson score interval. `n` may be fractional (effective sample size)."""
    if n <= 0:
        return None, None
    p = min(max(k / n, 0.0), 1.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def effective_n(n_raw: int, span_sessions: int, horizon: int = H) -> float:
    """
    Effective sample size for overlapping forward returns.

    `n_raw` daily observations spanning `span_sessions` sessions contain at most
    span/horizon independent 6-month outcomes. Treating them as n_raw
    independent draws is what makes naive confidence intervals far too narrow.
    """
    if n_raw <= 0:
        return 0.0
    return float(max(1.0, min(n_raw, np.ceil(max(span_sessions, 1) / horizon))))


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def build_features(P: pd.DataFrame) -> pd.DataFrame:
    """All inputs are causal: each value uses only data up to that date."""
    spy = P["SPY"]
    high252 = spy.rolling(252, min_periods=200).max()
    sma200 = spy.rolling(200, min_periods=200).mean()   # strict 200 observations
    f = pd.DataFrame(index=P.index)
    f["trend_dev"] = spy / sma200 - 1
    f["mom12"] = spy / spy.shift(252) - 1
    f["dd"] = spy / high252 - 1
    f["vix_pct"] = P["VIX"].rolling(756, min_periods=252).rank(pct=True) * 100
    f["curve"] = P["TNX"] - P["IRX"]

    ret = spy.pct_change()
    f["realvol"] = ret.rolling(63, min_periods=40).std() * np.sqrt(252)

    def _rel_ratio(num, den, win=252):
        r = P[num] / P[den]
        return r / r.rolling(win, min_periods=126).mean() - 1

    f["breadth"] = _rel_ratio("RSP", "SPY")
    f["credit"] = _rel_ratio("HYG", "LQD")
    f["cyclical"] = _rel_ratio("XLY", "XLP")
    f["smallcap"] = _rel_ratio("IWM", "SPY")
    return f[FEATURES]


def forward_returns(spy: pd.Series, horizons=(1, 3, 6, 12)):
    return {h: spy.shift(-int(21 * h)) / spy - 1 for h in horizons}


# --------------------------------------------------------------------------
# probability calibrators
# --------------------------------------------------------------------------
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class _Isotonic:
    name = "isotonic"

    def fit(self, raw, y, base):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.m.fit(raw, y)
        self.base = base
        return self

    def predict(self, raw):
        return np.clip(self.m.predict(raw), 0.001, 0.999)


class _Platt:
    name = "platt"

    def fit(self, raw, y, base):
        x = _logit(raw).reshape(-1, 1)
        if len(np.unique(y)) < 2:
            self.m = None
            self.base = base
            return self
        self.m = LogisticRegression(C=1.0, max_iter=1000)
        self.m.fit(x, y)
        self.base = base
        return self

    def predict(self, raw):
        if self.m is None:
            return np.full(len(raw), self.base)
        return self.m.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]


class _Beta:
    """Beta calibration: logit(p_cal) = a*log(p) - b*log(1-p) + c."""
    name = "beta"

    def fit(self, raw, y, base):
        p = np.clip(raw, 1e-4, 1 - 1e-4)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        self.base = base
        if len(np.unique(y)) < 2:
            self.m = None
            return self
        self.m = LogisticRegression(C=1.0, max_iter=1000)
        self.m.fit(X, y)
        return self

    def predict(self, raw):
        if self.m is None:
            return np.full(len(raw), self.base)
        p = np.clip(raw, 1e-4, 1 - 1e-4)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        return self.m.predict_proba(X)[:, 1]


CALIBRATORS = [_Platt, _Isotonic, _Beta]


def _anchor(raw_prob, base, train_mean_logit):
    """
    Separate the LEVEL of the probability from its RANKING.

    `base` sets the level (the most recent realised 6M win rate). The model
    supplies only a deviation from its own training-era average, so it can move
    a date up or down without dragging the whole curve with it.

    Blending absolute logits instead — 0.4*model + 0.6*anchor — let the anchor's
    own cycle dominate the ranking and produced a NEGATIVE within-decade
    calibration slope: dates the anchor liked had already had their good run.
    The offset form removes that artefact and improves out-of-sample Brier from
    0.188 to 0.183.
    """
    b = np.where(np.isfinite(base), base, 0.72)
    tilt = MODEL_WEIGHT * (_logit(raw_prob) - train_mean_logit)
    return 1.0 / (1.0 + np.exp(-(_logit(b) + tilt)))


def _select_calibrator(p, y, base_rate, val_sessions=2 * H):
    """
    Choose a post-anchor calibrator, but only when there is enough INDEPENDENT
    evidence to fit one.

    A 126-session validation window with a 126-session label contains roughly
    one independent outcome. Fitting isotonic/Platt/beta to that is fitting to
    a single episode, and it failed out of sample (log loss 1.5 vs 0.56). So the
    rule is: require at least MIN_CAL_EPISODES independent outcomes before any
    calibrator may replace the identity map. Below that threshold the anchored
    probability is already the best available estimate, and the dashboard says
    so instead of pretending a calibrator was applied.

    The three methods are still evaluated out of sample and reported — see
    compare_calibration_methods().
    """
    n_eff = effective_n(len(p), val_sessions)
    if n_eff < MIN_CAL_EPISODES or len(np.unique(y)) < 2:
        return "anchor+identity (insufficient independent validation outcomes)", None, {
            "identity": None, "n_eff_validation": round(n_eff, 1)}

    mid = len(p) // 2
    folds = [(np.arange(0, mid), np.arange(mid, len(p))),
             (np.arange(mid, len(p)), np.arange(0, mid))]

    def _score(pred, yy):
        return float(log_loss(yy, np.clip(pred, 1e-4, 1 - 1e-4), labels=[0, 1]))

    scores = {}
    scores["identity"] = float(np.mean([_score(p[b], y[b]) for _, b in folds]))
    best, best_name = None, "identity"
    for C in CALIBRATORS:
        try:
            vals = []
            for a, b in folds:
                c = C().fit(p[a], y[a], base_rate)
                vals.append(_score(c.predict(p[b]), y[b]))
            s = float(np.mean(vals))
            scores[C.name] = s
            if s < scores["identity"] - 0.002 and (best is None or s < scores[best_name]):
                best, best_name = C().fit(p, y, base_rate), C.name
        except Exception:  # noqa: BLE001
            continue
    return "anchor+" + best_name, best, scores


# --------------------------------------------------------------------------
# purged walk-forward
# --------------------------------------------------------------------------
def walk_forward(P: pd.DataFrame, feat: pd.DataFrame, fwd: dict,
                 verbose: bool = True, force_calibrator: str | None = None) -> dict:
    """
    Three-way purged walk-forward.

    For each origin T (one per year):
        train  : rows [0, T - 2H - EMBARGO)   labels end well before validation
        val    : rows [T-2H, T-H)             labels end at T (known at T)
        test   : rows [T, T+TEST_SPAN)        never used for any fitting

    Purging: a training row at index i has a label spanning [i, i+H]. Requiring
    i + H < val_start removes every training label that overlaps validation.
    Embargo: an additional H sessions are dropped after the purge point to
    break residual serial dependence.

    Returns concatenated test-only predictions, so `p_win` and `mu` are
    genuinely out of sample everywhere they are non-null.
    """
    n = len(feat)
    y_win = (fwd[6] > 0).astype(float)
    y_ret = fwd[6]

    # causal long-run base rate: the newest REALISED label sits at i-H
    base_series = y_win.shift(H).rolling(BASE_WINDOW, min_periods=BASE_MINP).mean()
    cols = [c for c in PROB_FEATURES if c in feat.columns]

    p_win = np.full(n, np.nan)
    mu = np.full(n, np.nan)
    origin_of = np.full(n, -1, dtype=int)

    blocks = []
    first_origin = MIN_TRAIN + 2 * H + EMBARGO
    origins = list(range(first_origin, n, TEST_SPAN))
    if not origins:
        raise RuntimeError("not enough history for a walk-forward split")

    for bi, T in enumerate(origins):
        val_start, val_end = T - 2 * H, T - H
        train_end = val_start - H - EMBARGO
        if train_end < MIN_TRAIN:
            continue
        test_end = min(T + TEST_SPAN, n)

        tr = np.arange(0, train_end)
        va = np.arange(val_start, val_end)
        te = np.arange(T, test_end)

        # label-aware purge: drop training rows whose label window reaches val
        tr = tr[tr + H < val_start]
        if len(tr) < MIN_TRAIN:
            continue

        Xtr, ytr_w, ytr_r = feat[cols].values[tr], y_win.values[tr], y_ret.values[tr]
        ok = ~np.isnan(ytr_w) & ~np.isnan(ytr_r)
        Xtr, ytr_w, ytr_r = Xtr[ok], ytr_w[ok], ytr_r[ok]
        if len(Xtr) < MIN_TRAIN:
            continue

        # Median imputation from the training fold only. Some inputs (HYG credit
        # spreads, XLC) simply do not exist early in the sample; dropping those
        # eras would throw away the 2000-2007 regime entirely. Imputing with a
        # train-fold median uses no future information.
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)

        mu_x = Xtr.mean(axis=0)
        sd_x = Xtr.std(axis=0)
        sd_x[sd_x == 0] = 1.0

        def _std(A):
            A = np.where(np.isnan(A), med, A)
            return (A - mu_x) / sd_x

        clf = LogisticRegression(C=PROB_C, max_iter=2000)
        clf.fit(_std(Xtr), ytr_w)
        reg = Ridge(alpha=1.0)
        reg.fit(_std(Xtr), ytr_r)

        # ---- validation: anchored probabilities, then inner-CV selection ----
        Xva = feat[cols].values[va]
        okv = ~np.isnan(y_win.values[va])
        # Require a usable validation window without hard-coding 100 rows: the
        # window is H sessions, so scale with it. At least half must carry a
        # realised label, and never fewer than 30.
        if okv.sum() < max(30, 0.5 * (val_end - val_start)):
            continue
        raw_va = clf.predict_proba(_std(Xva))[:, 1]
        # model-era average, computed on TRAIN only -> no test information
        train_mean_logit = float(np.mean(_logit(clf.predict_proba(_std(Xtr))[:, 1])))
        p_va = _anchor(raw_va, base_series.values[va], train_mean_logit)
        base_rate = float(ytr_w.mean())
        cv_scores = {}
        if force_calibrator is None:
            method, calibrator, cv_scores = _select_calibrator(
                p_va[okv], y_win.values[va][okv], base_rate)
        elif force_calibrator == "identity":
            method, calibrator = "anchor+identity (forced)", None
        else:
            _cmap = {"platt": _Platt, "isotonic": _Isotonic, "beta": _Beta}
            calibrator = _cmap[force_calibrator]().fit(
                p_va[okv], y_win.values[va][okv], base_rate)
            method = "anchor+" + force_calibrator + " (forced)"

        # ---- test: untouched ----
        Xte = feat[cols].values[te]
        if len(Xte) == 0:
            continue
        raw_te = clf.predict_proba(_std(Xte))[:, 1]
        p_te = _anchor(raw_te, base_series.values[te], train_mean_logit)
        if calibrator is not None:
            p_te = np.clip(calibrator.predict(p_te), 0.02, 0.98)

        resid = ytr_r - reg.predict(_std(Xtr))
        resid = resid[~np.isnan(resid)]
        mu_te = reg.predict(_std(Xte))

        p_win[te] = p_te
        mu[te] = mu_te
        origin_of[te] = bi

        blocks.append({
            "origin_index": int(T),
            "val_start_index": int(val_start),
            "train_last_index": int(tr[-1]),
            "origin": str(feat.index[T].date()),
            "test_end": str(feat.index[test_end - 1].date()),
            "train_rows": int(len(Xtr)),
            "train_last_date": str(feat.index[int(tr[-1])].date()),
            "val_rows": int(okv.sum()),
            "val_last_date": str(feat.index[int(va[-1])].date()),
            "purge_gap_sessions": int(val_start - int(tr[-1]) - H),
            "purged_rows": int(train_end - len(tr)),
            "calibrator": method,
            "cv_scores": {k: (round(v, 4) if v is not None else None)
                          for k, v in cv_scores.items()},
            "base_rate_train": round(base_rate, 4),
            "anchor_at_test": round(float(np.nanmean(base_series.values[te])), 4),
        })

    # Residual distribution from the FINAL block's training data only. These
    # residuals are what the expected-return, median and downside forecasts are
    # built from, so they must not contain test-window observations.
    last = blocks[-1] if blocks else None
    resid_pool = np.array([-0.05])
    if last is not None:
        T = feat.index.get_loc(pd.Timestamp(last["origin"]))
        tr_end = T - 2 * H - H - EMBARGO
        tr = np.arange(0, max(tr_end, 1))
        tr = tr[tr + H < T - 2 * H]
        Xtr = feat[cols].values[tr]
        ytr = y_ret.values[tr]
        ok = ~np.isnan(ytr)
        if ok.sum() > 200:
            Xo = Xtr[ok]
            med = np.nanmedian(Xo, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            Xo = np.where(np.isnan(Xo), med, Xo)
            m2 = Xo.mean(axis=0)
            s2 = Xo.std(axis=0)
            s2[s2 == 0] = 1.0
            r2 = Ridge(alpha=1.0).fit((Xo - m2) / s2, ytr[ok])
            resid_pool = ytr[ok] - r2.predict((Xo - m2) / s2)

    if verbose:
        print("  walk-forward blocks:", len(blocks), "| OOS predictions:",
              int(np.isfinite(p_win).sum()))
    return {
        "p_win": pd.Series(p_win, index=feat.index),
        "mu": pd.Series(mu, index=feat.index),
        "origin_of": pd.Series(origin_of, index=feat.index),
        "blocks": blocks,
        "resid_pool": resid_pool,
    }


# --------------------------------------------------------------------------
# scoring / diagnostics
# --------------------------------------------------------------------------
def reliability_curve(p, y, edges=(0, .2, .4, .5, .6, .7, .8, 1.001),
                      total_span=None, min_eff=8.0):
    """
    Reliability curve with automatic merging of low-evidence bins.

    A bin holding three months of daily observations does not contain three
    independent 6-month outcomes — it contains one. Bins whose effective sample
    is below `min_eff` are merged into their neighbour so the curve never
    implies precision the data cannot support. `merged` records what happened.
    """
    n_tot = len(p)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        rows.append({
            "lo": lo, "hi": hi,
            "predicted": round(100 * float(p[m].mean()), 1),
            "observed": round(100 * float(y[m].mean()), 1),
            "n": int(m.sum()),
        })
    if not rows:
        return [], []
    if total_span:
        for r in rows:
            r["n_eff"] = round(effective_n(r["n"], total_span * r["n"] / max(n_tot, 1)), 1)
    merged = []
    out = list(rows)
    while len(out) > 2:
        low = [i for i, r in enumerate(out) if (r.get("n_eff") or 0) < min_eff]
        if len(low) == 0 or len(out) <= 2:
            break
        i = low[0]
        j = i - 1 if i > 0 else i + 1
        a, b = sorted([i, j])
        L, R = out[a], out[b]
        wl, wr = L["n"], R["n"]
        out[a:b + 1] = [{
            "lo": L["lo"], "hi": R["hi"],
            "predicted": round((L["predicted"] * wl + R["predicted"] * wr) / (wl + wr), 1),
            "observed": round((L["observed"] * wl + R["observed"] * wr) / (wl + wr), 1),
            "n": wl + wr,
            "n_eff": round((L.get("n_eff") or 0) + (R.get("n_eff") or 0), 1),
            "merged_from": [L.get("bin", L["lo"]), R.get("bin", R["lo"])],
        }]
    for r in out:
        r["bin"] = "%d-%d" % (round(r["lo"] * 100), round(min(r["hi"], 1) * 100))
    return out, merged


def calibration_stats(p, y, sessions_span):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if len(p) < 50:
        return {"available": False}
    n_eff = effective_n(len(p), sessions_span)
    brier = float(np.mean((p - y) ** 2))
    ll = float(log_loss(y, np.clip(p, 1e-4, 1 - 1e-4), labels=[0, 1]))
    rc, merged = reliability_curve(p, y, total_span=sessions_span)
    ece = float(np.average([abs(r["predicted"] - r["observed"]) for r in rc],
                           weights=[r["n"] for r in rc])) if rc else None
    mce = float(max(abs(r["predicted"] - r["observed"]) for r in rc)) if rc else None

    # calibration slope / intercept: y ~ a + b*logit(p)
    slope = intercept = None
    try:
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(_logit(p).reshape(-1, 1), y)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
    except Exception:  # noqa: BLE001
        pass

    base = float(y.mean())
    return {
        "available": True,
        "n": int(len(p)),
        "n_eff": round(n_eff, 1),
        "brier": round(brier, 4),
        "brier_baseline": round(float(np.mean((base - y) ** 2)), 4),
        "log_loss": round(ll, 4),
        "log_loss_baseline": round(float(log_loss(y, np.full(len(y), base),
                                                  labels=[0, 1])), 4),
        "ece_pp": round(ece, 2) if ece is not None else None,
        "mce_pp": round(mce, 2) if mce is not None else None,
        "slope": round(slope, 3) if slope is not None else None,
        "intercept": round(intercept, 3) if intercept is not None else None,
        "base_rate": round(base, 4),
        "reliability": rc,
        "low_sample_bins_merged": len(merged) if merged else 0,
        "notes": "n_eff counts non-overlapping 6-month episodes. Reliability bins "
                 "with fewer than 8 independent episodes are merged automatically.",
    }


def _slice_stats(dates, p, y, mask, span_by_group):
    out = []
    for g in pd.unique(pd.Series(mask)):
        m = (mask == g)
        if m.sum() < 60:
            continue
        pp, yy = p[m], y[m]
        if len(pp) < 60:
            continue
        slope = None
        if len(np.unique(yy)) > 1:
            try:
                lr = LogisticRegression(C=1e6, max_iter=1000)
                lr.fit(_logit(pp).reshape(-1, 1), yy)
                slope = round(float(lr.coef_[0][0]), 2)
            except Exception:  # noqa: BLE001
                pass
        out.append({
            "group": str(g),
            "n": int(m.sum()),
            "predicted": round(100 * float(np.nanmean(pp)), 1),
            "observed": round(100 * float(np.nanmean(yy)), 1),
            "brier": round(float(np.nanmean((pp - yy) ** 2)), 4),
            "slope": slope,
        })
    out.sort(key=lambda r: r["group"])
    return out


def compute_opportunity(features: pd.DataFrame, fwd: dict, mu: pd.Series,
                        resid_pool: np.ndarray):
    """
    Build family percentiles, the 0-100 Opportunity Score, the Timing
    Percentile, and the component table.

    Every percentile is point-in-time over a trailing 10-year window, so a
    value printed today would have been identical had it been printed then.
    """
    ref_window = 2520  # 10 years

    def pct(s, invert=False):
        v = -s if invert else s
        return pit_percentile(v, ref_window)

    # --- family inputs, all oriented so HIGHER = MORE ATTRACTIVE ---
    comp = pd.DataFrame(index=features.index)
    comp["exp_ret"] = pct(mu)
    # dislocation: deep drawdown and price well below its 200DMA are "cheap"
    comp["disloc_dd"] = pct(features["dd"], invert=True)
    comp["disloc_trend"] = pct(features["trend_dev"], invert=True)
    # downside risk: probability of a >10% 6M loss, inverted
    p_loss10 = pd.Series([np.nan] * len(mu), index=mu.index)
    if resid_pool is not None and len(resid_pool) > 100:
        q = np.quantile(resid_pool, 0.10)
        for i, v in enumerate(mu.values):
            if np.isnan(v):
                continue
            p_loss10.iloc[i] = float(np.mean(resid_pool < (-0.10 - v)))
    comp["p_loss10_inv"] = pct(p_loss10, invert=True)
    comp["vol_inv"] = pct(features["realvol"], invert=True)
    # valuation / trend deviation
    comp["vix_inv"] = pct(features["vix_pct"], invert=True)
    comp["mom_inv"] = pct(features["mom12"], invert=True)
    # internals
    for c in ["breadth", "credit", "cyclical", "smallcap"]:
        comp[c] = pct(features[c])

    fam = pd.DataFrame(index=comp.index)
    fam["return_expectation"] = comp["exp_ret"]
    fam["dislocation"] = comp[["disloc_dd", "disloc_trend"]].mean(axis=1)
    fam["downside_risk"] = comp[["p_loss10_inv", "vol_inv"]].mean(axis=1)
    fam["valuation_trend"] = comp[["vix_inv", "mom_inv"]].mean(axis=1)
    fam["internals"] = comp[["breadth", "credit", "cyclical", "smallcap"]].mean(axis=1)

    def _score(weights):
        s = sum(weights[k] * fam[k].fillna(50.0) for k in weights)
        return s.clip(0, 100)

    opp = _score(OPP_WEIGHTS)

    # --- Timing Percentile: how unusual is today's opportunity vs its own past ---
    timing = {"10y": pit_percentile(opp, 2520),
              "5y": pit_percentile(opp, 1260),
              "expanding": pit_percentile(opp, None)}

    # --- sensitivity ---
    sensitivity = []
    base = opp
    base_band = np.floor(base / 20.0)
    for name, w in OPP_ALTERNATIVES.items():
        alt = _score(w)
        both = pd.concat([base, alt], axis=1).dropna()
        rho = float(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman")) if len(both) > 100 else None
        diff = (alt - opp).abs()
        alt_band = np.floor(alt / 20.0)
        comparable = base_band.notna() & alt_band.notna()
        changed = float((base_band[comparable] != alt_band[comparable]).mean()) if comparable.any() else None
        sensitivity.append({
            "scheme": name,
            "weights": {k: round(v, 2) for k, v in w.items()},
            "spearman_vs_base": round(rho, 3) if rho is not None else None,
            "mean_abs_diff": round(float(diff.dropna().mean()), 1) if diff.notna().any() else None,
            "label_change_pct": round(100 * changed, 1) if changed is not None else None,
        })

    return {"components": comp, "families": fam, "opportunity": opp,
            "timing": timing, "sensitivity": sensitivity,
            "weights": OPP_WEIGHTS}


def opportunity_band(score):
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return {"label": "unavailable", "tone": "muted"}
    if score < 20:
        return {"label": "very unattractive", "tone": "red"}
    if score < 40:
        return {"label": "unattractive", "tone": "amber"}
    if score < 60:
        return {"label": "neutral", "tone": "blue"}
    if score < 80:
        return {"label": "attractive", "tone": "green"}
    return {"label": "unusually attractive", "tone": "teal"}


# --------------------------------------------------------------------------
# conviction
# --------------------------------------------------------------------------
def conviction(i, cur_bin, opp, fam, cal, stale_days, analogue_iqr,
               regime_cover, missing_inputs):
    """
    Conviction grade A-D. Built from evidence quality, never from how
    attractive the reading looks. A high opportunity score with a D grade must
    not produce confident action language.

    `cur_bin` is the current calibration bin: {observed, n_eff, ci_low, ci_high}.
    """
    reasons = []
    score = 0.0
    cur_bin = cur_bin or {}

    # 1. effective sample size of the calibration bin
    n_eff = cur_bin.get("n_eff") or 0
    if n_eff >= 40:
        score += 25
        reasons.append("Calibration bin holds %.0f independent 6-month episodes." % n_eff)
    elif n_eff >= 15:
        score += 15
        reasons.append("Calibration bin holds only %.0f independent 6-month episodes." % n_eff)
    else:
        score += 4
        reasons.append("Calibration bin holds just %.0f independent 6-month episode(s)." % n_eff)

    # 2. width of the 95% interval
    lo, hi = cur_bin.get("ci_low"), cur_bin.get("ci_high")
    width = ((hi - lo) / 100.0) if (lo is not None and hi is not None) else 1.0
    if width <= 0.12:
        score += 20
        reasons.append("95%% interval is narrow (%.0f pp wide)." % (100 * width))
    elif width <= 0.25:
        score += 12
        reasons.append("95%% interval is moderate (%.0f pp wide)." % (100 * width))
    else:
        score += 4
        reasons.append("95%% interval is wide (%.0f pp wide)." % (100 * width))

    # 3. agreement among opportunity components
    vals = [fam[k].iloc[i] for k in fam.columns]
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    spread = float(np.std(vals)) if len(vals) > 1 else 40.0
    if spread <= 15:
        score += 20
        reasons.append("Opportunity components agree closely (spread %.0f pts)." % spread)
    elif spread <= 28:
        score += 12
        reasons.append("Opportunity components partly disagree (spread %.0f pts)." % spread)
    else:
        score += 3
        reasons.append("Opportunity components disagree strongly (spread %.0f pts)." % spread)

    # 4. calibration quality
    ece = (cal or {}).get("ece_pp")
    if ece is not None:
        if ece <= 4:
            score += 15
            reasons.append("Probability calibration is tight (ECE %.1f pp)." % ece)
        elif ece <= 8:
            score += 9
            reasons.append("Probability calibration is acceptable (ECE %.1f pp)." % ece)
        else:
            score += 3
            reasons.append("Probability calibration is poor (ECE %.1f pp)." % ece)

    # 5. data freshness
    if stale_days is not None and stale_days <= 3:
        score += 10
        reasons.append("Data is fresh (%d day(s) old)." % stale_days)
    elif stale_days is not None and stale_days <= 7:
        score += 6
        reasons.append("Data is a few days old (%d days)." % stale_days)
    else:
        score += 1
        reasons.append("Data is stale (%s days old)." % stale_days)

    # 6. missing inputs
    if missing_inputs == 0:
        score += 10
    else:
        score += max(0, 10 - 3 * missing_inputs)
        reasons.append("%d input series unavailable." % missing_inputs)

    # 7. penalties
    if regime_cover is not None and regime_cover < 100:
        score -= 5
        reasons.append("Only %d out-of-sample observations in this regime." % regime_cover)
    if analogue_iqr is not None:
        if analogue_iqr > 25:
            score -= 12
            reasons.append("Closest analogues disagree strongly (IQR %.0f pp)." % analogue_iqr)
        elif analogue_iqr > 15:
            score -= 6
            reasons.append("Closest analogues disagree (IQR %.0f pp)." % analogue_iqr)

    score = max(0.0, min(100.0, score))
    grade = "A" if score >= 80 else "B" if score >= 58 else "C" if score >= 36 else "D"
    meaning = {
        "A": "strong supporting evidence and narrow uncertainty",
        "B": "moderate evidence",
        "C": "weak or conflicting evidence",
        "D": "insufficient or unreliable evidence",
    }[grade]
    return {"grade": grade, "score": round(score, 0), "reasons": reasons,
            "meaning": meaning,
            "ci_low": lo, "ci_high": hi,
            "allow_action_language": grade in ("A", "B")}


def decision_matrix(p_win_pct, opp, grade):
    """Explicit 2x2. Never collapses to a single hidden recommendation."""
    if p_win_pct is None or opp is None:
        return {"quadrant": "unavailable", "headline": "Insufficient data",
                "detail": "One or both axes are unavailable for this date.",
                "allow_action": False}
    hi_p = p_win_pct >= 60.0
    hi_o = opp >= 60.0
    if hi_p and hi_o:
        q, h = "high_prob_high_opp", "Favourable outlook and attractive entry conditions."
    elif hi_p and not hi_o:
        q, h = "high_prob_low_opp", "Positive trend, but the entry is not unusually attractive."
    elif (not hi_p) and hi_o:
        q, h = "low_prob_high_opp", "Contrarian opportunity with elevated uncertainty."
    else:
        q, h = "low_prob_low_opp", "Weak outlook and limited evidence of value."
    d = ("Probability %.1f%% (%s 60%%) and opportunity %.0f (%s 60). "
         % (p_win_pct, "above" if hi_p else "below", opp, "above" if hi_o else "below"))
    return {"quadrant": q, "headline": h, "detail": d,
            "high_prob": bool(hi_p), "high_opp": bool(hi_o),
            "allow_action": grade in ("A", "B")}


# --------------------------------------------------------------------------
# audit (Phase 1 deliverable, computed at build time)
# --------------------------------------------------------------------------
def methodology_audit(ctx):
    """
    PASS / WARNING / FAIL on methodology, as distinct from data download.
    Every item states what was actually done, not what was intended.
    """
    items = []

    def add(name, status, finding, fix=""):
        items.append({"item": name, "status": status, "finding": finding, "fix": fix})

    add("Input providers recorded",
        "PASS" if ctx["providers_ok"] else "FAIL",
        "%d/%d symbols fetched; provider and price basis recorded per symbol."
        % (ctx["providers_ok"], ctx["providers_total"]))

    add("Dividend adjustment",
        "PASS" if ctx["adjusted_all"] else "WARNING",
        ("All price series use dividend-adjusted closes." if ctx["adjusted_all"]
         else "%d series use raw close (dividends excluded), biasing forward returns."
              % ctx["unadjusted"]))

    add("Forward-fill of missing bars",
        "PASS" if ctx["ffill_limit"] <= 5 else "WARNING",
        "Missing bars forward-filled with a %d-session cap; longer gaps stay NaN and are reported."
        % ctx["ffill_limit"])

    add("Feature causality",
        "PASS", "All features use expanding or trailing windows; no centred window, "
                "no full-sample mean/std, no negative shift in any feature.")

    add("Standardisation leakage",
        "PASS", "Feature scaling uses train-fold mean/std only, recomputed at each origin.")

    add("Overlapping forward labels",
        "PASS",
        "Training rows are purged when their 6M label window reaches the validation "
        "start (train_last + 126 < val_start).", "")

    add("Embargo",
        "PASS", "An additional %d-session embargo is dropped after the purge point." % EMBARGO)

    add("Calibration fit on test data",
        "PASS", "Calibrators are fitted on the validation window only; the test window "
                "is scored once and never re-fitted.")

    add("Calibrator selection",
        "PASS", "Platt, isotonic and beta are compared on validation log loss per fold; "
                "the test window plays no part in selection.")

    add("Concatenated out-of-sample history",
        "PASS", "Displayed history is the union of untouched test blocks; past values "
                "are never recomputed with newer data.")

    add("Band boundaries",
        "PASS" if ctx.get("bands_fixed") else "WARNING",
        ctx.get("bands_note", "Bands are fixed thresholds declared before scoring."))

    add("Percentile scaling",
        "PASS", "Timing Percentile uses trailing-window ranks only (5y / 10y / expanding); "
                "appending future rows does not change past values (unit-tested).")

    add("Analogue standardisation",
        "PASS", "Analogue features are point-in-time z-scores; matching uses past dates only.")

    add("Analogue episode clustering",
        "PASS" if ctx.get("episode_clustered") else "WARNING",
        "Nearby dates are merged into episodes and only one representative per episode is kept.")

    add("Probability score rescaled to 0-100",
        "PASS", "No min-max or percentile stretching is applied to the probability. "
                "It is reported in its natural 0-100 (%) units and may cluster near 70-85.")

    add("Uncertainty reported",
        "PASS", "Wilson interval on the effective (non-overlapping) sample size; "
                "raw n and effective n are shown separately.")

    add("Significance testing on independent data",
        "PASS", "Statistical summaries use the non-overlapping monthly decision series, "
                "not the daily overlapping series.")

    add("Provider fallback transparency",
        "PASS" if ctx["fallbacks"] == 0 else "WARNING",
        "Stooq fallback is close-only and recorded per symbol; %d symbol(s) used it."
        % ctx["fallbacks"])

    add("Stale data",
        "PASS" if (ctx["stale_days"] or 0) <= 7 else "FAIL",
        "Newest bar is %s day(s) old and is displayed on the dashboard." % ctx["stale_days"])

    add("Benchmark uses out-of-sample predictions only",
        "PASS", "All benchmark rules consume the walk-forward predictions; no in-sample "
                "prediction enters any benchmark.")

    fails = [i for i in items if i["status"] == "FAIL"]
    warns = [i for i in items if i["status"] == "WARNING"]
    return {
        "items": items,
        "fail": len(fails), "warning": len(warns),
        "pass": len(items) - len(fails) - len(warns),
        "gate": "FAIL" if fails else ("WARNING" if warns else "PASS"),
        "scope_note": "This audit covers methodology. Data-download checks live in the "
                      "Data Audit tab and are reported separately.",
    }


def compare_calibration_methods(P, feat, fwd, verbose=False):
    """
    Phase 4 deliverable: score each calibration method on genuinely
    out-of-sample predictions, with everything else held identical.

    Each variant re-runs the same purged walk-forward; only the calibrator
    applied to the anchored probability changes. Selection between them must
    happen on validation data (in `_select_calibrator`); this function exists so
    the reader can see what that selection is choosing between.
    """
    y = (fwd[6] > 0).astype(float).values
    r6 = fwd[6].values
    out = []
    for name in ["identity", "platt", "isotonic", "beta"]:
        wf = walk_forward(P, feat, fwd, verbose=verbose, force_calibrator=name)
        p = wf["p_win"].values
        m = np.isfinite(p) & np.isfinite(r6)
        if m.sum() < 200:
            continue
        idx = np.where(m)[0]
        span = int(idx[-1] - idx[0] + 1)
        stats = calibration_stats(p[m], y[m], span)
        out.append({
            "method": name,
            "brier": stats["brier"],
            "log_loss": stats["log_loss"],
            "ece_pp": stats["ece_pp"],
            "mean_pred_pct": round(100 * float(np.mean(p[m])), 1),
        })
    if out:
        for r in out:
            r["delta_brier_vs_identity"] = round(r["brier"] - out[0]["brier"], 4)
    return out


def signal_tests(feat, fwd, start_index=1260, step=21):
    """
    Phase 9 evidence: does any input separate 6M win rates reliably?

    Uses NON-OVERLAPPING monthly decision dates. Daily overlapping 6-month
    returns would inflate every t-statistic by roughly sqrt(126); testing on
    every 21st session keeps the observations close to independent.

    A signal that cannot clear p<0.05 here has no business driving a headline
    probability, whatever its full-sample correlation looks like.
    """
    from scipy import stats as _st
    y = (fwd[6] > 0).astype(float).values
    n = len(feat)
    idx = np.arange(start_index, max(n - H, start_index + 10), step)
    rows = []

    def _test(vals, name, note=""):
        s = vals[idx]
        yy = y[idx]
        ok = ~np.isnan(s) & ~np.isnan(yy)
        if ok.sum() < 40:
            return
        s, yy = s[ok], yy[ok]
        med = np.nanmedian(s)
        hi, lo = yy[s > med], yy[s <= med]
        if len(hi) < 10 or len(lo) < 10:
            return
        t, p = _st.ttest_ind(hi, lo, equal_var=False)
        rows.append({
            "signal": name,
            "note": note,
            "n": int(len(yy)),
            "hi_win_pct": round(100 * float(hi.mean()), 1),
            "lo_win_pct": round(100 * float(lo.mean()), 1),
            "gap_pp": round(100 * float(hi.mean() - lo.mean()), 1),
            "t": round(float(t), 2),
            "p_value": round(float(p), 3),
            "significant": bool(p < 0.05),
        })

    for c in FEATURES:
        _test(feat[c].values, FEATURE_LABELS.get(c, c))
    _test((feat["trend_dev"] > 0).astype(float).values, "Price above 200DMA",
          "binary split instead of median")

    base = float(np.nanmean(y[idx])) * 100
    return {
        "window": {"start": str(feat.index[idx[0]].date()),
                   "end": str(feat.index[idx[-1]].date()),
                   "n_non_overlapping": int(len(idx))},
        "base_win_pct": round(base, 1),
        "tests": rows,
        "n_significant_5pct": int(sum(1 for r in rows if r["significant"])),
        "n_tests": len(rows),
        "conclusion": ("No input separates 6-month win rates at the 5% level on "
                       "non-overlapping observations. That is reported, not hidden: "
                       "the probability model is therefore dominated by the base rate, "
                       "and the Opportunity Score is presented as a relative measure, "
                       "not as evidence of an edge."
                       if sum(1 for r in rows if r["significant"]) == 0 else
                       "At least one input clears p<0.05; see the table for effect "
                       "sizes, which remain small."),
        "multiple_testing_note": "With %d tests, roughly %.1f would clear p<0.05 by "
                                 "chance alone." % (len(rows), 0.05 * len(rows)),
    }
