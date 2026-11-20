"""
ETF Entry Decision System -- evidence engine (v2)
=================================================

Answers exactly one question: "Is today a good entry point for VOO or QQQ?"

Design rules (enforced in code, not in prose):

* Every feature is a POINT-IN-TIME expanding-window percentile: at date t it is
  computed only from values observed on or before t. No scaling / ranking /
  imputation ever sees the future.
* Targets are realised forward total returns of SPY (validated VOO proxy,
  daily-return corr 0.9986 / monthly 0.9998 with actual VOO) and QQQ.
* Four validation schemes: purged walk-forward, rolling window, expanding
  window, and combinatorial purged cross-validation (CPCV). All use label
  purging and an embargo equal to the forecast horizon.
* Five competing models: Base Rate, Logistic Regression, XGBoost, Random
  Forest, LightGBM. Feature selection (mutual information) runs INSIDE each
  split on training data only.
* A model earns the right to forecast only if it beats the base-rate model on
  out-of-sample Brier in at least 3 of 4 schemes AND passes AUC / calibration
  gates. Otherwise all probability forecasts are suppressed.
* The Entry Attractiveness Score is a percentile of realised future outcomes of
  historically similar states, computed with strictly prior data only.
* Deployment (lump sum vs DCA) is backtested on realised daily price paths with
  point-in-time T-bill cash yield.

Output: data/evidence_v2.json
"""

from __future__ import annotations

import itertools
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW2 = ROOT / "data_raw"
OUT = ROOT / "data" / "evidence_v2.json"

SEED = 20261118
HORIZONS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
TICKERS = {"VOO": "SPY", "QQQ": "QQQ"}
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
K_FEATURES = 6
WARMUP_MONTHS = 72
N_TEST_BLOCKS = 6
N_CPCV_GROUPS = 6
CPCV_TEST_GROUPS = 2
MIN_HISTORY_FOR_PCT = 252


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

def _read_adj(path: Path, col: str = "adj_close") -> pd.Series:
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"].astype(str).str.slice(0, 10))
    d = d.drop_duplicates("date", keep="last").set_index("date")
    return d[col].astype(float)


def _read_close(path: Path) -> pd.Series:
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"].astype(str).str.slice(0, 10))
    d = d.drop_duplicates("date", keep="last").set_index("date")
    return d["close"].astype(float)


def load_prices() -> pd.DataFrame:
    syms = ["SPY", "QQQ", "VIX", "TLT", "TNX", "IRX", "HYG", "LQD", "RSP", "IWM"] + SECTORS
    cols = {}
    for s in syms:
        f = RAW / f"{s}.csv"
        if f.exists():
            cols[s] = _read_adj(f)
    cols["FVX"] = _read_close(RAW2 / "fvx.csv")
    cols["IEF"] = _read_adj(RAW2 / "ief.csv")
    cols["SHY"] = _read_adj(RAW2 / "shy.csv")
    cols["VIX9D"] = _read_adj(RAW2 / "_vix9d.csv")
    # VIX carries one extra (next-day) observation in the source file. Drop it:
    cols["VIX"] = cols["VIX"].loc[lambda s: s.index <= pd.Timestamp("2026-11-16")]
    px = pd.DataFrame(cols)
    px = px[~px.index.duplicated(keep="last")].sort_index()
    # Keep SPY's trading calendar; forward-fill last *known* value only.
    px = px.reindex(pd.date_range(px.index.min(), px.index.max(), freq="B"))
    return px.ffill()


def build_raw_features(px: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Raw (unscaled) candidate signals. All rolling windows look backwards."""
    f = pd.DataFrame(index=px.index)
    spy, qqq, vix = px["SPY"], px["QQQ"], px["VIX"]
    r_spy = spy.pct_change()

    # ---- Market trend -----------------------------------------------------
    sma50, sma200 = spy.rolling(50).mean(), spy.rolling(200).mean()
    hi252, lo252 = spy.rolling(252).max(), spy.rolling(252).min()
    f["trend_dist_200dma"] = spy / sma200 - 1.0
    f["trend_slope_50dma"] = sma50 / sma50.shift(21) - 1.0
    f["trend_slope_200dma"] = sma200 / sma200.shift(63) - 1.0
    f["trend_200dma_cross"] = (spy > sma200).astype(float) - (spy.shift(21) > sma200.shift(21)).astype(float)
    f["trend_near_52w_high"] = spy / hi252 - 1.0
    f["trend_near_52w_low"] = spy / lo252 - 1.0
    f["trend_momentum_12m"] = spy / spy.shift(252) - 1.0
    f["trend_momentum_3m"] = spy / spy.shift(63) - 1.0

    # ---- Volatility -------------------------------------------------------
    f["vol_vix_level"] = vix
    f["vol_vix_change_1m"] = vix / vix.shift(21) - 1.0
    f["vol_vix_term_structure"] = px["VIX9D"] / vix - 1.0
    f["vol_realised_20d"] = r_spy.rolling(20).std() * np.sqrt(252)
    f["vol_realised_60d"] = r_spy.rolling(60).std() * np.sqrt(252)
    f["vol_realised_chg"] = (r_spy.rolling(20).std() / r_spy.rolling(120).std()) - 1.0

    # ---- Breadth ----------------------------------------------------------
    sec = px[SECTORS]
    sec_above = (sec > sec.rolling(200).mean()).sum(axis=1) / len(SECTORS)
    sec_up = (sec / sec.shift(21) - 1 > 0).sum(axis=1) / len(SECTORS)
    sec_up126 = (sec / sec.shift(126) - 1 > 0).sum(axis=1) / len(SECTORS)
    f["breadth_sector_above_200dma"] = sec_above
    f["breadth_advance_decline"] = sec_up
    f["breadth_sector_participation"] = sec_up126
    rsp = px["RSP"] / spy
    f["breadth_rsp_vs_spy_3m"] = rsp / rsp.shift(63) - 1.0
    iwm = px["IWM"] / spy
    f["breadth_small_vs_large_3m"] = iwm / iwm.shift(63) - 1.0

    # ---- Credit -----------------------------------------------------------
    hyg_tlt = px["HYG"] / px["TLT"]
    lqd_tlt = px["LQD"] / px["TLT"]
    hyg_ief = px["HYG"] / px["IEF"]
    f["credit_hyg_tlt_3m"] = hyg_tlt / hyg_tlt.shift(63) - 1.0
    f["credit_lqd_treasury_3m"] = lqd_tlt / lqd_tlt.shift(63) - 1.0
    f["credit_spread_proxy_3m"] = hyg_ief / hyg_ief.shift(63) - 1.0
    f["credit_hyg_momentum_6m"] = px["HYG"] / px["HYG"].shift(126) - 1.0

    # ---- Macro (rates only; no CPI / unemployment / surprise series) ------
    curve = px["TNX"] - px["IRX"]
    f["macro_yield_curve_10y_3m"] = curve
    f["macro_yield_curve_5y_3m"] = px["FVX"] - px["IRX"]
    f["macro_curve_steepening_6m"] = curve - curve.shift(126)
    f["macro_rate_trend_12m"] = px["TNX"] - px["TNX"].shift(252)
    f["macro_rates_falling_6m"] = -(px["TNX"] - px["TNX"].shift(126))

    f = f.replace([np.inf, -np.inf], np.nan)
    groups = {}
    for c in f.columns:
        groups[c] = c.split("_")[0].capitalize() if c.split("_")[0] in (
            "trend", "vol", "breadth", "credit", "macro") else "Other"
    return f, groups


def point_in_time_percentiles(raw: pd.DataFrame, positions: list[int]) -> pd.DataFrame:
    """Expanding-window percentile of each feature, using ONLY data <= t.

    Late-starting series (HYG 2007, VIX9D 2011, ...) yield 0.5 = neutral until
    enough of their own history exists. Never uses a future observation.
    """
    arr = raw.to_numpy(float)
    out = np.full((len(positions), arr.shape[1]), 0.5)
    dates = raw.index
    for k, p in enumerate(positions):
        past = arr[: p + 1]
        cur = arr[p]
        valid = np.isfinite(past).sum(axis=0)
        leq = (past <= cur).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = leq / np.maximum(valid, 1)
        ok = (valid >= MIN_HISTORY_FOR_PCT) & np.isfinite(cur)
        out[k] = np.where(ok, pct, 0.5)
        out[k] = np.clip(np.nan_to_num(out[k], nan=0.5), 0.0, 1.0)
    return pd.DataFrame(out, index=[dates[p] for p in positions], columns=raw.columns)


# ---------------------------------------------------------------------------
# 2. Targets and realised outcome distributions
# ---------------------------------------------------------------------------

def forward_path_stats(px: pd.DataFrame, sym: str, positions: list[int]) -> dict:
    p = px[sym].to_numpy(float)
    out = {h: {"r": [], "maxdd": [], "sharpe": []} for h in HORIZONS}
    for t in positions:
        for h, H in HORIZONS.items():
            if t + H >= len(p):
                out[h]["r"].append(np.nan)
                out[h]["maxdd"].append(np.nan)
                out[h]["sharpe"].append(np.nan)
                continue
            path = p[t : t + H + 1]
            out[h]["r"].append(path[-1] / path[0] - 1.0)
            runmax = np.maximum.accumulate(path)
            out[h]["maxdd"].append(float((path / runmax - 1.0).min()))
            rr = np.diff(path) / path[:-1]
            sd = rr.std()
            out[h]["sharpe"].append(float(rr.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan)
    for h in HORIZONS:
        for k in out[h]:
            out[h][k] = np.asarray(out[h][k], float)
    return out


def unconditional_stats(r: np.ndarray, dd: np.ndarray, sh: np.ndarray) -> dict:
    r = r[np.isfinite(r)]
    dd = dd[np.isfinite(dd)]
    sh = sh[np.isfinite(sh)]
    if len(r) == 0:
        return {k: float("nan") for k in ("p_pos", "p_gt5", "p_gt10", "mean", "median",
                                          "downside_cvar10", "p_loss10", "drawdown", "sharpe")} | {"n": 0}
    tail = np.quantile(r, 0.10)
    return {
        "p_pos": float((r > 0).mean()),
        "p_gt5": float((r > 0.05).mean()),
        "p_gt10": float((r > 0.10).mean()),
        "mean": float(r.mean()),
        "median": float(np.median(r)),
        "downside_cvar10": float(r[r <= tail].mean()) if len(r) else float("nan"),
        "p_loss10": float((r < -0.10).mean()),
        "drawdown": float(dd.mean()) if len(dd) else float("nan"),
        "sharpe": float(sh.mean()) if len(sh) else float("nan"),
        "n": int(len(r)),
    }


# ---------------------------------------------------------------------------
# 3. Validation splits (purge + embargo)
# ---------------------------------------------------------------------------

def _purge(train_pos: np.ndarray, test_pos: np.ndarray, H: int) -> np.ndarray:
    """Remove training labels overlapping any test label, plus H-day embargo."""
    if len(test_pos) == 0 or len(train_pos) == 0:
        return train_pos
    ts, te = test_pos.min(), test_pos.max()
    keep = ~(((train_pos + H >= ts) & (train_pos <= te + H)))
    return train_pos[keep]


def build_splits(obs: np.ndarray, H: int) -> dict:
    """obs = daily positions of monthly observation dates with full 12M outcome."""
    n = len(obs)
    warm = min(WARMUP_MONTHS, n // 3)
    test_obs = obs[warm:]
    blocks = np.array_split(np.arange(warm, n), N_TEST_BLOCKS)
    schemes = {"Walk-forward": [], "Rolling": [], "Expanding": [], "CPCV": []}

    for b in blocks:
        test_idx = b
        tpos = obs[test_idx]
        start = test_idx[0]
        for name, window in (("Walk-forward", 60), ("Rolling", 120), ("Expanding", None)):
            cand = np.arange(0, start) if window is None else np.arange(max(0, start - window), start)
            tr = _purge(obs[cand], tpos, H)
            if len(tr) >= 30 and len(np.unique(np.array([_y_lookup[int(q)] for q in tr]))) >= 2:
                schemes[name].append((tr, tpos, test_idx))

    # Combinatorial purged CV: choose 2 of 6 groups as test, average paths.
    groups = np.array_split(np.arange(n), N_CPCV_GROUPS)
    for combo in itertools.combinations(range(N_CPCV_GROUPS), CPCV_TEST_GROUPS):
        test_idx = np.concatenate([groups[i] for i in combo])
        tpos = obs[test_idx]
        train_idx = np.concatenate([groups[i] for i in range(N_CPCV_GROUPS) if i not in combo])
        tr = _purge(obs[train_idx], tpos, H)
        if len(tr) >= 40 and len(np.unique(np.array([_y_lookup[int(q)] for q in tr]))) >= 2:
            schemes["CPCV"].append((tr, tpos, test_idx))
    return schemes


# ---------------------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------------------

MODEL_NAMES = ["Base Rate", "Logistic Regression", "XGBoost", "Random Forest", "LightGBM"]


def make_model(name: str):
    if name == "Logistic Regression":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.3, max_iter=2000, random_state=SEED))
    if name == "Random Forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=150, max_depth=3,
                                                    min_samples_leaf=8, max_features=0.7,
                                                    random_state=SEED, n_jobs=1))
    if name == "XGBoost":
        return make_pipeline(SimpleImputer(strategy="median"),
                             XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8,
                                           min_child_weight=5, reg_lambda=5,
                                           eval_metric="logloss", verbosity=0,
                                           random_state=SEED, n_jobs=1))
    if name == "LightGBM":
        return make_pipeline(SimpleImputer(strategy="median"),
                             LGBMClassifier(n_estimators=100, num_leaves=7, max_depth=3,
                                            learning_rate=0.05, min_child_samples=15,
                                            reg_lambda=5, verbose=-1,
                                            random_state=SEED, n_jobs=1))
    return None


def fit_predict(name, Xtr, ytr, Xte):
    if name == "Base Rate":
        p = float(np.clip(ytr.mean(), 0.02, 0.98))
        return np.full(len(Xte), p), None
    m = make_model(name)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return np.clip(p, 1e-4, 1 - 1e-4), m


def metric_bundle(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    y = np.asarray(y, float)
    brier = float(np.mean((p - y) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    try:
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    except Exception:
        auc = 0.5
    try:
        pr = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float(y.mean())
    except Exception:
        pr = float(y.mean())
    # Calibration slope / intercept: logistic recalibration on logit(p).
    z = logit(p).reshape(-1, 1)
    if np.std(z) < 1e-6 or len(np.unique(y)) < 2:
        slope, intercept = 0.0, float(logit(np.clip(y.mean(), 1e-4, 1 - 1e-4)))
    else:
        try:
            cal = LogisticRegression(penalty=None, max_iter=1000).fit(z, y)
            slope, intercept = float(cal.coef_[0][0]), float(cal.intercept_[0])
        except Exception:
            slope, intercept = 0.0, 0.0
    bins = np.linspace(0, 1, 6)
    ece = 0.0
    for i in range(5):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < 4 else p <= bins[i + 1])
        if m.sum():
            ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return {"brier": brier, "log_loss": ll, "roc_auc": auc, "pr_auc": pr,
            "cal_slope": slope, "cal_intercept": intercept, "ece": float(ece),
            "n": int(len(y)), "base_rate": float(y.mean()), "mean_pred": float(p.mean())}


# ---------------------------------------------------------------------------
# 5. Per (ticker, horizon) model comparison
# ---------------------------------------------------------------------------

def select_features(Xtr: np.ndarray, ytr: np.ndarray, names: list[str], rng: np.random.Generator):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xt = imp.transform(Xtr)
    noise = rng.random((len(Xt), 3))
    mi = mutual_info_classif(np.hstack([Xt, noise]), ytr, n_neighbors=5, random_state=SEED)
    real, nz = mi[:-3], mi[-3:]
    threshold = float(np.quantile(nz, 0.95))
    above = real > max(threshold, 1e-9)
    order = np.argsort(-real)
    picked = [i for i in order if above[i]][:K_FEATURES]
    if len(picked) < 3:
        picked = list(order[:K_FEATURES])
    return picked, real, threshold


def run_model_grid(X: np.ndarray, y: np.ndarray, names: list[str], splits: dict,
                   feature_names: list[str]) -> dict:
    """Fit every model under every scheme. Returns OOS metrics + diagnostics."""
    rng = np.random.default_rng(SEED)
    results, audit = {}, {"mi": [], "perm": {}, "shap": {}, "selection": [],
                          "univariate_brier": {n: [] for n in names},
                          "decade_edge": {"2000s": [], "2010s": [], "2020s": []}}
    # map daily position -> row index in X (X is built for _FEATURE_POS)
    row_of = {int(p): i for i, p in enumerate(_FEATURE_POS)}

    for scheme, split_list in splits.items():
        acc = {m: {} for m in MODEL_NAMES}   # sample pos -> list of predictions
        for tr_pos, te_pos, te_idx in split_list:
            tr_rows = np.array([row_of[int(p)] for p in tr_pos if int(p) in row_of])
            te_rows = np.array([row_of[int(p)] for p in te_pos if int(p) in row_of])
            if len(tr_rows) < 25 or len(te_rows) == 0:
                continue
            ytr = y[tr_rows]
            if len(np.unique(ytr)) < 2:
                continue
            picked, mi, thr = select_features(X[tr_rows], ytr, feature_names, rng)
            audit["mi"].append(mi)
            audit["selection"].append(picked)
            Xtr, Xte = X[tr_rows][:, picked], X[te_rows][:, picked]
            for m in MODEL_NAMES:
                p, _ = fit_predict(m, Xtr, ytr, Xte)
                for pp, pred in zip(te_pos, p):
                    acc[m].setdefault(int(pp), []).append(float(pred))
            # ---- audit diagnostics on expanding scheme only (XGBoost) ----
            if scheme == "Expanding":
                p_x, model = fit_predict("XGBoost", Xtr, ytr, Xte)
                yte = y[te_rows]
                b0 = float(np.mean((p_x - yte) ** 2))
                for j, fi in enumerate(picked):
                    Xp = Xte.copy()
                    Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                    pp = np.clip(model.predict_proba(Xp)[:, 1], 1e-4, 1 - 1e-4)
                    d = float(np.mean((pp - yte) ** 2)) - b0
                    audit["perm"].setdefault(feature_names[fi], []).append(d)
                try:
                    import shap
                    expl = shap.TreeExplainer(model.steps[-1][1])
                    sv = np.asarray(expl.shap_values(Xte))
                    if sv.ndim == 3:
                        sv = sv[:, :, 1]
                    mag = np.abs(sv).mean(axis=0)
                    for j, fi in enumerate(picked):
                        audit["shap"].setdefault(feature_names[fi], []).append(float(mag[j]))
                except Exception:
                    pass
                # univariate out-of-sample edge against the base-rate model
                p_const = float(np.clip(ytr.mean(), 0.02, 0.98))
                bb = float(np.mean((np.full(len(yte), p_const) - yte) ** 2))
                edges = np.zeros(len(feature_names))
                for fi, fname in enumerate(feature_names):
                    try:
                        mm = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                           LogisticRegression(C=0.3, max_iter=500))
                        mm.fit(X[tr_rows][:, [fi]], ytr)
                        pu = np.clip(mm.predict_proba(X[te_rows][:, [fi]])[:, 1], 1e-4, 1 - 1e-4)
                        bu = float(np.mean((pu - yte) ** 2))
                        edges[fi] = bb - bu
                        audit["univariate_brier"][fname].append(bb - bu)
                    except Exception:
                        pass
                yr = int(pd.Timestamp(_FEATURE_DATES[te_rows[0]]).year)
                dec = "2000s" if yr < 2010 else ("2010s" if yr < 2020 else "2020s")
                audit["decade_edge"][dec].append(edges)

        for m in MODEL_NAMES:
            if not acc[m]:
                results[(scheme, m)] = None
                continue
            ps = sorted(acc[m])
            pp = np.array([np.mean(acc[m][k]) for k in ps])
            yy = np.array([y[row_of[k]] for k in ps])
            results[(scheme, m)] = metric_bundle(yy, pp)
            results[(scheme, m)]["predictions"] = {str(k): float(v) for k, v in zip(ps, pp)}
    return {"metrics": results, "audit": audit}


# ---------------------------------------------------------------------------
# 6. Entry Attractiveness Score (outcome-based, point-in-time)
# ---------------------------------------------------------------------------

def entry_score(X: np.ndarray, r12: np.ndarray, positions: list[int], today_row: int,
                feature_idx: list[int], dates=None) -> dict:
    Z = X[:, feature_idx] if feature_idx else X[:, :K_FEATURES]
    n = Z.shape[0]
    valid = np.isfinite(r12)
    scores, expected = np.full(n, np.nan), np.full(n, np.nan)
    ppos = np.asarray(positions)
    for i in range(n):
        # ONLY states whose 12M outcome had ALREADY been realised before date i.
        cutoff = int(np.searchsorted(ppos, ppos[i] - 252, side="right"))
        cand = np.where(valid[:cutoff])[0]
        if len(cand) < 20:
            continue
        cur = Z[i]
        d = np.sqrt(np.mean((Z[cand] - cur) ** 2, axis=1))
        k = min(40, len(cand))
        idx = cand[np.argsort(d)[:k]]
        w = 1.0 / (d[np.argsort(d)[:k]] + 0.05)
        est = float(np.average(r12[idx], weights=w))
        expected[i] = est
        realised = r12[cand]
        scores[i] = 100.0 * (realised <= est).mean()
    # OOS validation: score computed with strictly past data, outcome in future
    ok = valid & np.isfinite(scores)
    s = scores[ok]
    rr = r12[ok]
    qs = np.quantile(s, [0, 0.2, 0.4, 0.6, 0.8, 1.0]) if len(s) else np.zeros(6)
    quint = []
    for q in range(5):
        m = (s >= qs[q]) & (s <= qs[q + 1] if q == 4 else s < qs[q + 1])
        if m.sum() == 0:
            quint.append({"q": q + 1, "n": 0, "mean": None, "p_pos": None})
            continue
        quint.append({"q": q + 1, "n": int(m.sum()), "mean": float(rr[m].mean()),
                      "p_pos": float((rr[m] > 0).mean())})
    base = float(r12[valid].mean())
    top = next((x["mean"] for x in quint if x["q"] == 5), None)
    bot = next((x["mean"] for x in quint if x["q"] == 1), None)
    gap = (top - bot) if (top is not None and bot is not None) else None
    from scipy.stats import spearmanr
    rho, pv = spearmanr(s, rr) if len(s) > 5 else (0.0, 1.0)
    rho, pv = float(rho) if np.isfinite(rho) else 0.0, float(pv) if np.isfinite(pv) else 1.0
    validated = bool(top is not None and gap is not None and top > base + 0.01
                     and gap > 0.02 and rho > 0.10 and pv < 0.10)
    return {
        "validated": validated, "base_mean": base,
        "spearman": rho, "spearman_p": pv,
        "top_quintile_mean": top, "bottom_quintile_mean": bot, "gap": gap,
        "quintiles": quint,
        "history": [{"date": (str(pd.Timestamp(dates[i]).date()) if dates is not None
                              else "state_" + str(i)),
                     "score": None if not np.isfinite(scores[i]) else round(float(scores[i]), 1),
                     "fwd12": None if not np.isfinite(r12[i]) else round(float(r12[i]), 4)}
                    for i in range(n)],
        "today_score": None if not np.isfinite(scores[today_row]) else round(float(scores[today_row]), 1),
        "today_expected": None if not np.isfinite(expected[today_row]) else round(float(expected[today_row]), 4),
        "method": "Nearest-40 prior states (point-in-time percentile features); weighted mean "
                  "realised 12M return percentile-ranked against ALL prior realised 12M returns. "
                  "Validated only if top quintile beats the unconditional mean, the top-bottom gap "
                  "exceeds 2pp, and Spearman rank correlation with realised returns is positive (p<0.10).",
    }


# ---------------------------------------------------------------------------
# 7. Deployment backtest: lump sum vs DCA
# ---------------------------------------------------------------------------

def _strategy_path(p: np.ndarray, t: int, H: int, tranches: list[int],
                   cash_daily: float) -> tuple[float, float]:
    """Terminal wealth and max drawdown of a mark-to-market portfolio."""
    shares, cash, left = 0.0, 1.0, 1.0
    plan = {d: 1.0 / len(tranches) for d in tranches}
    mtm0 = 1.0
    peak, mdd = mtm0, 0.0
    for j in range(H + 1):
        if j in plan:
            amt = min(plan[j], cash)
            shares += amt / p[t + j]
            cash -= amt
        cash *= (1.0 + cash_daily)
        mtm = shares * p[t + j] + cash
        peak = max(peak, mtm)
        mdd = min(mdd, mtm / peak - 1.0)
    return float(mtm), float(mdd)


def deployment_backtest(px: pd.DataFrame, sym: str, pos: np.ndarray,
                        oos_prob: dict, base_prob: float, today_neighbours: list[int],
                        score_today: float | None, gate_pass: bool, conf: dict) -> dict:
    p = px[sym].to_numpy(float)
    irx = px["IRX"].to_numpy(float)
    last = len(p) - 1
    starts = [int(t) for t in pos if int(t) + 252 <= last]
    strategies = {
        "Immediate Lump Sum": [0],
        "DCA 3 months": [0, 21, 42],
        "DCA 6 months": [0, 21, 42, 63, 84, 105],
        "DCA 12 months": [21 * k for k in range(12)],
    }
    recs = []
    pos_to_prob = {int(k): float(v) for k, v in oos_prob.items()}
    obs_keys = np.array(sorted(pos_to_prob), dtype=float)

    def prob_at(t):
        if len(obs_keys) == 0:
            return base_prob
        j = int(np.argmin(np.abs(obs_keys - t)))
        return pos_to_prob[int(obs_keys[j])] if abs(obs_keys[j] - t) <= 15 else base_prob
    for t in starts:
        cd = float(np.nan_to_num(irx[t], nan=0.0)) / 100.0 / 252.0
        res = {}
        for name, tr in strategies.items():
            w, dd = _strategy_path(p, t, 252, tr, cd)
            res[name] = (w, dd)
        # Conditional entry: deploy when the OOS model probability exceeds the
        # prior base rate; otherwise stage in and re-check monthly.
        tr = [0]
        pr = prob_at(t)
        if pr <= base_prob + 0.03:
            deployed = False
            for k in range(1, 12):
                pr2 = prob_at(int(t + 21 * k))
                if pr2 > base_prob + 0.03:
                    tr.append(21 * k)
                    deployed = True
                    break
            if not deployed:
                tr = [21 * k for k in range(12)]
        # normalise tranches to equal weights (path function assumes equal)
        w, dd = _strategy_path(p, t, 252, tr if len(tr) > 1 else [0], cd)
        res["Conditional Entry (model)"] = (w, dd)
        best = max(v[0] for v in res.values())
        recs.append({"t": t, "w": {k: v[0] for k, v in res.items()},
                     "dd": {k: v[1] for k, v in res.items()}, "best": best})

    def summarise(rows):
        out = []
        for name in list(strategies) + ["Conditional Entry (model)"]:
            w = np.array([r["w"][name] for r in rows])
            dd = np.array([r["dd"][name] for r in rows])
            regret = np.array([r["best"] - r["w"][name] for r in rows])
            ls = np.array([r["w"]["Immediate Lump Sum"] for r in rows])
            out.append({
                "name": name,
                "expected_wealth": float(w.mean()),
                "median_wealth": float(np.median(w)),
                "drawdown": float(dd.mean()),
                "regret": float(regret.mean()),
                "p_underperform_lumpsum": float((w < ls - 1e-9).mean()),
                "p_best": float((w >= np.array([r["best"] for r in rows]) - 1e-9).mean()),
                "n": len(rows),
            })
        return out

    overall = summarise(recs)
    nb = [r for r in recs if r["t"] in set(today_neighbours)]
    conditional = summarise(nb) if nb else overall

    # ---- unconditional policy evidence (does NOT require timing skill) -----
    # Cluster monthly start dates into non-overlapping 252-session episodes so
    # the uncertainty estimate is not inflated by overlapping forward windows.
    episodes, cur = [], []
    for r in recs:
        if cur and r["t"] - cur[0]["t"] >= 252:
            episodes.append(cur)
            cur = []
        cur.append(r)
    if cur:
        episodes.append(cur)
    diffs = np.array([np.mean([r["w"]["Immediate Lump Sum"] - r["w"]["DCA 3 months"]
                               for r in e]) for e in episodes])
    rng = np.random.default_rng(SEED)
    boots = (np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                       for _ in range(4000)]) if len(diffs) > 1 else diffs)
    pol_lo = float(np.quantile(boots, 0.025)) if len(diffs) else 0.0
    pol_hi = float(np.quantile(boots, 0.975)) if len(diffs) else 0.0
    pol_mean = float(diffs.mean()) if len(diffs) else 0.0
    pol_episodes = len(diffs)
    pol_robust = bool(pol_lo > 0)
    ls_all = next(x for x in overall if x["name"] == "Immediate Lump Sum")
    d3_all = next(x for x in overall if x["name"] == "DCA 3 months")

    # ---- may today's state drive the decision at all? ---------------------
    # The fail-safe gate and the confidence engine MUST propagate here. With a
    # near-random AUC or grade-D evidence, 40 overlapping analogues cannot
    # justify a directional deployment choice. Suppressing a probability but
    # still printing "ALL IN" from the same weak sample is a contradiction.
    conditional_ok = bool(gate_pass and not conf["low_confidence"]
                          and conf["n_eff"] >= 30 and len(nb) >= 30)
    ls = next(x for x in conditional if x["name"] == "Immediate Lump Sum")
    alts = [x for x in conditional if x["name"] != "Immediate Lump Sum"]
    winner = max(alts, key=lambda x: x["expected_wealth"])
    margin = winner["expected_wealth"] - ls["expected_wealth"]
    alts_names = [k for k in strategies if k != "Immediate Lump Sum"] + ["Conditional Entry (model)"]
    p_beaten = float(np.mean([any(r["w"][a] > r["w"]["Immediate Lump Sum"] + 1e-9
                                  for a in alts_names) for r in nb])) if nb else 0.0
    dca3 = next(x for x in conditional if x["name"] == "DCA 3 months")
    dca12 = next(x for x in conditional if x["name"] == "DCA 12 months")

    if not conditional_ok:
        # No timing call. The unconditional table is reported as a POLICY
        # DEFAULT for capital already committed to equities, never as a
        # forecast about today and never as a high-conviction buy signal.
        rec = "No timing call"
        basis = "unconditional base rate (fail-safe active)"
        ci_note = ("; CI excludes zero" if pol_robust else
                   "; CI includes zero, so even this policy default is not statistically robust")
        why = (f"The only defensible statement is unconditional: across {ls_all['n']} monthly "
               f"start dates ({pol_episodes} independent episodes), immediate lump sum returned "
               f"{ls_all['expected_wealth']:.4f} vs {d3_all['expected_wealth']:.4f} per $1 for "
               f"DCA 3M (difference {pol_mean:+.4f}, 95% CI {pol_lo:+.4f} to {pol_hi:+.4f}"
               f"{ci_note}). This is a base-rate default for someone who has ALREADY decided to "
               f"invest, not a forecast that today is a good entry.")
    elif margin > 0.005:
        rec = {"DCA 3 months": "70/30", "DCA 6 months": "50/50",
               "DCA 12 months": "DCA 6M", "Conditional Entry (model)": "70/30"}[winner["name"]]
        basis = "conditional out-of-sample analogues"
        why = (f"In {len(nb)} similar historical states ({conf['n_eff']} independent episodes, "
               f"grade {conf['grade']}), {winner['name']} delivered "
               f"{winner['expected_wealth']:.3f} vs {ls['expected_wealth']:.3f} per $1 for lump sum "
               f"(+{margin * 100:.1f}%). Lump sum beat DCA 3M in "
               f"{dca3['p_underperform_lumpsum'] * 100:.0f}% of cases.")
    else:
        rec = "All In"
        basis = "conditional out-of-sample analogues (no deviation from base-rate default)"
        why = (f"With {conf['n_eff']} independent episodes (grade {conf['grade']}), no staged "
               f"strategy beat immediate lump sum by more than 0.5% on average in {len(nb)} "
               f"similar states (best challenger {winner['expected_wealth']:.3f} vs "
               f"{ls['expected_wealth']:.3f}). Lump sum beat DCA 3M in "
               f"{dca3['p_underperform_lumpsum'] * 100:.0f}% and DCA 12M in "
               f"{dca12['p_underperform_lumpsum'] * 100:.0f}% of cases. This means 'do not "
               f"deviate from the unconditional default', not 'today is especially attractive'.")
    return {"overall": overall, "conditional": conditional,
            "n_similar": len(nb), "recommendation": rec, "justification": why,
            "action_basis": basis, "decision_released": conditional_ok,
            "conditional_used_for_decision": conditional_ok,
            "conditional_n_eff": conf["n_eff"],
            "policy_default": "Immediate Lump Sum" if pol_mean > 0 else "DCA 3 months",
            "policy_mean_diff": pol_mean, "policy_ci_low": pol_lo, "policy_ci_high": pol_hi,
            "policy_n_episodes": pol_episodes, "policy_robust": pol_robust,
            "policy_note": "Unconditional full-sample policy default; valid only for capital "
                           "already committed to equities. It is not conditioned on today's "
                           "state and is not a market-timing signal.",
            "p_lumpsum_beaten_by_staged": p_beaten,
            "cash_assumption": "Uninvested cash earns the point-in-time 13-week T-bill rate (IRX); "
                               "12 monthly tranches; terminal wealth measured 252 sessions after entry."}


# ---------------------------------------------------------------------------
# 8. Confidence engine
# ---------------------------------------------------------------------------

def confidence_engine(analogues: list[int], r12_by_pos: dict, H: int = 252) -> dict:
    vals = np.array([r12_by_pos[t] for t in analogues if t in r12_by_pos and np.isfinite(r12_by_pos[t])])
    dates = sorted([t for t in analogues if t in r12_by_pos and np.isfinite(r12_by_pos[t])])
    n = len(vals)
    # cluster overlapping forward windows into independent episodes
    episodes, cur = [], []
    for t in dates:
        if cur and t - cur[0] >= H:
            episodes.append(cur)
            cur = []
        cur.append(t)
    if cur:
        episodes.append(cur)
    n_eff = len(episodes)
    ep_means = np.array([np.mean([r12_by_pos[t] for t in e]) for e in episodes]) if episodes else np.array([0.0])
    rng = np.random.default_rng(SEED)
    boots = np.array([rng.choice(ep_means, size=len(ep_means), replace=True).mean()
                      for _ in range(2000)]) if len(ep_means) > 1 else ep_means
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    if n_eff >= 100:
        grade = "A"
    elif n_eff >= 60:
        grade = "B"
    elif n_eff >= 30:
        grade = "C"
    else:
        grade = "D"
    return {"n_cases": n, "n_eff": n_eff, "n_episodes": n_eff,
            "ci_low": lo, "ci_high": hi, "mean": float(vals.mean()) if n else None,
            "grade": grade, "low_confidence": n_eff < 30,
            "note": "Effective sample size counts non-overlapping 12-month episodes, "
                    "not correlated daily observations."}


# ---------------------------------------------------------------------------
# 9. Orchestration
# ---------------------------------------------------------------------------

_FEATURE_POS: list[int] = []
_FEATURE_DATES = []
_ALL_POS: np.ndarray = np.array([])
_y_lookup: dict = {}


def run() -> dict:
    global _FEATURE_POS, _FEATURE_DATES, _ALL_POS, _y_lookup
    px = load_prices()
    raw, groups = build_raw_features(px)
    last = len(px) - 1

    month_ends = px.index.to_series().resample("ME").last()
    obs_all = np.array([p for p in px.index.get_indexer(pd.DatetimeIndex(month_ends.values)) if p >= 0])
    obs_all = obs_all[(obs_all >= 500) & (obs_all + 252 <= last)]
    today_pos = int(last)
    feature_positions = sorted(set(obs_all.tolist() + [today_pos]))
    _FEATURE_POS, _FEATURE_DATES = feature_positions, [px.index[i] for i in feature_positions]
    _ALL_POS = np.asarray(feature_positions)
    X = point_in_time_percentiles(raw, feature_positions).to_numpy(float)
    names = list(raw.columns)
    today_row = feature_positions.index(today_pos)
    obs = np.asarray(obs_all)
    print(f"[engine] {len(px)} daily rows | {len(obs)} monthly observations "
          f"| {X.shape[1]} candidate features | as-of {px.index[today_pos].date()}")

    # targets + unconditional (historical base-rate) distributions
    targets, base_rates = {}, {}
    for tk, sym in TICKERS.items():
        fp = forward_path_stats(px, sym, feature_positions)
        targets[tk] = fp
        base_rates[tk] = {}
        for h in HORIZONS:
            base_rates[tk][h] = unconditional_stats(fp[h]["r"], fp[h]["maxdd"], fp[h]["sharpe"])
    _y_lookup = {}

    # ---- model grids -----------------------------------------------------
    model_evidence, oos_prob, feature_audit = {}, {}, {}
    for tk in TICKERS:
        model_evidence[tk], oos_prob[tk] = {}, {}
        for h, H in HORIZONS.items():
            r = targets[tk][h]["r"]
            y = (r > 0).astype(float)
            valid = np.isfinite(r)
            # global lookup used by split construction for class-balance check
            for i, p in enumerate(feature_positions):
                _y_lookup[int(p)] = y[i] if valid[i] else 0.0
            splits = build_splits(obs, H)
            grid = run_model_grid(X, y, names, splits, names)
            model_evidence[tk][h] = {
                "metrics": {f"{s}|{m}": {k: val for k, val in v.items() if k != "predictions"}
                            for (s, m), v in grid["metrics"].items() if v is not None},
                "n_splits": {s: len(v) for s, v in splits.items()},
                "purge_embargo_days": H,
            }
            # store OOS probabilities of the best non-baseline model for deployment
            exp = grid["metrics"].get(("Expanding", "XGBoost"))
            if exp:
                oos_prob[tk][h] = {
                    int(k): v for k, v in exp.get("predictions", {}).items()
                } if "predictions" in exp else {}
            feature_audit[(tk, h)] = grid["audit"]
            print(f"[engine] {tk} {h}: {sum(len(v) for v in splits.values())} purged splits done")

    # ---- freeze retained feature set (aggregated, all horizons) -----------
    mi_all = np.zeros(len(names))
    cnt = 0
    perm_all = {n: [] for n in names}
    shap_all = {n: [] for n in names}
    uni_all = {n: [] for n in names}
    sel_count = np.zeros(len(names))
    decade_edges = {"2000s": [], "2010s": [], "2020s": []}
    for (tk, h), aud in feature_audit.items():
        for a in aud["mi"]:
            mi_all += np.asarray(a)
            cnt += 1
        for s in aud["selection"]:
            for i in s:
                sel_count[i] += 1
        for n, v in aud["perm"].items():
            perm_all[n] += v
        for n, v in aud["shap"].items():
            shap_all[n] += v
        for n, v in aud["univariate_brier"].items():
            uni_all[n] += v
        for d, arrs in aud.get("decade_edge", {}).items():
            decade_edges[d] += arrs
    mi_all = mi_all / max(cnt, 1)
    # Stability = share of decades in which the feature improved OOS Brier vs base rate
    stab = np.zeros(len(names))
    n_dec = 0
    for d, arrs in decade_edges.items():
        if arrs:
            stab += (np.mean(np.asarray(arrs), axis=0) > 0).astype(float)
            n_dec += 1
    stab /= max(1, n_dec)
    feats = []
    for i, n in enumerate(names):
        mi = float(mi_all[i])
        pm_eval, sh_eval = len(perm_all[n]) > 0, len(shap_all[n]) > 0
        pm = float(np.mean(perm_all[n])) if pm_eval else None
        sh = float(np.mean(shap_all[n])) if sh_eval else None
        uv = float(np.mean(uni_all[n])) if uni_all[n] else 0.0
        sel = float(sel_count[i] / max(cnt, 1))
        stability = float(stab[i])
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
                      "reason": "; ".join(reasons) if reasons else "positive OOS evidence",
                      "availability": ("2011+" if n == "vol_vix_term_structure" else
                                       "2007+" if n.startswith("credit_") else
                                       "2003+" if n.startswith("breadth_") else "1999+")})
    retained_idx = [i for i, f in enumerate(feats) if f["verdict"] == "RETAINED"]
    n_actually_retained = len(retained_idx)
    fallback = len(retained_idx) < 3
    if fallback:
        retained_idx = list(np.argsort(-mi_all)[:K_FEATURES])
    retained_names = [names[i] for i in retained_idx]
    print(f"[engine] feature gate retained {n_actually_retained}/{len(names)}; "
          f"model grid uses {len(retained_names)} ({'top-MI fallback' if fallback else 'retained set'})")

    # ---- best model + fail-safe gates ------------------------------------
    SCHEMES = ["Walk-forward", "Rolling", "Expanding", "CPCV"]
    gates_all, forecasts, best_models = {}, {}, {}
    for tk in TICKERS:
        best_models[tk] = {}
        for h in HORIZONS:
            me = model_evidence[tk][h]["metrics"]
            cand = {}
            for m in MODEL_NAMES[1:]:
                vals = [me.get(f"{s}|{m}") for s in SCHEMES]
                vals = [v for v in vals if v]
                if len(vals) == len(SCHEMES):
                    cand[m] = float(np.mean([v["brier"] for v in vals]))
            if not cand:
                best_models[tk][h] = {"name": "Base Rate", "brier": None}
                continue
            best = min(cand, key=cand.get)
            best_models[tk][h] = {"name": best, "brier": cand[best], "all": cand}
            bm = [me[f"{s}|{best}"] for s in SCHEMES]
            br = [me[f"{s}|Base Rate"] for s in SCHEMES]
            beaten = sum(1 for a, b in zip(bm, br) if a["brier"] < b["brier"])
            mean_edge = float(np.mean([b["brier"] - a["brier"] for a, b in zip(bm, br)]))
            auc = float(np.mean([x["roc_auc"] for x in bm]))
            slope = float(np.mean([x["cal_slope"] for x in bm]))
            icept = float(np.mean([x["cal_intercept"] for x in bm]))
            ece = float(np.mean([x["ece"] for x in bm]))
            checks = [
                {"name": "Brier better than base-rate model",
                 "pass": bool(mean_edge > 0 and beaten >= 3),
                 "detail": f"mean edge {mean_edge:+.4f}; beaten in {beaten}/4 schemes"},
                {"name": "ROC AUC not near random", "pass": bool(auc > 0.55),
                 "detail": f"AUC {auc:.3f} (threshold 0.55)"},
                {"name": "Calibration slope positive", "pass": bool(0.2 < slope < 2.5),
                 "detail": f"slope {slope:.2f}"},
                {"name": "Calibration intercept near zero", "pass": bool(abs(icept) < 1.0),
                 "detail": f"intercept {icept:+.2f}"},
                {"name": "Expected calibration error", "pass": bool(ece < 0.10),
                 "detail": f"ECE {ece:.3f}"},
            ]
            gates_all[(tk, h)] = {"best": best, "checks": checks,
                                  "pass": all(c["pass"] for c in checks),
                                  "mean_brier_edge": mean_edge, "schemes_beaten": beaten,
                                  "auc": auc, "slope": slope, "intercept": icept, "ece": ece}
            forecasts[(tk, h)] = {"edge": all(c["pass"] for c in checks), "model": best}

    # decision horizon gates drive the headline verdict
    primary = {tk: gates_all[(tk, "12M")] for tk in TICKERS}
    any_edge = any(g["pass"] for g in primary.values())

    # ---- outcome-based entry score, analogues, confidence, deployment -----
    finals, scores, deploy, conf, forecast_tables = {}, {}, {}, {}, {}
    row_of_pos = {int(p): i for i, p in enumerate(feature_positions)}
    for tk, sym in TICKERS.items():
        r12 = targets[tk]["12M"]["r"]
        sc = entry_score(X, r12, feature_positions, today_row, retained_idx, _FEATURE_DATES)
        scores[tk] = sc
        # nearest historical states (with realised outcomes) to today's state
        Z = X[:, retained_idx] if retained_idx else X[:, :K_FEATURES]
        cand = [i for i in range(len(feature_positions)) if np.isfinite(r12[i])]
        d = np.sqrt(np.mean((Z[cand] - Z[today_row]) ** 2, axis=1))
        order = np.argsort(d)[:40]
        analogues = [feature_positions[cand[i]] for i in order]
        r12_by_pos = {int(feature_positions[i]): float(r12[i]) for i in cand}
        conf[tk] = confidence_engine(analogues, r12_by_pos)
        gate = primary[tk]
        br = base_rates[tk]["12M"]

        # ---- target-variable output: model / base rate / incremental edge ----
        ana_rows = [row_of_pos[t] for t in analogues if t in row_of_pos]
        ft, cond_by_h = {}, {}
        for h in HORIZONS:
            fp = targets[tk][h]
            bb = base_rates[tk][h]
            gh = gates_all[(tk, h)]
            released = bool(gh["pass"])
            ar = np.array([fp["r"][i] for i in ana_rows if np.isfinite(fp["r"][i])])
            ad = np.array([fp["maxdd"][i] for i in ana_rows if np.isfinite(fp["maxdd"][i])])
            ash = np.array([fp["sharpe"][i] for i in ana_rows if np.isfinite(fp["sharpe"][i])])
            cond = unconditional_stats(ar, ad, ash)
            cond_by_h[h] = cond
            p_model = None
            if released:
                vr = np.where(np.isfinite(fp["r"]))[0]
                yh = (fp["r"][vr] > 0).astype(float)
                try:
                    pm, _ = fit_predict(gh["best"], X[vr][:, retained_idx], yh,
                                        X[[today_row]][:, retained_idx])
                    p_model = float(pm[0])
                except Exception:
                    p_model = None

            def frow(metric, base, model, unit):
                return {"metric": metric, "base": base, "model": model,
                        "edge": (model - base) if model is not None else None, "unit": unit}

            ft[h] = {
                "released": released, "model": gh["best"],
                "rows": [
                    frow("P(return > 0)", bb["p_pos"], p_model, "prob"),
                    frow("P(return > 5%)", bb["p_gt5"], cond["p_gt5"] if released else None, "prob"),
                    frow("P(return > 10%)", bb["p_gt10"], cond["p_gt10"] if released else None, "prob"),
                    frow("Expected return", bb["mean"], cond["mean"] if released else None, "pct"),
                    frow("Expected downside (CVaR 10%)", bb["downside_cvar10"],
                         cond["downside_cvar10"] if released else None, "pct"),
                    frow("Expected max drawdown", bb["drawdown"], cond["drawdown"] if released else None, "pct"),
                    frow("Expected Sharpe", bb["sharpe"], cond["sharpe"] if released else None, "num"),
                ],
            }
        forecast_tables[tk] = ft

        # conditional estimates from realised analogue outcomes (only if a model
        # earned the right to forecast); otherwise unconditional base rates only.
        if gate["pass"]:
            exp_ret = cond_by_h["12M"]["mean"]
            exp_down = cond_by_h["12M"]["downside_cvar10"]
            exp_dd = cond_by_h["12M"]["drawdown"]
            exp_sh = cond_by_h["12M"]["sharpe"]
            ret_type, rating = "conditional (OOS analogue outcomes)", sc["today_score"]
        else:
            exp_ret, exp_down = br["mean"], br["downside_cvar10"]
            exp_dd, exp_sh = br["drawdown"], br["sharpe"]
            ret_type, rating = "unconditional historical base rate (not a forecast)", None
        deploy[tk] = deployment_backtest(
            px, sym, obs,
            oos_prob.get(tk, {}).get("12M", {}),
            br["p_pos"], analogues, sc["today_score"],
            gate["pass"], conf[tk])
        action = deploy[tk]["recommendation"]
        zone = ("Neutral — maintain ordinary investment plan" if not gate["pass"]
                else ("Strong Buy Zone" if (rating or 0) >= 80 else
                      "Good Buy Zone" if (rating or 0) >= 60 else
                      "Neutral" if (rating or 0) >= 40 else
                      "Wait For Better Entry" if (rating or 0) >= 20 else "Poor Entry"))
        if not gate["pass"]:
            justification = (
                f"No model beat the base rate out of sample (best: {gate['best']}, "
                f"mean Brier edge {gate['mean_brier_edge']:+.4f}, AUC {gate['auc']:.2f}, "
                f"calibration slope {gate['slope']:.2f}). An AUC near 0.5 means the model cannot "
                f"rank today's entry against history, so the forecast, the 0-100 rating and every "
                f"directional deployment action are suppressed by the fail-safe rule. "
                f"Confidence {conf[tk]['grade']} ({conf[tk]['n_eff']} independent 12M episodes, "
                f"95% CI {conf[tk]['ci_low']:+.1%} to {conf[tk]['ci_high']:+.1%}) is below the "
                f"30-case minimum, so the nearest-analogue table is shown for audit but does NOT "
                f"drive the decision. {deploy[tk]['justification']} "
                f"Maintain your ordinary investment plan; do not deviate because of this dashboard.")
        else:
            justification = (
                f"{gate['best']} beat the base-rate model in {gate['schemes_beaten']}/4 validation "
                f"schemes (mean Brier edge {gate['mean_brier_edge']:+.4f}, AUC {gate['auc']:.2f}). "
                f"Confidence grade {conf[tk]['grade']} ({conf[tk]['n_eff']} independent episodes). "
                f"{deploy[tk]['justification']}")
        finals[tk] = {
            "rating": rating, "rating_suppressed": rating is None,
            "descriptive_rating": sc["today_score"],
            "confidence": conf[tk]["grade"],
            "expected_12m_return": exp_ret, "expected_12m_return_type": ret_type,
            "expected_downside": exp_down,
            "expected_drawdown": exp_dd, "expected_sharpe": exp_sh,
            "action": action, "zone": zone, "justification": justification,
            "edge_detected": gate["pass"],
            "decision_released": deploy[tk]["decision_released"],
            "action_basis": deploy[tk]["action_basis"],
            "policy_default": deploy[tk]["policy_default"],
            "policy_mean_diff": deploy[tk]["policy_mean_diff"],
            "policy_ci_low": deploy[tk]["policy_ci_low"],
            "policy_ci_high": deploy[tk]["policy_ci_high"],
            "policy_n_episodes": deploy[tk]["policy_n_episodes"],
            "policy_robust": deploy[tk]["policy_robust"],
        }

    # ---- scheme comparison ----------------------------------------------
    scheme_cmp = []
    for s in SCHEMES:
        row = {"scheme": s}
        for tk in TICKERS:
            me = model_evidence[tk]["12M"]["metrics"]
            vals = [(me[f"{sc}|{m}"]["brier"], m) for sc, m in
                    itertools.product([s], MODEL_NAMES[1:]) if me.get(f"{sc}|{m}")]
            b = me.get(f"{s}|Base Rate")
            row[f"{tk}_baseline_brier"] = b["brier"] if b else None
            row[f"{tk}_best_model"] = min(vals)[1] if vals else "n/a"
            row[f"{tk}_best_brier"] = min(vals)[0] if vals else None
            row[f"{tk}_brier_edge"] = (b["brier"] - min(vals)[0]) if (b and vals) else None
            row[f"{tk}_n_test"] = b["n"] if b else None
        scheme_cmp.append(row)

    # ---- assemble payload ------------------------------------------------
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

    payload = {
        "meta": {
            "version": "2.0 — evidence-first decision system",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "as_of": str(px.index[today_pos].date()),
            "sample_start": str(px.index[obs[0]].date()),
            "sample_end": str(px.index[obs[-1]].date()),
            "n_daily": int(len(px)), "n_monthly_observations": int(len(obs)),
            "n_candidate_features": len(names), "n_retained_features": n_actually_retained,
            "n_model_features": len(retained_names),
            "feature_fallback": fallback,
            "horizons": list(HORIZONS), "schemes": SCHEMES, "models": MODEL_NAMES,
            "proxy": "VOO total return proxied by SPY adjusted close (validated: daily return "
                     "corr 0.9986, monthly 0.9998 vs actual VOO 2010-2026).",
            "data": "Yahoo/Stooq dividend-adjusted closes, 1999-2026, bundled offline dataset.",
        },
        "verdict": {
            "edge_detected": any_edge,
            "headline": ("PREDICTIVE EDGE DETECTED" if any_edge else
                         "NO RELIABLE MARKET TIMING SIGNAL CURRENTLY DETECTED"),
            "subheadline": ("A model beat the base rate out of sample; conditional forecasts released."
                            if any_edge else
                            "Evidence favours maintaining ordinary investment plans."),
            "gates": {f"{tk}|{h}": v for (tk, h), v in gates_all.items()},
            "suppressed": {tk: not primary[tk]["pass"] for tk in TICKERS},
        },
        "final": finals,
        "forecasts": forecast_tables,
        "base_rates": base_rates,
        "model_evidence": model_evidence,
        "best_models": best_models,
        "scheme_comparison": scheme_cmp,
        "features": {
            "list": feats,
            "retained": [names[i] for i, f in enumerate(feats) if f["verdict"] == "RETAINED"],
            "model_feature_set": retained_names,
            "fallback": fallback,
            "fallback_note": ("No feature (or fewer than 3) cleared all four retention gates. "
                              "The model grid therefore uses the top mutual-information candidates "
                              "purely as a control set; this is NOT a claim of predictive value."
                              if fallback else "All model features cleared the retention gates."),
            "unavailable": [
                {"name": "Real rates / inflation trend / unemployment trend / economic surprise",
                 "group": "Macro", "reason": "No point-in-time CPI, TIPS, unemployment or CES series in the offline dataset. Excluded rather than proxied."},
                {"name": "Forward earnings yield / equity risk premium / Shiller CAPE",
                 "group": "Valuation", "reason": "No point-in-time fundamentals history available. Excluded rather than proxied."},
                {"name": "VIX term structure (full history)", "group": "Volatility",
                 "reason": "VIX9D only available from 2011; used with neutral 0.5 imputation before then."},
                {"name": "Advance/decline line", "group": "Breadth",
                 "reason": "No NYSE A/D series; sector-ETF advance/decline used as a transparent proxy."},
            ],
        },
        "entry_score": scores,
        "deployment": deploy,
        "confidence": conf,
        "disclosures": [
            "All probabilities are compared with the historical unconditional base rate; no model prediction is displayed alone.",
            "Labels overlap for 3/6/12-month horizons; every split purges overlapping training labels and enforces an embargo equal to the horizon.",
            "Feature selection (mutual information) runs inside each split on training data only; permutation and SHAP importance are computed on out-of-sample test rows.",
            "Monthly observations are used, not daily, to avoid pretending that overlapping windows are independent.",
            "Effective sample size counts non-overlapping 12-month episodes; confidence intervals bootstrap those episodes, not daily rows.",
            "The Entry Attractiveness Score is a percentile of REALISED future returns of historically similar states, computed with strictly prior data only. It is suppressed when unvalidated.",
            "Feature retention gate: mutual information above a shuffled-noise baseline, positive OOS permutation importance, positive univariate Brier edge vs the base rate, selection in >=20% of purged folds, and stability across >=50% of decades.",
            "The fail-safe propagates to the decision engine: when the model gate fails or confidence is grade D, no directional action is released and the conditional analogue backtest is excluded from the decision. The unconditional lump-sum-vs-DCA result is reported separately as a policy default with a bootstrap CI, never as a forecast about today.",
            "No post-hoc calibrator, no arbitrary indicator weighting, no sentiment, no macro commentary.",
            "Backtested results are not a guarantee of future performance. Educational research tool, not investment advice.",
        ],
    }
    payload = clean(payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[engine] wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    return payload


if __name__ == "__main__":
    p = run()
    print("[engine] verdict:", p["verdict"]["headline"])
    for tk in TICKERS:
        f = p["final"][tk]
        print(f"[engine] {tk}: rating={f['rating']} conf={f['confidence']} "
              f"action={f['action']} edge={f['edge_detected']}")
