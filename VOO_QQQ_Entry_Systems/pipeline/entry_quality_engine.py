#!/usr/bin/env python3
"""
ETF Entry Quality System v5
===========================

A DIFFERENT QUESTION FROM v2-v4.

v2-v4 asked "can we predict the market?" and correctly answered NO.
v5 asks the decision question a long-term investor actually has:

    "Is today a relatively attractive long-term entry point?"

This is a DESCRIPTIVE / DECISION-SUPPORT system, not a return forecast.

  Layer 1  LONG-TERM VALUATION   -> Valuation Score 0-100
  Layer 2  TACTICAL ENTRY QUALITY -> Entry Quality Score 0-100
  Layer 3  RISK ASSESSMENT       -> Risk Meter + conditional drawdown probabilities

  FINAL    four questions, always answered. There is NO "Evidence Insufficient"
           state: valuation, entry quality and risk are observable today even
           when a predictive model has no edge. The internal backtest reports
           how much historical support the descriptive scores have, but it is
           NOT a gate on the answer.

ANTI-LOOKAHEAD
--------------
  * Every component is an EXPANDING percentile: at date t only observations
    with first-known-date <= t are used. No shrinking/normalising over the
    full sample.
  * Macro uses first-known date (nominal + publication lag).
  * Level/technical features use RAW closes; forward outcomes use ADJUSTED
    closes (dividends after t are not embedded in a feature).
  * The backtest uses non-overlapping block bootstrap and BH-FDR, but these
    are reported as evidence strength, never as a suppression switch.

Output: data/evidence_v5.json
"""
from __future__ import annotations

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
from pipeline import evidence_engine_v2 as E2          # noqa: E402
from pipeline import evidence_engine_v4 as V4          # noqa: E402
from pipeline import macro_data as MD                  # noqa: E402
from pipeline import entry_data as ED                  # noqa: E402

OUT = ROOT / "data" / "evidence_v5.json"
SEED = 20261119
RNG = np.random.default_rng(SEED)

# Pre-registered, NOT tuned on results.
MIN_SESSIONS = 252          # daily-derived component percentile warm-up
MIN_MONTHS = 36             # macro component percentile warm-up
MONTH_END = "ME"
BACKTEST_START = "2000-01-31"
N_BOOT = 2000
K_NEIGHBOURS = 80
BLOCK_MONTHS = 3            # non-overlapping bootstrap blocks

VAL_LABELS = [(90, "Extremely Cheap"), (75, "Attractive"), (60, "Fair"),
              (40, "Expensive"), (20, "Very Expensive"), (0, "Bubble-like")]
ENTRY_LABELS = [(90, "Rare Opportunity"), (80, "Strong Entry"), (60, "Good Entry"),
                (40, "Neutral"), (20, "Poor Entry"), (0, "Euphoric Market")]
RISK_ORDER = ["Very Low", "Low", "Moderate", "High", "Extreme"]

# Probability -> risk band (pre-registered from unconditional base rates)
DD10_BANDS = [(0.15, "Very Low"), (0.25, "Low"), (0.40, "Moderate"),
              (0.60, "High"), (1.01, "Extreme")]
DD20_BANDS = [(0.10, "Very Low"), (0.18, "Low"), (0.30, "Moderate"),
              (0.45, "High"), (1.01, "Extreme")]
PCT_BANDS = [(0.85, "Extreme"), (0.70, "High"), (0.50, "Moderate"), (0.30, "Low"),
             (0.0, "Very Low")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_from_score(score: float, labels: list[tuple[float, str]]) -> str:
    for thr, lab in labels:
        if score >= thr:
            return lab
    return labels[-1][1]


def band_from_pct(p: float, bands: list[tuple[float, str]] | None = None) -> str:
    """Severity bands: higher percentile = worse. Thresholds descending."""
    for thr, lab in (bands or PCT_BANDS):
        if p >= thr:
            return lab
    return "Very Low"


def band_from_prob(p: float, bands: list[tuple[float, str]]) -> str:
    """Drawdown probability bands: thresholds ASCENDING, first unmet wins."""
    if not np.isfinite(p):
        return "Moderate"
    for thr, lab in bands:
        if p < thr:
            return lab
    return "Extreme"


def band_index(lab: str) -> int:
    return RISK_ORDER.index(lab) if lab in RISK_ORDER else 2


def worse(a: str, b: str) -> str:
    return a if band_index(a) >= band_index(b) else b


def expanding_pct(series: pd.Series, at_dates, min_periods: int) -> pd.Series:
    """PIT percentile in [0,1]. At date t uses ONLY values known at/before t.

    Returns 0.5 when there is not enough own history. This is deliberately
    neutral rather than fabricated.
    """
    s = series.dropna().sort_index()
    out = {}
    v = s.to_numpy(float)
    for d in at_dates:
        d = pd.Timestamp(d)
        past = v[s.index.to_numpy() <= d]
        cur = s.get(d, np.nan)
        if not np.isfinite(cur):
            # last known value is acceptable (it was public by d)
            known = s.loc[:d]
            cur = known.iloc[-1] if len(known) else np.nan
        if len(past) < min_periods or not np.isfinite(cur):
            out[d] = 0.5
        else:
            out[d] = float(np.clip((past <= cur).sum() / len(past), 0.0, 1.0))
    return pd.Series(out, dtype=float)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, ctr - half)), float(min(1.0, ctr + half)))


def bh_fdr(pvals: list[float], q: float = 0.10) -> list[bool]:
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    out = np.zeros(n, dtype=bool)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        j = order[i]
        prev = min(prev, p[j] * n / (i + 1))
        out[j] = prev <= q
    return out.tolist()


def path_outcomes(adj: pd.DataFrame, sym: str, t: int, H: int) -> dict:
    p = adj[sym].to_numpy(float)
    if t + H >= len(p):
        return {k: np.nan for k in ("r", "maxdd", "sharpe", "sortino", "calmar")}
    path = p[t:t + H + 1]
    r = path[-1] / path[0] - 1.0
    peak = np.maximum.accumulate(path)
    mdd = float((path / peak - 1.0).min())
    rr = np.diff(path) / path[:-1]
    rr = rr[np.isfinite(rr)]
    if len(rr) < 5:
        return {"r": float(r), "maxdd": mdd, "sharpe": np.nan,
                "sortino": np.nan, "calmar": np.nan}
    sd = float(rr.std())
    dn = rr[rr < 0]
    dsd = float(np.sqrt(np.mean(dn ** 2))) if len(dn) else np.nan
    sharpe = float(rr.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan
    sortino = float(rr.mean() / dsd * np.sqrt(252)) if dsd and dsd > 0 else np.nan
    years = H / 252.0
    calmar = float(r / years / abs(mdd)) if mdd < -1e-9 else np.nan
    return {"r": float(r), "maxdd": mdd, "sharpe": sharpe,
            "sortino": sortino, "calmar": calmar}


# ---------------------------------------------------------------------------
# 1. Data / daily component construction
# ---------------------------------------------------------------------------

def build_macro_pit(index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    pit = V4.macro_pit(index)
    obs = json.loads((ED.MACRO / "entry_observations.json").read_text(encoding="utf-8"))
    meta = {}
    for k, lag in ED.LAGS.items():
        if k == "FNG":
            pit[k] = MD.pit_daily(index, obs.get(k, []), lag)
        else:
            pit[k] = MD.pit_daily(index, obs.get(k, []), lag)
        rows = obs.get(k, [])
        meta[k] = {"n": len(rows), "last": rows[-1]["date"] if rows else None,
                   "lag_days": lag}
    return pit.replace([np.inf, -np.inf], np.nan), meta


def build_daily_components(px_raw: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    """Daily raw values for every scored component. Higher = more attractive
    for valuation/entry layers. Every transform is backward-looking only."""
    f = pd.DataFrame(index=px_raw.index)
    spy = px_raw["SPY"]
    vix = px_raw["VIX"]

    # ---- Layer 1: valuation (higher = cheaper) --------------------------
    f["cape"] = pit["CAPE"]
    f["buffett"] = pit["W5000"] / pit["GDP"]
    f["earnings_yield"] = pit["EARNINGS_YIELD"]
    f["equity_risk_premium"] = pit["EARNINGS_YIELD"] - px_raw["TNX"]
    f["real_yield"] = pit["DFII10"]
    f["credit_spread"] = pit["BAMLH0A0HYM2"]
    f["curve_10y2y"] = pit["T10Y2Y"]
    f["fng_context"] = pit["FNG"]

    # ---- Layer 2: entry quality (higher = more fearful / better entry) --
    f["vix_level"] = vix
    ath = spy.cummax()
    f["dd_depth"] = 1.0 - spy / ath
    f["below_200dma"] = -(spy / spy.rolling(200).mean() - 1.0)
    rsp_rel = px_raw["RSP"] / spy
    ema19 = rsp_rel.ewm(span=19, adjust=False).mean()
    ema39 = rsp_rel.ewm(span=39, adjust=False).mean()
    f["mcclellan_neg"] = -(ema19 - ema39)
    f["breadth_thrust_neg"] = -(rsp_rel / rsp_rel.shift(21) - 1.0)
    f["putcall_proxy"] = px_raw["VIX9D"] / vix - 1.0
    universe = ["XLK", "XLF", "XLV", "XLI", "XLP", "XLY", "XLE", "XLU", "RSP", "IWM"]
    above = pd.concat([(px_raw[c] > px_raw[c].rolling(200).mean()).astype(float)
                       for c in universe], axis=1).mean(axis=1)
    f["pct_above_200dma_neg"] = 1.0 - above
    # Constructed Fear & Greed = mean of the seven scored fear components.
    # Display-only: assigned ZERO weight in the score to avoid double count.
    f["fear_greed_constructed"] = f[
        ["vix_level", "dd_depth", "below_200dma", "mcclellan_neg",
         "breadth_thrust_neg", "putcall_proxy", "pct_above_200dma_neg"]
    ].mean(axis=1)

    # ---- Layer 3: risk raw inputs ---------------------------------------
    f["credit_stress_price"] = -(px_raw["HYG"] / px_raw["LQD"]
                                  / (px_raw["HYG"] / px_raw["LQD"]).shift(63) - 1.0)
    f["nfci"] = pit["NFCI"]
    f["walcl_growth"] = pit["WALCL"] / pit["WALCL"].shift(126) - 1.0
    f["m2_growth"] = pit["M2SL"] / pit["M2SL"].shift(252) - 1.0
    f["gdp"] = pit["GDP"]
    return f.replace([np.inf, -np.inf], np.nan)


# direction: +1 means high raw value = attractive; -1 means invert
VAL_SPECS = [
    ("cape", "Shiller CAPE", -1, "Lower cyclically-adjusted P/E = cheaper long-term entry."),
    ("buffett", "Buffett Indicator (W5000/GDP proxy)", -1, "Lower market-cap/GDP = less extended valuation."),
    ("earnings_yield", "Earnings Yield", +1, "Higher trailing earnings yield = more value per dollar."),
    ("equity_risk_premium", "Equity Risk Premium (EY - 10Y)", +1, "Larger compensation over Treasuries."),
    ("real_yield", "Real Yield (10Y TIPS)", -1, "Higher real discount rate is a headwind to equity multiples."),
    ("credit_stress_price", "Credit Spread / Stress (HYG-LQD proxy)", -1,
     "High-yield underperformance versus investment grade marks tight/wary credit conditions."),
]
ENTRY_SPECS = [
    ("vix_level", "VIX Level", +1, "Elevated implied volatility marks fear-based pricing."),
    ("dd_depth", "Drawdown from All-Time High", +1, "Deeper drawdowns improve the entry location."),
    ("below_200dma", "Distance Below 200DMA", +1, "Prices further below trend have historically been better entries."),
    ("mcclellan_neg", "McClellan Oscillator (breadth proxy)", +1, "Negative breadth impulse indicates washed-out internals."),
    ("breadth_thrust_neg", "Breadth Thrust (equal vs cap weight)", +1, "Weak equal-weight thrust = narrow, fearful market."),
    ("putcall_proxy", "Put/Call Fear Proxy (VIX term structure)", +1, "9-day implied vol above 30-day signals urgent hedging demand."),
    ("pct_above_200dma_neg", "% Above 200DMA (inverted)", +1, "Few names above trend = broad capitulation."),
]

FNG_NOTE = ("CNN Fear & Greed is shown as current context only: its freely available "
            "history is about one year, which cannot support a since-2000 percentile. "
            "The dashboard shows a constructed equivalent from seven long-history "
            "components and gives it zero weight to avoid double counting.")


def score_layers(daily: pd.DataFrame, dates) -> tuple[pd.DataFrame, dict]:
    """Expanding PIT percentiles -> 0-100 scores. No full-sample statistics."""
    dates = list(pd.DatetimeIndex(dates))
    pct = {}
    meta = {"valuation": [], "entry": []}

    for key, name, direction, why in VAL_SPECS:
        minp = MIN_SESSIONS if key == "credit_stress_price" else MIN_MONTHS
        s = expanding_pct(daily[key], dates, minp)
        pct[key] = s if direction > 0 else 1.0 - s
        meta["valuation"].append({"key": key, "name": name, "direction": direction,
                                  "why": why, "coverage": int(daily[key].notna().sum())})

    for key, name, direction, why in ENTRY_SPECS:
        s = expanding_pct(daily[key], dates, MIN_SESSIONS)
        pct[key] = s if direction > 0 else 1.0 - s
        meta["entry"].append({"key": key, "name": name, "direction": direction,
                              "why": why, "coverage": int(daily[key].notna().sum())})

    P = pd.DataFrame(pct, index=pd.DatetimeIndex(dates))
    val = P[[k for k, _, _, _ in VAL_SPECS]].mean(axis=1) * 100.0
    ent = P[[k for k, _, _, _ in ENTRY_SPECS]].mean(axis=1) * 100.0
    return pd.DataFrame({"valuation": val, "entry": ent,
                         "combined": 0.5 * val + 0.5 * ent}, index=P.index), meta


# ---------------------------------------------------------------------------
# 2. Layer 3 -- conditional drawdown risk (non-parametric analogue estimate)
# ---------------------------------------------------------------------------

RISK_FEATURES = ["vix_level", "dd_depth", "credit_stress_price",
                 "pct_above_200dma_neg", "nfci", "curve_10y2y"]


def risk_probabilities(daily: pd.DataFrame, dates: list[pd.Timestamp],
                       adj: pd.DataFrame) -> pd.DataFrame:
    """k-NN empirical P(drawdown) using ONLY dates strictly before t.

    Features are standardised with expanding mean/std. Targets are realised
    forward drawdown events from adjusted prices. No fitted model is stored:
    each probability is the event frequency among the k closest historical
    risk states, which is auditable and cannot be retrained on the future.
    """
    dates = list(pd.DatetimeIndex(dates))
    pos = {d: int(adj.index.get_loc(d)) for d in dates if d in adj.index}
    usable = [d for d in dates if d in pos]

    X = daily[RISK_FEATURES].reindex(dates).to_numpy(float)
    ev10 = np.full(len(dates), np.nan)
    ev20 = np.full(len(dates), np.nan)
    adj_np = adj[["SPY"]].to_numpy(float)[:, 0]
    for i, d in enumerate(dates):
        t = pos.get(d)
        if t is None:
            continue
        for H, thresh, arr in ((63, -0.10, ev10), (252, -0.20, ev20)):
            if t + H < len(adj_np):
                arr[i] = float(path_outcomes(adj, "SPY", t, H)["maxdd"] <= thresh)

    out = {d: {"p10": np.nan, "p20": np.nan, "n10": 0, "n20": 0} for d in dates}
    # expanding standardisation uses only prior rows
    for i, d in enumerate(dates):
        hist_all = X[:i]
        finite_mask = np.isfinite(hist_all).all(axis=1)
        hist = hist_all[finite_mask]
        orig_idx = np.arange(i)[finite_mask]
        valid_y10 = np.isfinite(ev10[:i])
        valid_y20 = np.isfinite(ev20[:i])
        if len(hist) < 30 or not np.isfinite(X[i]).all():
            continue
        m = np.nanmean(hist, axis=0)
        sd = np.nanstd(hist, axis=0)
        sd[sd < 1e-12] = 1.0
        z = (X[i] - m) / sd
        zh = (hist - m) / sd
        dist = np.sqrt(np.nansum((zh - z) ** 2, axis=1))
        k = min(K_NEIGHBOURS, len(dist))
        sel = orig_idx[np.argsort(dist)[:k]]

        ok10 = np.zeros(i, dtype=bool)
        ok10[sel] = True
        ok10 &= valid_y10
        if ok10.sum() >= 20:
            out[d]["p10"] = float(ev10[:i][ok10].mean())
            out[d]["n10"] = int(ok10.sum())

        ok20 = np.zeros(i, dtype=bool)
        ok20[sel] = True
        ok20 &= valid_y20
        if ok20.sum() >= 20:
            out[d]["p20"] = float(ev20[:i][ok20].mean())
            out[d]["n20"] = int(ok20.sum())
    return pd.DataFrame.from_dict(out, orient="index")


def risk_subsignals(daily: pd.DataFrame, dates, probs: pd.DataFrame) -> pd.DataFrame:
    dates = list(pd.DatetimeIndex(dates))
    d = daily.reindex(dates)
    rows = []
    for dt in dates:
        p10, p20 = probs.loc[dt, "p10"], probs.loc[dt, "p20"]
        b10 = band_from_prob(p10, DD10_BANDS)
        b20 = band_from_prob(p20, DD20_BANDS)

        curve = d.loc[dt, "curve_10y2y"]
        nfci_p = expanding_pct(daily["nfci"], [dt], MIN_MONTHS).iloc[0]
        recession_hits = int(np.isfinite(curve) and curve < 0) + int(nfci_p > 0.80)
        recession = "Low" if recession_hits == 0 else ("Moderate" if recession_hits == 1 else "High")

        # FRED HY OAS starts only in 2023 (~3y), too short for a since-2000
        # percentile. Use the long-history HYG/LQD stress proxy; report the
        # raw OAS level as context only.
        hy_p = expanding_pct(daily["credit_stress_price"], [dt], MIN_SESSIONS).iloc[0]
        credit = band_from_pct(hy_p)
        hy_oas = daily["credit_spread"].reindex([dt]).iloc[0]

        wal_p = expanding_pct(daily["walcl_growth"], [dt], MIN_MONTHS).iloc[0]
        m2_p = expanding_pct(daily["m2_growth"], [dt], MIN_MONTHS).iloc[0]
        liq_stress = float(np.mean([nfci_p, 1.0 - wal_p, 1.0 - m2_p]))
        liquidity = band_from_pct(liq_stress)

        meter = b10
        for b in (b20, recession, credit, liquidity):
            meter = worse(meter, b)
        rows.append({"date": dt, "p10": p10, "p20": p20,
                     "n10": probs.loc[dt, "n10"], "n20": probs.loc[dt, "n20"],
                     "dd10_band": b10, "dd20_band": b20, "recession": recession,
                     "credit": credit, "liquidity": liquidity, "meter": meter,
                     "hy_pct": hy_p, "hy_oas_context": (None if not np.isfinite(hy_oas)
                                                        else float(hy_oas)),
                     "nfci_pct": nfci_p,
                     "liq_stress": liq_stress, "curve": curve})
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# 3. Decision (always answers)
# ---------------------------------------------------------------------------

def decide(val: float, ent: float, risk: str) -> dict:
    """Pre-registered purchase-pacing rules. Never a market-direction call."""
    safe = risk in ("Very Low", "Low", "Moderate")
    if risk == "Extreme":
        act, mult = "Slow planned purchases by 25%-50%; keep investing", "slow_strong"
    elif risk == "High" and ent < 60:
        act, mult = "Slow planned purchases by 25%; keep investing", "slow"
    elif ent >= 90 and val >= 75 and safe:
        act, mult = "Accelerate planned purchases by 50%-100%", "accelerate_strong"
    elif ent >= 80 and safe:
        act, mult = "Accelerate planned purchases by 25%-50%", "accelerate"
    elif ent >= 65 and val >= 70 and safe:
        act, mult = "Accelerate planned purchases by 10%-25%", "accelerate_mild"
    elif ent < 40:
        act, mult = "Slow planned purchases by 25%; keep investing", "slow"
    else:
        act, mult = "Buy according to your plan", "normal"
    return {"action": act, "mode": mult,
            "note": "Purchase pacing only. This is not a forecast and never a reason "
                    "to abandon a long-term plan."}


def explain(val: float, ent: float, risk: str, comps: dict) -> str:
    parts = []
    parts.append(f"Valuation is {label_from_score(val, VAL_LABELS).lower()} at {val:.0f}/100")
    parts.append(f"entry conditions are {label_from_score(ent, ENTRY_LABELS).lower()} at {ent:.0f}/100")
    parts.append(f"downside risk is {risk.lower()}")
    if ent >= 90 and val >= 75 and risk in RISK_ORDER[:3]:
        tail = "This is a rare alignment of cheap valuation, fear and manageable risk."
    elif ent >= 80:
        tail = ("This is not a generational opportunity, but conditions are materially "
                "better than the typical entry date.")
    elif ent < 40:
        tail = "Conditions are less favourable than usual; patience is rewarded by pacing, not by exiting."
    else:
        tail = "Conditions are close to the historical average; discipline matters more than timing."
    return "; ".join(parts) + ". " + tail


# ---------------------------------------------------------------------------
# 4. Internal backtest -- does high entry quality coincide with better outcomes?
# ---------------------------------------------------------------------------

def run_backtest(scores: pd.DataFrame, adj: pd.DataFrame, sym: str = "SPY") -> dict:
    positions = {d: int(adj.index.get_loc(d)) for d in scores.index if d in adj.index}
    rows = []
    for d, s in scores.iterrows():
        t = positions.get(d)
        if t is None:
            continue
        rec = {"date": d, "entry": s["entry"], "valuation": s["valuation"],
               "combined": s["combined"]}
        for h, H in E2.HORIZONS.items():
            rec.update({f"{h}_{k}": v for k, v in path_outcomes(adj, sym, t, H).items()})
        rows.append(rec)
    df = pd.DataFrame(rows).dropna(subset=["12M_r"])
    if len(df) < 60:
        return {"available": False, "n_obs": int(len(df))}

    # quintiles on the full historical score distribution (descriptive, not a gate)
    df["bucket"] = pd.qcut(df["entry"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    quarters = []
    for b, g in df.groupby("bucket", observed=True):
        rec = {"bucket": int(b), "n": int(len(g)),
               "entry_range": [round(float(g.entry.min()), 1), round(float(g.entry.max()), 1)]}
        for h in E2.HORIZONS:
            rec[f"{h}_median_return"] = round(float(g[f"{h}_r"].median()) * 100, 2)
            rec[f"{h}_win_rate"] = round(float((g[f"{h}_r"] > 0).mean()) * 100, 1)
        for k in ("sharpe", "sortino", "calmar", "maxdd"):
            rec[f"12M_{k}_median"] = (round(float(g[f"12M_{k}"].median()), 3)
                                      if g[f"12M_{k}"].notna().any() else None)
        quarters.append(rec)

    lo = df[df.bucket == 1]
    hi = df[df.bucket == 5]
    tests = []
    for h in E2.HORIZONS:
        a = lo[f"{h}_r"].dropna().to_numpy(float)
        b = hi[f"{h}_r"].dropna().to_numpy(float)
        diff = float(np.median(b) - np.median(a))
        # non-overlapping blocks: take every BLOCK_MONTHS-th observation
        block_idx = np.arange(0, min(len(a), len(b)), BLOCK_MONTHS)
        aa, bb = a[block_idx], b[block_idx]
        if len(aa) < 12:
            continue
        boot = np.empty(N_BOOT)
        for i in range(N_BOOT):
            pick = RNG.integers(0, len(aa), len(aa))
            boot[i] = np.median(bb[pick]) - np.median(aa[pick])
        ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        # two-sided permutation-style p from bootstrap centred at 0
        p = float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        tests.append({"horizon": h, "metric": "median_forward_return",
                      "top_minus_bottom_pp": round(diff * 100, 2),
                      "ci_low_pp": round(ci[0] * 100, 2),
                      "ci_high_pp": round(ci[1] * 100, 2),
                      "p_value": p, "n_blocks": int(len(aa))})

    # Sharpe difference test (12M)
    a = lo["12M_sharpe"].dropna().to_numpy(float)
    b = hi["12M_sharpe"].dropna().to_numpy(float)
    if min(len(a), len(b)) >= 20:
        bi = np.arange(0, min(len(a), len(b)), BLOCK_MONTHS)
        aa, bb = a[bi], b[bi]
        diff = float(np.median(bb) - np.median(aa))
        boot = np.empty(N_BOOT)
        for i in range(N_BOOT):
            pick = RNG.integers(0, len(aa), len(aa))
            boot[i] = np.median(bb[pick]) - np.median(aa[pick])
        p = float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        tests.append({"horizon": "12M", "metric": "median_sharpe",
                      "top_minus_bottom_pp": round(diff, 3),
                      "ci_low_pp": round(float(np.quantile(boot, 0.025)), 3),
                      "ci_high_pp": round(float(np.quantile(boot, 0.975)), 3),
                      "p_value": p, "n_blocks": int(len(aa))})

    flags = bh_fdr([t["p_value"] for t in tests], 0.10)
    for t, f in zip(tests, flags):
        t["fdr_significant"] = bool(f)
    n_sig = sum(f and t["top_minus_bottom_pp"] > 0 for t, f in zip(tests, flags))
    if n_sig >= 2:
        concl, strength = "Historical support is moderate to strong.", "Moderate-Strong"
    elif n_sig == 1:
        concl, strength = "Historical support is limited and horizon-specific.", "Limited"
    else:
        concl, strength = ("No statistically reliable performance gap survives multiple-testing "
                           "correction. The scores remain useful as descriptive entry-context "
                           "measures, not as return predictors."), "Not established"

    return {"available": True, "n_obs": int(len(df)),
            "window": [str(df.date.min().date()), str(df.date.max().date())],
            "quintiles": quarters, "tests": tests, "fdr_q": 0.10,
            "conclusion": concl, "strength": strength,
            "method": "Non-overlapping 3-month block bootstrap (2000 draws), BH-FDR over all "
                      "tests. Reported as evidence strength; it never suppresses the four answers."}


# ---------------------------------------------------------------------------
# 5. Buying scores -- percentile vs all historical entry dates
# ---------------------------------------------------------------------------

def buying_scores(scores: pd.DataFrame, daily: pd.DataFrame,
                  px_raw: pd.DataFrame, today: pd.Timestamp) -> dict:
    out = {}
    hist = scores["combined"].dropna()
    for sym in ("VOO", "QQQ"):
        proxy = "SPY" if sym == "VOO" else sym
        px = px_raw[proxy]
        sym_dd = 1.0 - px / px.cummax()
        sym_below = -(px / px.rolling(200).mean() - 1.0)
        sym_comb = pd.DataFrame({"dd": expanding_pct(sym_dd, hist.index, MIN_SESSIONS),
                                 "below": expanding_pct(sym_below, hist.index, MIN_SESSIONS)})
        sym_at_today = pd.DataFrame({
            "dd": expanding_pct(sym_dd, [today], MIN_SESSIONS),
            "below": expanding_pct(sym_below, [today], MIN_SESSIONS)})
        # instrument-specific overlay (25%) on top of the common decision context
        hist_score = 0.75 * scores["combined"] + 0.25 * sym_comb.mean(axis=1) * 100
        today_score = (0.75 * float(scores.loc[today, "combined"])
                       + 0.25 * float(sym_at_today.mean(axis=1).iloc[0]) * 100)
        prior = hist_score.loc[:today].dropna()
        prior = prior[prior.index < today]
        pct = float((prior <= today_score).mean()) if len(prior) >= 36 else 0.5
        out[sym] = {
            "score": int(round(pct * 100)),
            "rank_text": f"Better than {round(pct * 100)}% of historical entry dates since 2000",
            "n_history": int(len(prior)),
            "instrument_context": {
                "drawdown_from_ath_pct": round(float(1.0 - px.loc[today] / px.cummax().loc[today]) * 100, 2),
                "distance_below_200dma_pct": round(float(sym_below.loc[today]) * 100, 2),
            },
            "method": "75% common valuation+entry context, 25% instrument-specific drawdown/trend "
                      "percentile; ranked against prior month-end dates only (PIT). "
                      "Not a return forecast.",
        }
    return out


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def run() -> dict:
    adj, raw = V4.load_frames()
    pit, macro_meta = build_macro_pit(raw.index)
    daily = build_daily_components(raw, pit)

    all_dates = pd.date_range(BACKTEST_START, raw.index.max(), freq=MONTH_END)
    dates = [d for d in all_dates if d in raw.index]
    if dates and dates[-1] == pd.Timestamp(raw.index.max().date()):
        dates = dates[:-1]
    today = pd.Timestamp(raw.index.max())
    score_dates = dates + [today]

    scores, meta = score_layers(daily, score_dates)
    probs = risk_probabilities(daily, score_dates, adj)
    risks = risk_subsignals(daily, score_dates, probs)

    cur = scores.loc[today]
    cur_r = risks.loc[today]
    val, ent = float(cur["valuation"]), float(cur["entry"])
    meter = str(cur_r["meter"])
    dec = decide(val, ent, meter)
    if val >= 60:
        opportunity = "above"
    elif val <= 40:
        opportunity = "below"
    else:
        opportunity = "close to"
    opportunity_line = (
        f" Valuation-implied long-term opportunity is {opportunity} the historical average; "
        "this is a relative-value statement, not a return forecast.")

    bt = run_backtest(scores.loc[dates], adj, "SPY")
    buys = buying_scores(scores, daily, raw, today)

    def comp_rows(layer: str, row: pd.Series) -> list[dict]:
        specs = meta[layer]
        rows = []
        for sp in specs:
            k = sp["key"]
            minp = (MIN_SESSIONS if layer == "entry" or k == "credit_stress_price"
                    else MIN_MONTHS)
            p = float(expanding_pct(daily[k], [today], minp).iloc[0])
            if sp["direction"] < 0:
                p = 1.0 - p
            rawv = daily[k].loc[today]
            rows.append({"key": k, "name": sp["name"], "raw": (None if not np.isfinite(rawv)
                         else round(float(rawv), 4)), "pct": int(round(p * 100)),
                         "direction": sp["direction"], "why": sp["why"],
                         "coverage": sp["coverage"]})
        return rows

    val_comps = comp_rows("valuation", cur)
    ent_comps = comp_rows("entry", cur)

    payload = {
        "meta": {
            "version": "5.0 — Entry Quality System (valuation · entry · risk)",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "as_of": str(today.date()),
            "sample_start": str(raw.index.min().date()),
            "n_monthly_observations": len(dates),
            "framework": "descriptive decision support; explicitly NOT a market-timing model",
            "price_basis": "RAW closes for level features; ADJUSTED closes for forward outcomes",
            "macro_alignment": "first-known date (nominal date + publication lag)",
            "seed": SEED,
            "inherited": "v2 PIT primitives; v4 first-known-date macro alignment. "
                         "No v2-v4 predictive model, AUC gate or evidence grade is used.",
        },
        "final_page": {
            "q1": {"question": "How attractive is the market today?",
                   "answer": f"{val:.0f}/100", "label": label_from_score(val, VAL_LABELS),
                   "detail": f"{val:.0f}/100 — {label_from_score(val, VAL_LABELS)}. "
                             "Based on six valuation percentiles, each computed using only history "
                             "known at this date." + opportunity_line},
            "q2": {"question": "How favourable is today's entry point versus history?",
                   "answer": f"{ent:.0f}/100", "label": label_from_score(ent, ENTRY_LABELS),
                   "detail": f"{ent:.0f}/100 — {label_from_score(ent, ENTRY_LABELS)}. "
                             "Measures how fearful / washed-out conditions are relative to past "
                             "entry dates. It does not predict returns."},
            "q3": {"question": "What is the downside risk?",
                   "answer": meter, "label": meter,
                   "detail": f"Risk Meter: {meter}. Empirical P(10% drawdown in 3M) = "
                             f"{cur_r['p10'] * 100:.0f}% (n={int(cur_r['n10'])} analogues); "
                             f"P(20% drawdown in 12M) = {cur_r['p20'] * 100:.0f}% "
                             f"(n={int(cur_r['n20'])} analogues)."},
            "q4": {"question": "What should a long-term investor do?",
                   "answer": dec["action"], "label": dec["mode"],
                   "detail": explain(val, ent, meter, val_comps),
                   "guardrail": dec["note"]},
            "always_answered": True,
            "no_suppression": "Evidence strength is reported separately. It never replaces an "
                              "observable valuation, entry or risk assessment with 'insufficient'.",
        },
        "layers": {
            "valuation": {"score": round(val, 1), "label": label_from_score(val, VAL_LABELS),
                          "method": "Equal-weight mean of six PIT valuation percentiles. "
                                    "Equal weights are pre-registered, not optimised.",
                          "components": val_comps},
            "entry": {"score": round(ent, 1), "label": label_from_score(ent, ENTRY_LABELS),
                      "method": "Equal-weight mean of seven PIT fear/oversold percentiles. "
                                "Fear & Greed is displayed but given zero weight (see disclosures).",
                      "components": ent_comps,
                      "fear_greed": {
                          "constructed": int(round(float(expanding_pct(
                              daily["fear_greed_constructed"], [today], MIN_SESSIONS).iloc[0]) * 100)),
                          "cnn_current": (None if daily["fng_context"].dropna().empty else
                                          round(float(daily["fng_context"].dropna().iloc[-1]), 1)),
                          "note": FNG_NOTE}},
            "risk": {"meter": meter,
                     "probabilities": {
                         "dd10_3m": {"p": round(float(cur_r["p10"]) * 100, 1),
                                     "ci": [round(x * 100, 1) for x in wilson(
                                         int(round(cur_r["p10"] * cur_r["n10"])), int(cur_r["n10"]))]
                                     if cur_r["n10"] else [0, 100],
                                     "n_analogues": int(cur_r["n10"])},
                         "dd20_12m": {"p": round(float(cur_r["p20"]) * 100, 1),
                                      "ci": [round(x * 100, 1) for x in wilson(
                                          int(round(cur_r["p20"] * cur_r["n20"])), int(cur_r["n20"]))]
                                      if cur_r["n20"] else [0, 100],
                                      "n_analogues": int(cur_r["n20"])}},
                     "sub_signals": [
                         {"name": "3M -10% drawdown", "level": cur_r["dd10_band"]},
                         {"name": "12M -20% drawdown", "level": cur_r["dd20_band"]},
                         {"name": "Recession pressure", "level": cur_r["recession"]},
                         {"name": "Credit stress", "level": cur_r["credit"]},
                         {"name": "Liquidity stress", "level": cur_r["liquidity"]}],
                     "hy_oas_context_pct": (None if not np.isfinite(cur_r["hy_oas_context"])
                                            else round(float(cur_r["hy_oas_context"]), 2)),
                     "method": "k-nearest historical risk states (k=80, expanding standardisation) "
                               "with Wilson intervals; rule-based macro stress signals. "
                               "Worst sub-signal sets the meter. Credit uses the long-history "
                               "HYG/LQD proxy; the short FRED HY OAS series is context only."},
        },
        "buying_scores": buys,
        "backtest": bt,
        "data_quality": {
            "macro_sources": macro_meta,
            "excluded": [
                "CNN Fear & Greed full-history percentile — only ~1 year available",
                "Forward P/E — no genuine historical real-time vintage",
                "True McClellan Oscillator — no free long-history NYSE advance/decline series; "
                "equal-weight/cap-weight relative impulse used as a disclosed proxy",
                "True put/call ratio — no free long-history series; VIX9D/VIX term-structure proxy used",
                "Real-time vintage macro — FRED/multpl serve latest revised data",
            ],
        },
        "disclosures": [
            "This system measures entry CONDITIONS, not future returns. No model predicts direction.",
            "Scores are percentiles of today versus histories known at today; equal weights prevent tuning.",
            "The backtest is reported as supporting evidence, never as a gate on the decision.",
            "Purchase-pacing changes are deliberately capped and never require selling.",
            "Past conditions and outcomes do not guarantee future results. Not investment advice.",
        ],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    print(f"[v5] wrote {OUT.relative_to(ROOT)} | as-of {payload['meta']['as_of']}")
    print(f"[v5] valuation {val:.1f} ({payload['layers']['valuation']['label']}) | "
          f"entry {ent:.1f} ({payload['layers']['entry']['label']}) | risk {meter}")
    print(f"[v5] action: {dec['action']}")
    print(f"[v5] backtest strength: {bt.get('strength')} | buying VOO {buys['VOO']['score']} "
          f"QQQ {buys['QQQ']['score']}")
    return payload


if __name__ == "__main__":
    run()
