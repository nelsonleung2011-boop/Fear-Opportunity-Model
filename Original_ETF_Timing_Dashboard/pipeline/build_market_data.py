#!/usr/bin/env python3
"""
Build the canonical JSON datasets + the dashboard payload (model v4).

What changed vs v3
------------------
The headline number is no longer a rescaled "opportunity" composite. Three
separate quantities are reported, and they are never multiplied together:

  prob        calibrated P(6M total return > 0)        — literal probability
  opp         Opportunity Score 0-100                  — relative attractiveness
  timing      Timing Percentile 0-100                  — how unusual that is

All historical values are produced by pipeline/model_v4.walk_forward, i.e. by
models that never saw the date they are predicting. See data/METHODOLOGY.md.
"""
import os
import json
import math
import datetime as dt

import numpy as np
import pandas as pd

import model_v4
import benchmarks
import analogues as analogues_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
DATA = os.path.join(ROOT, "data")

MODEL_VERSION = "v4.0"
SCHEMA_VERSION = 4
FFILL_LIMIT = 5

SECTORS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLB", "XLRE", "XLU", "XLC"]
SECTOR_NAMES = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLI": "Industrials", "XLE": "Energy", "XLB": "Materials",
    "XLRE": "Real Estate", "XLU": "Utilities", "XLC": "Communication Services",
}
CORE_SYMBOLS = ["SPY", "QQQ", "VIX", "TLT", "TNX", "IRX", "HYG", "LQD", "RSP", "IWM"]


def load_close(key):
    """Dividend-adjusted close when available; raw close otherwise."""
    df = pd.read_csv(os.path.join(RAW, key + ".csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date", keep="last").set_index("date").sort_index()
    if "adj_close" in df.columns and pd.to_numeric(df["adj_close"], errors="coerce").notna().any():
        s = pd.to_numeric(df["adj_close"], errors="coerce")
        if "close" in df.columns:
            s = s.fillna(pd.to_numeric(df["close"], errors="coerce"))
    else:
        s = pd.to_numeric(df["close"], errors="coerce")
    return s.astype(float).sort_index()


def arr(s, nd=2):
    out = []
    for x in s:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            out.append(None)
        else:
            out.append(round(float(x), nd))
    return out


def main():
    prov = json.load(open(os.path.join(RAW, "provenance.json")))
    keys = CORE_SYMBOLS + SECTORS
    series = {k: load_close(k) for k in keys if os.path.exists(os.path.join(RAW, k + ".csv"))}

    spy = series["SPY"]
    idx = spy.index
    P = pd.DataFrame(index=idx)
    for k, s in series.items():
        # bounded forward-fill: a provider hole must not silently repeat a stale
        # price for months. Longer gaps stay NaN and surface in the audit.
        P[k] = s.reindex(idx).ffill(limit=FFILL_LIMIT)
    P = P.dropna(subset=["SPY"])

    dates = [d.strftime("%Y-%m-%d") for d in P.index]
    N = len(P)
    as_of = dates[-1]
    stale_days = int((pd.Timestamp.utcnow().tz_localize(None).normalize()
                      - P.index[-1].normalize()).days)

    # ---------------- features & labels ----------------
    feat = model_v4.build_features(P)
    fwd = model_v4.forward_returns(P["SPY"])

    # ---------------- purged walk-forward ----------------
    wf = model_v4.walk_forward(P, feat, fwd)
    p_win = wf["p_win"]
    mu = wf["mu"]
    resid = wf["resid_pool"]

    # forward-return distribution implied by training residuals (no test data)
    med6 = mu + float(np.median(resid))
    q10 = float(np.quantile(resid, 0.10))
    lo6 = mu + q10
    es_res = resid[resid <= q10]
    es6 = mu + (float(es_res.mean()) if len(es_res) else q10)
    p_loss10 = pd.Series([np.nan] * N, index=P.index)
    for i, v in enumerate(mu.values):
        if not np.isnan(v):
            p_loss10.iloc[i] = float(np.mean(resid < (-0.10 - v)))

    # ---------------- regime (needed for stratified reporting) ----------------
    sma200 = P["SPY"].rolling(200, min_periods=200).mean()
    above200 = (P["SPY"] > sma200).astype(float)
    mom12 = P["SPY"] / P["SPY"].shift(252) - 1
    breadth_rsp = P["RSP"] / P["SPY"]
    risk_hyg = P["HYG"] / P["LQD"]
    reg_raw = (above200
               + (mom12 > 0).astype(float)
               + (breadth_rsp / breadth_rsp.rolling(252, min_periods=126).mean() > 1).astype(float)
               + (risk_hyg / risk_hyg.rolling(252, min_periods=126).mean() > 1).astype(float)
               - (P["VIX"] > P["VIX"].rolling(756, min_periods=252).quantile(0.8)).astype(float))
    regime = pd.Series("Neutral", index=P.index)
    regime[reg_raw >= 3] = "Risk-On"
    regime[reg_raw <= 1] = "Risk-Off"

    # ---------------- out-of-sample evaluation ----------------
    eval_mask = (p_win.notna() & fwd[6].notna()).values
    eval_idx = np.where(eval_mask)[0]
    span = int(eval_idx[-1] - eval_idx[0] + 1) if len(eval_idx) else 0
    pv = p_win.values[eval_mask]
    yv = (fwd[6].values[eval_mask] > 0).astype(float)
    r6v = fwd[6].values[eval_mask]

    cal = model_v4.calibration_stats(pv, yv, span)

    # Phase 4: score each calibrator on out-of-sample predictions. Selection
    # itself happens on validation data; this table shows what was at stake.
    try:
        cal["methods_comparison"] = model_v4.compare_calibration_methods(P, feat, fwd)
    except Exception as e:  # noqa: BLE001
        cal["methods_comparison"] = []
        cal["methods_comparison_error"] = "%s: %s" % (type(e).__name__, e)

    # stratified diagnostics
    dts = np.array(dates)
    decade = pd.Series([d[:3] + "0s" for d in dts[eval_mask]])
    cal["by_decade"] = model_v4._slice_stats(dts[eval_mask], pv, yv, decade, None)
    regv = regime.values[eval_mask]
    cal["by_regime"] = model_v4._slice_stats(dts[eval_mask], pv, yv, regv, None)
    vol = feat["realvol"].values[eval_mask]
    if np.isfinite(vol).sum() > 100:
        qs = np.nanquantile(vol, [1 / 3, 2 / 3])
        terc = np.where(vol <= qs[0], "low vol", np.where(vol <= qs[1], "mid vol", "high vol"))
        cal["by_vol_tercile"] = model_v4._slice_stats(dts[eval_mask], pv, yv, terc, None)

    # non-overlapping monthly series for the significance-flavoured numbers
    monthly = np.arange(0, len(pv), 21)
    cal["n_independent"] = int(len(monthly))
    cal["brier_monthly"] = round(float(np.mean((pv[monthly] - yv[monthly]) ** 2)), 4)

    # ---------------- current reading ----------------
    i_now = N - 1
    prob_now = float(p_win.iloc[i_now]) if pd.notna(p_win.iloc[i_now]) else None
    # current calibration bin = decile of OOS predictions containing today's value
    cur_bin = None
    if prob_now is not None:
        pctl = float((pv <= prob_now).mean())
        lo_b, hi_b = max(0.0, min(0.9, math.floor(pctl * 10) / 10)), 0.0
        hi_b = min(1.0, lo_b + 0.1)
        m = (pv >= lo_b) & (pv < hi_b)
        if m.sum() < 30:
            m = (pv >= prob_now - 0.05) & (pv <= prob_now + 0.05)
        if m.sum() >= 20:
            idxs = eval_idx[m]
            n_raw = int(m.sum())
            n_eff = model_v4.effective_n(n_raw, int(idxs[-1] - idxs[0] + 1))
            obs = float(yv[m].mean())
            lo_w, hi_w = model_v4.wilson_interval(obs * n_eff, n_eff)
            cur_bin = {
                "predicted": round(100 * float(pv[m].mean()), 1),
                "observed": round(100 * obs, 1),
                "n_raw": n_raw, "n_eff": n_eff,
                "ci_low": round(100 * lo_w, 1) if lo_w is not None else None,
                "ci_high": round(100 * hi_w, 1) if hi_w is not None else None,
                "range": "%d-%d%%" % (round(100 * lo_b), round(100 * hi_b)),
            }

    # ---------------- signal significance (non-overlapping) ----------------
    try:
        sig_tests = model_v4.signal_tests(feat, fwd)
    except Exception as e:  # noqa: BLE001
        sig_tests = {"tests": [], "conclusion": "%s: %s" % (type(e).__name__, e)}

    # ---------------- opportunity / timing ----------------
    op = model_v4.compute_opportunity(feat, fwd, mu, resid)
    opp_now = float(op["opportunity"].iloc[i_now]) if pd.notna(op["opportunity"].iloc[i_now]) else None
    timing_now = {k: (float(v.iloc[i_now]) if pd.notna(v.iloc[i_now]) else None)
                  for k, v in op["timing"].items()}

    # ---------------- analogues ----------------
    ana = analogues_mod.build(feat, dates, fwd, i_now)

    # ---------------- conviction & matrix ----------------
    regime_now = str(regime.iloc[i_now])
    regime_cover = int(((regime.values[eval_mask] == regime_now)).sum())
    missing_inputs = int(sum(1 for k in CORE_SYMBOLS
                             if k in P.columns and P[k].iloc[i_now] != P[k].iloc[i_now]))
    conv = model_v4.conviction(
        i_now, cur_bin, op["opportunity"], op["families"], cal or {},
        stale_days,
        ana.get("dispersion_6m_iqr") if ana.get("available") else None,
        regime_cover, missing_inputs)
    matrix = model_v4.decision_matrix(
        prob_now * 100 if prob_now is not None else None, opp_now, conv["grade"])

    # ---------------- benchmarks ----------------
    try:
        bench = benchmarks.run(P["SPY"], dates, p_win, op["opportunity"], feat, regime)
    except Exception as e:  # noqa: BLE001 - a benchmark failure must not kill the build
        bench = {"available": False, "reason": "%s: %s" % (type(e).__name__, e)}

    # ---------------- methodology audit ----------------
    prov_ok = sum(1 for k in CORE_SYMBOLS if prov.get(k, {}).get("ok"))
    unadj = [k for k in keys if prov.get(k, {}).get("price_basis") == "close"]
    fallbacks = [k for k in keys if prov.get(k, {}).get("provider") == "Stooq"]
    audit_m = model_v4.methodology_audit({
        "providers_ok": prov_ok, "providers_total": len(CORE_SYMBOLS),
        "adjusted_all": len(unadj) == 0, "unadjusted": len(unadj),
        "ffill_limit": FFILL_LIMIT, "fallbacks": len(fallbacks),
        "stale_days": stale_days, "episode_clustered": True,
        "bands_fixed": True,
        "bands_note": "Opportunity bands are fixed at 0/20/40/60/80 and were declared before "
                      "any outcome was inspected; they are not fitted quantiles.",
    })

    # ---------------- sectors (unchanged logic) ----------------
    sector_rows = []
    sec_metrics = {}
    for sym in SECTORS:
        if sym not in P.columns:
            continue
        s = P[sym]
        rs = s / P["SPY"]
        mom = s / s.shift(126) - 1
        trend = s / s.rolling(200, min_periods=150).mean() - 1
        vol = s.pct_change().rolling(63, min_periods=40).std() * np.sqrt(252)
        lead = rs / rs.shift(126) - 1
        sec_metrics[sym] = {"rs": rs.iloc[-1], "mom": mom.iloc[-1], "trend": trend.iloc[-1],
                            "vol": vol.iloc[-1], "lead": lead.iloc[-1]}
    if sec_metrics:
        mets = pd.DataFrame(sec_metrics).T
        for c in ["rs", "mom", "trend", "lead"]:
            mets[c + "_pct"] = mets[c].rank(pct=True) * 100
        mets["vol_pct"] = (1 - mets["vol"].rank(pct=True)) * 100
        mets["buying_score"] = (0.30 * mets.rs_pct + 0.25 * mets.mom_pct
                                + 0.20 * mets.trend_pct + 0.15 * mets.vol_pct
                                + 0.10 * mets.lead_pct).round(1)
        for sym in mets.index:
            row = mets.loc[sym]
            sector_rows.append({
                "symbol": sym, "name": SECTOR_NAMES.get(sym, sym),
                "buying_score": round(float(row["buying_score"]), 1),
                "relative_strength": round(float(row["rs"]), 4),
                "momentum_6m": round(float(row["mom"]), 4),
                "trend_strength": round(float(row["trend"]), 4),
                "volatility": round(float(row["vol"]), 4),
            })
        sector_rows.sort(key=lambda r: -r["buying_score"])
        for rank, r in enumerate(sector_rows, 1):
            r["rank"] = rank
    sector_out = {"as_of": as_of, "sectors": sector_rows,
                  "top3": [r["symbol"] for r in sector_rows[:3]],
                  "top5": [r["symbol"] for r in sector_rows[:5]],
                  "bottom3": [r["symbol"] for r in sector_rows[-3:]]}

    # ==================================================================
    # payload
    # ==================================================================
    current = {
        "date": as_of,
        "prob_pct": round(100 * prob_now, 1) if prob_now is not None else None,
        "prob_bin": cur_bin,
        "calibration_method": (wf["blocks"][-1]["calibrator"] if wf["blocks"] else None),
        "ece_pp": cal.get("ece_pp") if cal else None,
        "calibration_warning": bool(cal and cal.get("ece_pp") is not None
                                    and cal["ece_pp"] > 5.0),
        "exp_ret_pct": round(100 * float(mu.iloc[i_now]), 2) if pd.notna(mu.iloc[i_now]) else None,
        "median_ret_pct": round(100 * float(med6.iloc[i_now]), 2) if pd.notna(med6.iloc[i_now]) else None,
        "p_loss10_pct": round(100 * float(p_loss10.iloc[i_now]), 1) if pd.notna(p_loss10.iloc[i_now]) else None,
        "es6_pct": round(100 * float(es6.iloc[i_now]), 2) if pd.notna(es6.iloc[i_now]) else None,
        "opp": round(opp_now, 1) if opp_now is not None else None,
        "opp_band": model_v4.opportunity_band(opp_now),
        "timing": {k: (round(v, 1) if v is not None else None) for k, v in timing_now.items()},
        "families": {k: (round(float(op["families"][k].iloc[i_now]), 1)
                         if pd.notna(op["families"][k].iloc[i_now]) else None)
                     for k in op["families"].columns},
        "weights": op["weights"],
        "regime": regime_now,
        "conviction": conv,
        "matrix": matrix,
        "stale_days": stale_days,
        "spy": round(float(P["SPY"].iloc[i_now]), 2),
        "vix": round(float(P["VIX"].iloc[i_now]), 2),
        "drawdown_pct": round(100 * float(P["SPY"].iloc[i_now] / P["SPY"].iloc[-252:].max() - 1), 1),
    }

    D = {
        "meta": {
            "generated_at": dt.datetime.utcnow().isoformat() + "Z",
            "as_of": as_of, "n": N, "start": dates[0],
            "schema_version": SCHEMA_VERSION, "model_version": MODEL_VERSION,
            "horizon_sessions": model_v4.H,
            "embargo_sessions": model_v4.EMBARGO,
            "price_basis": ("dividend-adjusted" if not unadj else "raw close"),
            "providers": {k: prov.get(k, {}).get("provider") for k in keys if k in prov},
            "stale_days": stale_days,
            "disclaimer": "Educational decision-support research. Not investment advice, "
                          "not a guarantee, and not a personalised recommendation.",
        },
        "dates": dates, "n": N,
        "prob": arr(p_win * 100, 1),
        "mu": arr(mu * 100, 2),
        "median6": arr(med6 * 100, 2),
        "ploss10": arr(p_loss10 * 100, 1),
        "opp": arr(op["opportunity"], 1),
        "timing_10y": arr(op["timing"]["10y"], 1),
        "timing_5y": arr(op["timing"]["5y"], 1),
        "timing_exp": arr(op["timing"]["expanding"], 1),
        "fam": {k: arr(v, 1) for k, v in op["families"].items()},
        "comp": {k: arr(v, 1) for k, v in op["components"].items()},
        "spy": arr(P["SPY"], 2), "vix": arr(P["VIX"], 2),
        "dd": arr(feat["dd"] * 100, 1), "vixpct": arr(feat["vix_pct"], 1),
        "sma200": arr(sma200, 2), "high52": arr(P["SPY"].rolling(252, min_periods=200).max(), 2),
        "trend_dev": arr(feat["trend_dev"] * 100, 1),
        "regime": list(regime),
        "current": current,
        "calibration": cal,
        "sensitivity": op["sensitivity"],
        "signal_tests": sig_tests,
        "analogues": ana,
        "benchmarks": bench,
        "blocks": wf["blocks"],
        "audit_method": audit_m,
        "audit_data": json.load(open(os.path.join(DATA, "audit.json"))),
        "sectors": sector_out,
        "legacy": _legacy_comparison(p_win, fwd, dates),
    }
    _dump_at_root("dashboard_data.json", D)

    # ---------------- canonical datasets ----------------
    _dump(os.path.join(DATA, "historical_data.json"), {
        "generated_at": D["meta"]["generated_at"], "start": dates[0], "end": as_of,
        "n": N, "dates": dates,
        "prob": D["prob"], "mu": D["mu"], "opp": D["opp"],
        "timing_10y": D["timing_10y"], "timing_5y": D["timing_5y"],
        "regime": D["regime"],
        "features": {k: arr(feat[k], 4) for k in feat.columns},
        "spy": D["spy"], "vix": D["vix"],
    })
    _dump(os.path.join(DATA, "market_data.json"), {
        "generated_at": D["meta"]["generated_at"], "as_of": as_of,
        "source": "GitHub Actions / Yahoo Finance (Stooq fallback)",
        "market": {"spy": current["spy"], "vix": current["vix"],
                   "prob": current["prob_pct"], "opp": current["opp"],
                   "regime": regime_now},
    })
    _dump(os.path.join(DATA, "sector_etf_data.json"), sector_out)
    _dump(os.path.join(DATA, "backtest_results.json"), {
        "generated_at": D["meta"]["generated_at"], "calibration": cal,
        "benchmarks": bench, "blocks": wf["blocks"], "sensitivity": op["sensitivity"],
    })
    _dump(os.path.join(DATA, "validation_report.json"), cal or {})
    _dump(os.path.join(DATA, "methodology_audit.json"), audit_m)

    print("\n=== model %s built ===" % MODEL_VERSION)
    print("  N=%d  %s -> %s  (stale %d d)" % (N, dates[0], as_of, stale_days))
    print("  P(up 6M)=%s  expected 6M=%s%%  opportunity=%s  timing(10y)=%s"
          % (current["prob_pct"], current["exp_ret_pct"], current["opp"],
             current["timing"]["10y"]))
    print("  conviction=%s (%s)  quadrant=%s"
          % (conv["grade"], conv["score"], matrix["quadrant"]))
    if cal:
        print("  OOS n=%d (n_eff %.0f)  Brier=%.4f (base %.4f)  logloss=%.4f  ECE=%.2fpp"
              % (cal["n"], cal["n_eff"], cal["brier"], cal["brier_baseline"],
                 cal["log_loss"], cal["ece_pp"]))
    print("  walk-forward blocks=%d  calibrators=%s"
          % (len(wf["blocks"]), sorted({b["calibrator"] for b in wf["blocks"]})))
    print("  methodology audit: %s (P%d / W%d / F%d)"
          % (audit_m["gate"], audit_m["pass"], audit_m["warning"], audit_m["fail"]))
    if bench.get("available"):
        top = bench["strategies"][:3]
        print("  benchmarks (top 3 by terminal wealth):",
              [(r["strategy"], r["terminal_wealth"]) for r in top])


def _legacy_comparison(p_win, fwd, dates):
    """Retained only to show why a win-rate-only score was not sufficient."""
    path = os.path.join(ROOT, "score_history.csv")
    if not os.path.exists(path):
        return {"available": False}
    lh = pd.read_csv(path, index_col=0, parse_dates=True)
    legacy = lh["opp_score"].reindex(pd.to_datetime(dates)).ffill()
    out = {"available": True, "bands": []}
    for lo, hi in [(50, 59), (60, 69), (70, 79), (80, 89), (90, 101)]:
        m = legacy.notna() & (legacy >= lo) & (legacy <= hi) & fwd[6].notna()
        r6 = fwd[6][m]
        if len(r6) < 50:
            continue
        out["bands"].append({
            "band": "%d-%d" % (lo, hi), "n": int(len(r6)),
            "avg_6m": round(100 * float(r6.mean()), 2),
            "win_6m": round(100 * float((r6 > 0).mean()), 1),
        })
    return out


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), default=str)
    print("  wrote %-34s %7.1f KB" % (os.path.relpath(path, ROOT),
                                      os.path.getsize(path) / 1024))


def _dump_at_root(name, obj):
    _dump(os.path.join(ROOT, name), obj)


if __name__ == "__main__":
    main()
