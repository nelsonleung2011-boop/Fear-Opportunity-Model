#!/usr/bin/env python3
"""
benchmarks.py — does timing actually add value?

Ten strategies are run over the same window, on the same monthly decision
dates, using only walk-forward quantities. Every rule is evaluated on the
*out-of-sample* predictions produced by model_v4.walk_forward, so no rule can
see a prediction that was fitted with hindsight.

The output is deliberately a trade-off table, not a leaderboard: a rule that
earns more by sitting in cash for 40% of the period is not "better".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST_BPS = 5.0          # per buy/sell leg, 0.05% — realistic for liquid ETFs
TRADING_DAYS = 252


def _monthly_index(n):
    """Decision dates: every 21 sessions, which is the non-overlapping series."""
    return list(range(0, n, 21))


def _equity_curve(P: pd.Series, idx, weights):
    """
    weights[i] = fraction of the $1 budget deployed at decision date idx[i].
    Uninvested cash earns 0% — a deliberately conservative assumption.
    """
    prices = [P.iloc[i] for i in idx]
    units = 0.0
    cash = 1.0
    curve = []
    buys = 0
    for i, p in zip(idx, prices):
        w = weights.get(i, 0.0)
        if w > 0 and cash > 1e-12:
            amount = min(w, cash)
            amount *= (1 - COST_BPS / 1e4)     # transaction cost
            units += amount / p
            cash -= min(w, cash)
            buys += 1
        curve.append(units * p + cash)
    return np.array(curve), buys


def _metrics(curve, idx, P, n_months, buys, buy_returns=None):
    curve = np.asarray(curve, float)
    if len(curve) < 3 or curve[0] <= 0:
        return None
    years = max(len(curve) / 12.0, 1e-9)
    terminal = float(curve[-1] / curve[0])
    cagr = terminal ** (1 / years) - 1 if terminal > 0 else -1

    r = np.diff(curve) / curve[:-1]
    r = r[np.isfinite(r)]
    vol = float(np.std(r, ddof=1) * np.sqrt(12)) if len(r) > 2 else 0.0
    mean_m = float(np.mean(r)) if len(r) else 0.0
    sharpe = (mean_m * 12) / vol if vol > 1e-9 else 0.0
    downside = r[r < 0]
    dd_vol = float(np.std(downside, ddof=1) * np.sqrt(12)) if len(downside) > 2 else 0.0
    sortino = (mean_m * 12) / dd_vol if dd_vol > 1e-9 else 0.0

    peak = np.maximum.accumulate(curve)
    mdd = float(np.min(curve / peak - 1))
    calmar = cagr / abs(mdd) if abs(mdd) > 1e-9 else 0.0

    # 12-month rolling return distribution -> worst 5% experience
    worst5 = None
    if len(curve) > 13:
        roll = curve[12:] / curve[:-12] - 1
        worst5 = float(np.percentile(roll, 5))

    invested_months = int(np.sum(np.abs(np.diff(np.concatenate([[0], curve]))) >= 0))  # placeholder
    return {
        "terminal_wealth": round(terminal, 3),
        "cagr_pct": round(100 * cagr, 2),
        "vol_pct": round(100 * vol, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown_pct": round(100 * mdd, 1),
        "calmar": round(calmar, 2),
        "worst_5pct_12m": None if worst5 is None else round(100 * worst5, 1),
        "buys": int(buys),
        "months": int(n_months),
    }


def run(P_spy: pd.Series, dates, p_win: pd.Series, opp: pd.Series,
        feat: pd.DataFrame, regime: pd.Series) -> dict:
    """
    All rules are reduced to a set of monthly buy dates, then simulated with
    the same engine so the comparison is apples to apples.
    """
    n = len(P_spy)
    idx = _monthly_index(n)
    # start where the model has an out-of-sample prediction, so every rule sees
    # the same information set
    start = next((k for k, i in enumerate(idx)
                  if pd.notna(p_win.iloc[i]) and pd.notna(opp.iloc[i])), 0)
    idx = idx[start:]
    if len(idx) < 24:
        return {"available": False, "reason": "not enough out-of-sample months"}
    n_months = len(idx)

    def flags_to_weights(flags):
        buy_idx = [i for i, f in zip(idx, flags) if f]
        if not buy_idx:
            return {}, []
        w = 1.0 / len(buy_idx)
        return {i: w for i in buy_idx}, buy_idx

    def signal(series, cond):
        return [bool(cond(series.iloc[i])) if pd.notna(series.iloc[i]) else False
                for i in idx]

    pw = p_win.reindex(P_spy.index)
    op = opp.reindex(P_spy.index)

    rule_defs = {}
    # 1. immediate lump sum
    rule_defs["Immediate lump sum"] = {i: 1.0 for i in [idx[0]]}

    # 2/3. instalments — deploy 1/k at the first k decision dates
    for k in (3, 6, 12):
        rule_defs["Instalments over %dM" % k] = {i: 1.0 / k for i in idx[:k]}
    m = min(24, n_months)
    rule_defs["Fixed monthly (24M)"] = {i: 1.0 / m for i in idx[:m]}

    # 4. buy after fixed drawdowns
    dd = feat["dd"]
    for thr in (0.05, 0.10, 0.15, 0.20):
        f = [bool(dd.iloc[i] <= -thr) if pd.notna(dd.iloc[i]) else False for i in idx]
        rule_defs["Buy after -%d%% drawdown" % round(thr * 100)] = flags_to_weights(f)[0]

    # 5. trend
    td = feat["trend_dev"]
    f = [bool(td.iloc[i] > 0) if pd.notna(td.iloc[i]) else False for i in idx]
    rule_defs["SPY above 200DMA"] = flags_to_weights(f)[0]
    f = [bool(td.iloc[i] < 0) if pd.notna(td.iloc[i]) else False for i in idx]
    rule_defs["SPY below 200DMA"] = flags_to_weights(f)[0]

    # 6. VIX percentile
    vp = feat["vix_pct"]
    f = [bool(vp.iloc[i] < 50) if pd.notna(vp.iloc[i]) else False for i in idx]
    rule_defs["VIX percentile below 50"] = flags_to_weights(f)[0]

    # 7. trend + drawdown
    f = [bool(td.iloc[i] > 0 and dd.iloc[i] <= -0.03) if (pd.notna(td.iloc[i]) and pd.notna(dd.iloc[i])) else False
         for i in idx]
    rule_defs["Trend + 3% pullback"] = flags_to_weights(f)[0]

    # 8. existing probability score
    f = [bool(pw.iloc[i] >= 0.70) if pd.notna(pw.iloc[i]) else False for i in idx]
    rule_defs["Probability score >= 70%"] = flags_to_weights(f)[0]

    # 9. new two-dimensional matrix
    f = [bool(pw.iloc[i] >= 0.60 and op.iloc[i] >= 60) if (pd.notna(pw.iloc[i]) and pd.notna(op.iloc[i])) else False
         for i in idx]
    rule_defs["Prob>=60% and Opportunity>=60"] = flags_to_weights(f)[0]

    rows = []
    lump_terminal = None
    for name, weights in rule_defs.items():
        curve, buys = _equity_curve(P_spy, idx, weights)
        met = _metrics(curve, idx, P_spy, n_months, buys)
        if met is None:
            continue
        invested = 0.0
        cash = 1.0
        months_invested = 0
        for i in idx:
            w = weights.get(i, 0.0)
            if w > 0 and cash > 1e-12:
                cash -= min(w, cash)
            if cash < 1 - 1e-9:
                months_invested += 1
        met["time_invested_pct"] = round(100 * months_invested / n_months, 1)
        met["time_in_cash_pct"] = round(100 - met["time_invested_pct"], 1)
        first_buy = min(weights) if weights else None
        met["avg_wait_months"] = (round((idx.index(first_buy)) if first_buy in idx else None, 1)
                                  if first_buy is not None else None)
        met["turnover"] = round(COST_BPS * buys / 1e4, 4)
        met["cost_bps_per_leg"] = COST_BPS
        if name == "Immediate lump sum":
            lump_terminal = met["terminal_wealth"]
        rows.append({"strategy": name, **met})

    for r in rows:
        r["regret_vs_lump_sum"] = (None if lump_terminal is None
                                   else round(lump_terminal - r["terminal_wealth"], 3))

    # 10. regime-matched random months (Monte Carlo)
    rng = np.random.default_rng(7)
    reg_by_month = [regime.iloc[i] if pd.notna(regime.iloc[i]) else "Neutral" for i in idx]
    pool = {g: [i for i, g2 in zip(idx, reg_by_month) if g2 == g] for g in set(reg_by_month)}
    sims = []
    for _ in range(200):
        k = 6
        picks = []
        for _ in range(k):
            g = rng.choice(list(pool.keys()))
            picks.append(rng.choice(pool[g]))
        picks = list(dict.fromkeys(picks))
        w = {i: 1.0 / len(picks) for i in picks}
        cv, b = _equity_curve(P_spy, idx, w)
        met = _metrics(cv, idx, P_spy, n_months, b)
        if met:
            sims.append(met["terminal_wealth"])
    if sims:
        rows.append({
            "strategy": "Random months (regime-matched, mean of 200)",
            "terminal_wealth": round(float(np.mean(sims)), 3),
            "cagr_pct": None, "vol_pct": None, "sharpe": None, "sortino": None,
            "max_drawdown_pct": None, "calmar": None, "worst_5pct_12m": None,
            "buys": None, "months": n_months,
            "time_invested_pct": None, "time_in_cash_pct": None,
            "avg_wait_months": None, "turnover": None, "cost_bps_per_leg": COST_BPS,
            "regret_vs_lump_sum": (None if lump_terminal is None
                                   else round(lump_terminal - float(np.mean(sims)), 3)),
            "note": "5th-95th percentile of terminal wealth: %.3f - %.3f"
                    % (np.percentile(sims, 5), np.percentile(sims, 95)),
        })

    rows.sort(key=lambda r: -(r["terminal_wealth"] or 0))
    best = rows[0] if rows else None
    return {
        "available": True,
        "window": {"start": str(dates[idx[0]]), "end": str(dates[idx[-1]]),
                   "months": n_months},
        "assumptions": {
            "cost_bps_per_leg": COST_BPS,
            "cash_return": "0% (conservative — real T-bill interest would favour waiting rules)",
            "dividends": "included (dividend-adjusted closes)",
            "decision_frequency": "every 21 trading sessions (non-overlapping)",
            "budget": "$1 notional, deployed in equal tranches at the rule's buy dates",
            "worst_5pct": "5th percentile of the strategy's rolling 12-month return",
        },
        "strategies": rows,
        "best_by_terminal_wealth": best["strategy"] if best else None,
        "caveat": "Ranking by terminal wealth alone is misleading. Read terminal wealth "
                  "against max drawdown and time in cash. A rule that sits in cash for "
                  "half the sample has not been shown to time anything.",
    }
