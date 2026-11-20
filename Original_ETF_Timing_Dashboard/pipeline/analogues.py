#!/usr/bin/env python3
"""
analogues.py — "what did the market do the last time it looked like this?"

Rules that make this honest:
  * features are point-in-time z-scores, so the target vector contains no
    future information and neither does any historical candidate vector;
  * highly correlated features are dropped before matching, otherwise the
    distance is dominated by three ways of saying "volatility";
  * near-duplicate dates are merged into *episodes* so five consecutive days
    in one crash cannot masquerade as five independent precedents;
  * the table reports dispersion, because agreement among five similar dates
    is not evidence and must never be labelled as confidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPISODE_GAP = 63          # sessions; a new episode starts after a 3-month gap
N_EPISODES = 5
CORR_DROP = 0.90          # drop a feature if |rho| exceeds this with a kept one


def _dedupe_features(Z: pd.DataFrame):
    """Greedy removal of near-collinear features, keeping the earlier name."""
    kept, dropped = [], []
    corr = Z.corr().abs()
    for c in Z.columns:
        if all(corr.loc[c, k] < CORR_DROP for k in kept):
            kept.append(c)
        else:
            dropped.append({"feature": c,
                            "collinear_with": next(k for k in kept if corr.loc[c, k] >= CORR_DROP),
                            "rho": round(float(corr.loc[c, next(k for k in kept if corr.loc[c, k] >= CORR_DROP)]), 2)})
    return kept, dropped


def _episodes(n: int, max_index: int):
    """
    Assign each index to a market episode.

    Two dates less than EPISODE_GAP sessions apart share most of their 6-month
    forward window, so their outcomes are not independent observations. Cutting
    the calendar into fixed EPISODE_GAP blocks is the simplest defensible
    clustering: no two representatives can overlap by more than half a horizon.
    """
    return np.arange(max_index) // EPISODE_GAP


def build(feat: pd.DataFrame, dates, fwd: dict, target_i: int) -> dict:
    """
    Find the N_EPISODES closest distinct historical episodes to `target_i`.
    Only dates whose 12-month outcome is already realised are eligible.
    """
    Z = feat.copy()
    # point-in-time standardisation: expanding z, already causal per column
    Z = (Z - Z.expanding(min_periods=252).mean()) / Z.expanding(min_periods=252).std()
    Z = Z.replace([np.inf, -np.inf], np.nan)

    kept, dropped = _dedupe_features(Z.dropna(how="all"))
    Zk = Z[kept]

    # eligible candidate dates: must have a realised 12M outcome
    hist_limit = target_i - (21 * 12 + 1)
    if hist_limit < 300:
        return {"available": False, "reason": "not enough history"}
    sub = Zk.iloc[:hist_limit]
    valid = sub.notna().all(axis=1).values
    cand = np.where(valid)[0]
    if len(cand) < 200:
        return {"available": False, "reason": "too few complete candidate rows"}

    tgt = Zk.iloc[target_i].values
    if np.isnan(tgt).any():
        return {"available": False, "reason": "target feature vector incomplete"}

    diff = sub.values[cand] - tgt
    dist = np.sqrt(np.nansum(diff ** 2, axis=1))

    ep_id = _episodes(len(Z), hist_limit)
    cand_ep = ep_id[cand]

    # one representative per episode: the closest member of each episode
    best_per_ep = {}
    for k, (j, e) in enumerate(zip(cand, cand_ep)):
        if e not in best_per_ep or dist[k] < best_per_ep[e][1]:
            best_per_ep[e] = (int(j), float(dist[k]), k)

    order = sorted(best_per_ep.items(), key=lambda kv: kv[1][1])[:N_EPISODES]

    rows = []
    for e, (j, d, k) in order:
        contrib = (diff[k] ** 2) / (d ** 2) if d > 0 else np.zeros(len(kept))
        top = sorted(zip(kept, contrib), key=lambda x: -x[1])[:3]
        rows.append({
            "date": str(pd.Timestamp(dates[j]).date()),
            "episode": int(e),
            "distance": round(float(d), 2),
            "similarity_pct": round(100 * max(0.0, 1 - d / np.sqrt(len(kept) * 9)), 1),
            "fwd_1m": _pct(fwd[1].iloc[j]),
            "fwd_3m": _pct(fwd[3].iloc[j]),
            "fwd_6m": _pct(fwd[6].iloc[j]),
            "fwd_12m": _pct(fwd[12].iloc[j]),
            "drivers": [{"feature": f, "share_pct": round(100 * float(s), 0)} for f, s in top],
        })

    def _stat(key):
        v = [r[key] for r in rows if r[key] is not None]
        if not v:
            return None
        v = np.array(v, float)
        return {"median": round(float(np.median(v)), 1),
                "mean": round(float(np.mean(v)), 1),
                "p25": round(float(np.percentile(v, 25)), 1),
                "p75": round(float(np.percentile(v, 75)), 1),
                "worst": round(float(np.min(v)), 1),
                "best": round(float(np.max(v)), 1),
                "n": int(len(v))}

    stats = {k: _stat(k) for k in ("fwd_1m", "fwd_3m", "fwd_6m", "fwd_12m")}
    iqr6 = (stats["fwd_6m"]["p75"] - stats["fwd_6m"]["p25"]) if stats["fwd_6m"] else None

    return {
        "available": bool(rows),
        "target_date": str(pd.Timestamp(dates[target_i]).date()),
        "features_used": kept,
        "features_dropped": dropped,
        "episodes_found": len(order),
        "episode_gap_sessions": EPISODE_GAP,
        "rows": rows,
        "stats": stats,
        "dispersion_6m_iqr": iqr6,
        "disagreement_warning": bool(iqr6 is not None and iqr6 > 25),
        "disclaimer": "Similarity is a distance in feature space, not a probability and "
                      "not confidence. Five close matches can all be the same episode seen "
                      "twice; only distinct episodes are shown.",
    }


def _pct(x):
    return None if pd.isna(x) else round(100 * float(x), 1)
