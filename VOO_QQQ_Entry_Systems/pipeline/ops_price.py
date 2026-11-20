#!/usr/bin/env python3
"""Dynamic price-frame loader for the v7 operational pipeline.

The inherited v4 ``load_frames()`` truncates every frame at the literal date
2026-11-16. That is fine for a frozen research snapshot but fatal for a daily
system: refreshed prices would be silently discarded and the "as of" date would
never move. This module reproduces the exact same construction (raw close for
level features, adjusted close for labels, business-day reindex + ffill) while
resolving the cutoff from the data itself.

v2-v6 files are not modified.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import evidence_engine_v2 as E
from pipeline import evidence_engine_v4 as V4

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW2 = ROOT / "data_raw"

SYMS = (["SPY", "QQQ", "VIX", "TLT", "TNX", "IRX", "HYG", "LQD", "RSP", "IWM"]
        + E.SECTORS + ["XLC", "XLRE", "SOXX", "SMH", "W5000"])
AUX = (("FVX", RAW2 / "fvx.csv", False),
       ("IEF", RAW2 / "ief.csv", True),
       ("SHY", RAW2 / "shy.csv", True),
       ("VIX9D", RAW2 / "_vix9d.csv", True))


def build_frame(d: dict, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Pure frame construction. No hard-coded calendar date."""
    px = pd.DataFrame(d)
    px = px[~px.index.duplicated(keep="last")].sort_index()
    if cutoff is not None:
        px = px.loc[px.index <= cutoff]
    idx = pd.date_range(px.index.min(), px.index.max(), freq="B")
    return px.reindex(idx).ffill()


def resolve_as_of(raw: pd.DataFrame, explicit: str | None = None) -> pd.Timestamp:
    """Canonical scoring date = last date on which the core inputs all exist.

    SPY is the anchor: a stale auxiliary series must not drag the as-of date
    backwards, and a single freshly updated auxiliary must not push it forward.
    """
    if explicit:
        return pd.Timestamp(explicit)
    for key in ("SPY", "QQQ", "VIX"):
        s = raw[key].dropna()
        if len(s):
            return pd.Timestamp(s.index.max())
    return pd.Timestamp(raw.index.max())


def load_frames(as_of: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (adjusted, raw) frames truncated at a dynamically resolved date."""
    adj, raw = {}, {}
    for s in SYMS:
        f = RAW / f"{s}.csv"
        if not f.exists():
            raise FileNotFoundError(f"missing {f}; run ops_refresh.py")
        adj[s] = E._read_adj(f)
        raw[s] = E._read_close(f)
    for name, path, has_adj in AUX:
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        raw[name] = E._read_close(path)
        adj[name] = E._read_adj(path) if has_adj else E._read_close(path)

    cutoff = resolve_as_of(pd.DataFrame(raw), as_of)
    return build_frame(adj, cutoff), build_frame(raw, cutoff)


def last_dates() -> dict:
    """Per-file last observation date, used by the data audit and refresh log."""
    out = {}
    for f in sorted(RAW.glob("*.csv")):
        try:
            out[f.stem] = str(pd.to_datetime(E._read_close(f).index.max()).date())
        except Exception:
            out[f.stem] = None
    for name, path, _ in AUX:
        try:
            out[name] = str(pd.to_datetime(E._read_close(path).index.max()).date())
        except Exception:
            out[name] = None
    return out


def price_snapshot(n: int = 8) -> dict:
    """Last n raw closes from the actual CSV files, for the DATA tab."""
    keys = ["SPY", "QQQ", "VIX", "HYG", "LQD", "RSP", "IWM"]
    frames = {}
    for k in keys:
        f = RAW / f"{k}.csv"
        frames[k] = E._read_close(f) if f.exists() else pd.Series(dtype=float)
    idx = sorted(set().union(*[set(s.index[-n:]) for s in frames.values()]))[-n:]
    dates = [str(pd.Timestamp(d).date()) for d in idx]
    series = {}
    for k, s in frames.items():
        vals = []
        for d in idx:
            v = s.get(pd.Timestamp(d), np.nan)
            vals.append(None if not np.isfinite(v) else round(float(v), 2))
        series[k] = vals
    return {"dates": dates, "series": series}
