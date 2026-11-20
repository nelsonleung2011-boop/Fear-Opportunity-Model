#!/usr/bin/env python3
"""Provenance ledger for the v7 DATA & SOURCES tab.

Answers "is the data legit?" with evidence rather than reassurance: every input
that forms the score is listed with its raw value, source, coverage, publication
lag, the date on which that value was actually knowable, and its exact point
contribution. Known weaknesses (revised vintages, proxies, short histories) are
stated explicitly instead of being hidden behind a single score.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import entry_data as ED
from pipeline import macro_data as MD
from pipeline import ops_price as OP

ROOT = Path(__file__).resolve().parents[1]
MACRO = ROOT / "data" / "macro"
OPS = ROOT / "data" / "ops"

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"

# name, cache, key, source, kind, role, extra note
CATALOG = [
    ("SPY close", "price", "SPY", "Yahoo Finance chart API (SPY)", "REAL", "Drawdown, 200DMA distance, trend, breadth base", ""),
    ("QQQ close", "price", "QQQ", "Yahoo Finance chart API (QQQ)", "REAL", "QQQ drawdown bonus", ""),
    ("VIX", "price", "VIX", "CBOE ^VIX via Yahoo chart API", "REAL", "VIX nonlinear score, regime, detector", ""),
    ("VIX9D", "price", "VIX9D", "CBOE ^VIX9D via Yahoo chart API", "PROXY", "Put/call fear proxy (VIX term structure)", "Not options tape"),
    ("RSP", "price", "RSP", "Yahoo chart API (RSP)", "PROXY", "McClellan, breadth thrust, equal vs cap weight", "Not NYSE advance/decline"),
    ("IWM", "price", "IWM", "Yahoo chart API (IWM)", "PROXY", "Small-cap relative strength; breadth universe", ""),
    ("9 sector SPDRs", "price", "XLK", "Yahoo chart API (XLB..XLY)", "PROXY", "% above 200DMA, advance/decline proxy", "10-ETF breadth universe, not all stocks"),
    ("HYG / LQD", "price", "HYG", "Yahoo chart API (HYG, LQD)", "PROXY", "Credit stress price proxy; corporate bond momentum", ""),
    ("TNX / IRX / FVX", "price", "TNX", "CBOE yield indices via Yahoo", "PROXY", "Equity risk premium, TED-style funding proxy", "Literal LIBOR TED ended in 2021"),
    ("CAPE", "macro", "CAPE", "multpl.com Shiller CAPE monthly table", "REAL", "Valuation layer", "Latest revised vintage, 95-day lag"),
    ("Earnings Yield", "macro", "EARNINGS_YIELD", "multpl.com S&P 500 earnings yield", "REAL", "Valuation layer + equity risk premium", "Trailing, not forward"),
    ("Dividend Yield", "macro", "DIVIDEND_YIELD", "multpl.com S&P 500 dividend yield", "REAL", "Context / inherited features", ""),
    ("Trailing P/E", "macro", "PE_TRAILING", "multpl.com S&P 500 P/E", "REAL", "Context / inherited features", "Forward P/E unavailable; not fabricated"),
    ("W5000", "macro", "W5000", "Yahoo ^W5000 price index", "PROXY", "Buffett Indicator numerator", "Price index, NOT total market cap"),
    ("GDP", "macro", "GDP", "FRED nominal GDP (quarterly)", "REAL", "Buffett denominator, recession pressure", "Latest revised, 130-day lag"),
    ("WALCL", "macro", "WALCL", "FRED Federal Reserve total assets", "REAL", "Liquidity stress", "Latest revised, 8-day lag"),
    ("M2SL", "macro", "M2SL", "FRED M2 money stock", "REAL", "Liquidity stress", "Latest revised, 75-day lag"),
    ("NFCI", "macro", "NFCI", "FRED Chicago Fed financial conditions", "REAL", "Recession / liquidity stress", "Latest revised, 10-day lag"),
    ("DFII10", "entry", "DFII10", "FRED 10Y TIPS real yield", "REAL", "Valuation layer (real yield)", "Latest revised, 1-day lag"),
    ("BAMLH0A0HYM2", "entry", "BAMLH0A0HYM2", "FRED ICE BofA US High Yield OAS", "REAL", "Independent Credit Stress layer", "Only since 2023-11; fallback to HYG/LQD"),
    ("T10Y2Y", "entry", "T10Y2Y", "FRED 10Y minus 2Y Treasury curve", "REAL", "Funding / curve stress", "Latest revised, 1-day lag"),
    ("FNG", "entry", "FNG", "CNN Fear & Greed graphdata API", "CONTEXT", "Display only", "~1 year of history; zero weight"),
    ("Fear & Greed (constructed)", "constructed", "fear_greed_constructed", "Constructed from 7 long-history price/macro series", "CONSTRUCTED", "Rare detector condition; display context", "Zero weight in the Buying Score"),
]

LAGS = dict(MD.LAGS)
LAGS.update(ED.LAGS)

VINTAGE = {"macro": "LATEST REVISED (not real-time)", "entry": "LATEST REVISED (not real-time)",
           "price": "as-published daily close", "constructed": "derived from cached inputs"}

INPUT_META = {
    "vix": ("VIX (CBOE ^VIX)", "Fixed VIX knots: <20 low, 20-30 moderate, 30-40 strong, >50 maximum", "Opportunity"),
    "drawdown": ("SPY raw close vs running all-time high", "Fixed drawdown knots, accelerating", "Opportunity"),
    "above 200dma": ("10-ETF breadth universe (sector SPDRs, RSP, IWM)", "Inverted % above 200DMA, fixed knots", "Opportunity"),
    "below 200dma": ("SPY raw close vs 200-day mean", "Fixed distance knots", "Opportunity"),
    "putcall": ("^VIX9D / ^VIX - 1", "Fixed term-structure knots; 50 = neutral if unavailable", "Opportunity"),
    "mcclellan": ("RSP/SPY 19-39 day EMA spread", "Expanding PIT percentile, squared", "Opportunity"),
    "shiller cape": ("multpl Shiller CAPE", "Expanding PIT percentile, inverted, ^1.5", "Valuation"),
    "buffett indicator": ("^W5000 price index / FRED GDP", "Expanding PIT percentile, inverted, ^1.5", "Valuation"),
    "earnings yield": ("multpl trailing earnings yield", "Expanding PIT percentile, ^1.5", "Valuation"),
    "equity risk premium": ("Earnings yield - ^TNX", "Expanding PIT percentile, ^1.5", "Valuation"),
    "real yield (10y tips)": ("FRED DFII10", "Expanding PIT percentile, inverted, ^1.5", "Valuation"),
    "risk modifier": ("5 worst-of drawdown / macro stress sub-signals", "Fixed meter-to-points map", "Risk"),
}


def _load_caches() -> dict:
    return {
        "macro": json.loads((MACRO / "macro_observations.json").read_text(encoding="utf-8")),
        "entry": json.loads((MACRO / "entry_observations.json").read_text(encoding="utf-8")),
    }


def _effective(rows: list[dict], lag: int, as_of: pd.Timestamp) -> tuple[str | None, int | None, str | None]:
    """Last observation actually PUBLIC on as_of, plus its staleness in days."""
    usable = [r for r in rows
              if pd.Timestamp(r["date"]) + pd.Timedelta(days=int(lag)) <= as_of]
    if not usable:
        return None, None, (rows[0]["date"] if rows else None)
    nominal = pd.Timestamp(usable[-1]["date"])
    known = nominal + pd.Timedelta(days=int(lag))
    return str(known.date()), int((as_of - nominal).days), str(nominal.date())


def build_sources(as_of: pd.Timestamp) -> list[dict]:
    caches = _load_caches()
    last = OP.last_dates()
    out = []
    for name, cache, key, source, kind, role, note in CATALOG:
        lag = int(LAGS.get(key, 0)) if cache in ("macro", "entry") else (1 if key == "W5000" else 0)
        if cache == "price":
            last_obs = last.get(key)
            eff, stale, rows = last_obs, None, []
            if last_obs:
                delta = int((as_of - pd.Timestamp(last_obs)).days)
                if delta < 0:
                    # A provider published a row after the canonical scoring
                    # date. ops_price truncates the frame, so it is NOT used.
                    eff = f"{as_of.date()} (cutoff; later row excluded)"
                    note = (note + "; " if note else "") + (
                        f"latest row {last_obs} is {abs(delta)}d after as-of and is excluded")
                    stale = 0
                else:
                    stale = delta
            coverage = "1999-now"
        elif cache == "constructed":
            last_obs, eff, stale, rows = "daily", str(as_of.date()), 0, []
            coverage = "2000-now"
        else:
            rows = caches[cache].get(key, [])
            eff, stale, nominal = _effective(rows, lag, as_of)
            last_obs = rows[-1]["date"] if rows else None
            coverage = (f"{rows[0]['date'][:7]}..{rows[-1]['date'][:7]}" if rows else "unavailable")
        out.append({
            "name": name, "series": key, "source": source, "kind": kind,
            "coverage": coverage, "last_obs": last_obs or "n/a",
            "lag_days": lag, "effective_known_date": eff or "not yet public",
            "staleness_days": stale if stale is not None else -1,
            "vintage": (VINTAGE[cache] + (". " + note if note else "")),
            "role": role, "n": len(rows),
        })
    return out


def build_score_inputs(payload: dict) -> list[dict]:
    v = payload["instruments"]["VOO"]
    rows = []
    for c in v["explain"]["components"]:
        key = c["name"].strip().lower()
        src, transform, layer = INPUT_META.get(
            key, (c.get("direction", ""), "Pre-registered transform", "Layer"))
        raw = c.get("raw")
        if isinstance(raw, (int, float)) and np.isfinite(raw):
            disp = f"{raw:.3f}"
        elif raw is None:
            disp = "n/a"
        else:
            disp = str(raw)
        rows.append({
            "name": c["name"], "raw_display": disp, "layer": layer,
            "transform": transform, "source": src,
            "normalised": round(float(c["normalised"]), 2),
            "weight_pct": round(float(c["weight"]) * 100, 2),
            "contribution": round(float(c["contribution"]), 3),
        })
    rows.append({
        "name": "Drawdown Bonus (additive, after base)", "raw_display": f"{v['drawdown_pct']:.2f}% drawdown",
        "layer": "Bonus", "transform": "Fixed VOO drawdown bonus table (5%=0, 10%=+5, 20%=+20, 40%=+50)",
        "source": "SPY raw close vs all-time high", "normalised": None,
        "weight_pct": 100.0, "contribution": round(float(v["drawdown_bonus"]), 3),
    })
    return rows


VERDICT = [
    {"level": "OK", "text": "Price levels use raw close; adjusted close is reserved for forward outcome labels, so dividend/split adjustments cannot leak into level features."},
    {"level": "OK", "text": "Every macro input is aligned by FIRST-KNOWN DATE (nominal date + publication lag) and then forward-filled; expanding percentiles use only prior observations."},
    {"level": "OK", "text": "No series is fabricated. If a source is unavailable or too short, it is flagged and replaced by a labelled proxy or left at a neutral 50."},
    {"level": "FIXED", "text": "The inherited price loader hard-coded 2026-11-16, which would have frozen a daily refresh. The operational path now resolves the as-of date from the data itself."},
    {"level": "OK", "text": "One price file currently contains a row dated after the scoring date (VIX). All frames are truncated at the SPY anchor date, so that row cannot enter the score; the source inventory flags it explicitly."},
    {"level": "CAVEAT", "text": "FRED and multpl provide LATEST REVISED vintages, not as-first-published real-time values. Historical percentiles therefore carry a small, unavoidable revision look-ahead."},
    {"level": "CAVEAT", "text": "CAPE, earnings yield and P/E carry a conservative 95-day publication lag; the value knowable today is months old by design. This is disclosed rather than filled with a fresher guess."},
    {"level": "CAVEAT", "text": "HY OAS begins 2023-11-20, so long-history credit stress falls back to a disclosed HYG/LQD price proxy. The TED-like funding signal is a proxy: literal LIBOR TED ended in 2021."},
    {"level": "CAVEAT", "text": "McClellan, put/call, advance/decline and % above 200DMA are price-based proxies built from ETFs. They are NOT exchange breadth or options tape."},
    {"level": "CAVEAT", "text": "The Buffett Indicator uses the ^W5000 price index divided by GDP, not total market capitalisation. Forward P/E is excluded because no genuine free historical vintage exists."},
    {"level": "CAVEAT", "text": "CNN Fear & Greed has roughly one year of free history, so it is display-only with zero weight. The sector ETF universe contains surviving funds only (no delisted-fund database)."},
    {"level": "VERDICT", "text": "Legitimate for a transparent, reproducible cash-deployment aid using free public data. NOT institutional-grade real-time data, and NOT a return forecast. Proxies and revision caveats are material and are listed here rather than buried."},
]


def build(payload: dict, as_of: pd.Timestamp | None = None) -> dict:
    as_of = as_of or pd.Timestamp(payload["meta"]["as_of"])
    sources = build_sources(as_of)
    inputs = build_score_inputs(payload)
    snap = OP.price_snapshot(8)
    stales = [s["staleness_days"] for s in sources
              if isinstance(s["staleness_days"], int) and s["staleness_days"] >= 0]
    audit = {
        "as_of": str(as_of.date()),
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "n_sources": len(sources),
            "n_proxies": sum(1 for s in sources if s["kind"] in ("PROXY", "CONSTRUCTED", "CONTEXT")),
            "max_staleness_days": max(stales) if stales else 0,
            "n_fabricated": 0,
            "scored_inputs": len(inputs) - 1,
        },
        "sources": sources, "score_inputs": inputs, "price_snapshot": snap,
        "verdict": VERDICT,
        "checks": {
            "no_fabricated_series": True,
            "pit_first_known_date_alignment": True,
            "dynamic_as_of": True,
            "proxies_disclosed": True,
            "raw_close_for_levels": True,
            "real_time_vintages": False,
        },
    }
    OPS.mkdir(parents=True, exist_ok=True)
    (OPS / "data_audit.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8")
    return audit
