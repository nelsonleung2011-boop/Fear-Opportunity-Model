#!/usr/bin/env python3
"""v7 engine tests: regime, credit, internals, deployment, explainability."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))
from pipeline import entry_decision_engine as X  # noqa: E402

PASS, FAIL = [], []


def ck(cond, msg):
    (PASS if cond else FAIL).append(msg)


# ---- pure function tests ---------------------------------------------------
ck(X.regime_vote(30, X.VIX_REGIME) == "BEAR", "regime_vote VIX 30 -> BEAR")
ck(X.regime_vote(36, X.VIX_REGIME) == "PANIC", "regime_vote VIX 36 -> PANIC")
ck(X.regime_vote(60, X.ABOVE_REGIME) == "CORRECTION", "breadth 60 -> CORRECTION (band order fixed)")
ck(X.regime_vote(80, X.ABOVE_REGIME) == "BULL", "breadth 80 -> BULL")
ck(X.regime_vote(10, X.ABOVE_REGIME) == "PANIC", "breadth 10 -> PANIC")
ck(X.regime_vote(-0.03, X.TREND_REGIME) == "CORRECTION", "trend -3% -> CORRECTION (band order fixed)")
ck(X.regime_vote(0.02, X.TREND_REGIME) == "BULL", "trend +2% -> BULL")
ck(X.regime_vote(-0.20, X.TREND_REGIME) == "PANIC", "trend -20% -> PANIC")
ck(X.severity_index("PANIC") > X.severity_index("BEAR"), "severity ordering")

ck(X.deployment_for(95)["percent"] == 100, "deploy 95 -> 100%")
ck(X.deployment_for(85)["percent"] == 75, "deploy 85 -> 75%")
ck(X.deployment_for(65)["percent"] == 50, "deploy 65 -> 50%")
ck(X.deployment_for(45)["percent"] == 25, "deploy 45 -> 25%")
ck(X.deployment_for(15)["percent"] == 0, "deploy 15 -> 0%")

ck(X.label_from(90, X.VAL_LABELS) == "Extremely Cheap", "valuation label top")
ck(X.label_from(0, X.VAL_LABELS) == "Bubble-like", "valuation label bottom")
ck(X.band_from(80, X.CREDIT_LABELS) == "Extreme", "credit label 80 -> Extreme")
ck(X.band_from(60, X.CREDIT_LABELS) == "High", "credit label 60 -> High")

parts = [
    {"name": "A", "raw": 1, "normalised": 80, "weight": 0.5, "contribution": 40,
     "direction": "up", "why": "x"},
    {"name": "B", "raw": 2, "normalised": 20, "weight": 0.5, "contribution": 10,
     "direction": "down", "why": "y"},
]
ex = X.explain_score(50, parts, "A*0.5 + B*0.5")
ck(ex["final"] == 50, "explain final")
ck(ex["top_positive"][0]["name"] == "A", "top positive by contribution")
ck(ex["top_negative"][0]["name"] == "B", "top negative by contribution")
ck(ex["components"][0]["delta_vs_neutral"] == 15, "delta vs neutral A = 40 - 25")
ck(ex["components"][1]["delta_vs_neutral"] == -15, "delta vs neutral B = 10 - 25")
ck(ex["audit_trail"][-1]["running_total"] == 50, "audit running total")

# knot scoring inherited and finite
ck(0 <= X.knot_score if False else True, "placeholder") if False else None
kk = X.V6.knot_score(45, X.VIX_KNOTS)
ck(abs(kk - 95) < 1e-9, "VIX knot 45 -> 95")
ck(abs(X.V6.knot_score(1000, X.VIX_KNOTS) - 100) < 1e-9, "knot clamps at 100")

# ---- payload integration tests --------------------------------------------
d = json.loads((ROOT / "data" / "evidence_v7.json").read_text(encoding="utf-8"))

ck(d["meta"]["version"].startswith("7."), "version 7")
ck(d["hero"]["always_answered"] is True, "hero always answered")
for key in ("market_regime", "valuation", "opportunity", "deployment", "reason"):
    ck(key in d["hero"] and d["hero"][key] not in (None, ""), f"hero has {key}")
ck(d["hero"]["market_regime"] in X.REGIME_ORDER, "regime is valid state")
ck(d["regime"]["agreement_pct"] > 0, "regime agreement positive")
ck(sum(d["regime"]["votes"].values()) == 5, "five regime votes")

ck(0 <= d["credit"]["score"] <= 100, "credit score in range")
ck(len(d["credit"]["components"]) == 4, "four credit inputs")
ck(all(abs(c["weight"] - 0.25) < 1e-9 for c in d["credit"]["components"].values()),
   "credit weights equal")
ck(all(0 <= c["normalised"] <= 100 for c in d["credit"]["components"].values()),
   "credit normalised in range")
ck(d["credit"]["components"]["funding_proxy"]["raw"]["irx"] is not None,
   "funding proxy has real IRX (fixed source)")

ck(len(d["internals"]["components"]) == 4, "four internals inputs")
ck(isinstance(d["internals"]["hidden_weakness"], bool), "hidden weakness boolean")

ck(set(d["instruments"]) == {"VOO", "QQQ"}, "two instruments")
for sym in ("VOO", "QQQ"):
    x = d["instruments"][sym]
    ck(0 <= x["score"] <= 100, f"{sym} score in range")
    ck(x["deployment"]["percent"] in (0, 25, 50, 75, 100), f"{sym} deployment valid")
    ck("explain" in x and "components" in x["explain"], f"{sym} explainable")
    ck(len(x["explain"]["audit_trail"]) > 0, f"{sym} audit trail present")
    ck(x["explain"]["final"] == x["score"], f"{sym} explanation shows the FINAL score")
    ck(x["explain"]["base_score"] == x["base_score"], f"{sym} base score disclosed")
    ck(x["explain"]["bonus"] == x["drawdown_bonus"], f"{sym} bonus disclosed")
    ck("drawdown bonus" in x["explain"]["reconciliation"], f"{sym} reconciliation explicit")
    ck(len(x["audit"]) == 5, f"{sym} manual audit 5 rows")

# manual audit reproduces score (allow rounding drift)
v = d["instruments"]["VOO"]
recomputed = sum(r["result"] for r in v["audit"][:-1])
ck(abs(recomputed - v["score"]) <= 0.3, f"VOO audit recomputes score ({recomputed} vs {v['score']})")

# weighted layer audit formula
ck(abs(X.W_VAL * d["valuation"]["score"] + X.W_OPP * d["opportunity"]["score"]
       + X.W_RISK * d["risk"]["opportunity_score"] + v["drawdown_bonus"]
       - v["score"]) <= 0.3, "weighted layer formula reproduces VOO")

ck(len(d["sensitivity"]) == 4, "four sensitivity scenarios")
ck(all(0 <= s["score"] <= 100 for s in d["sensitivity"]), "sensitivity scores in range")
ck(d["sensitivity"][-1]["score"] >= d["sensitivity"][0]["score"],
   "combined panic scenario scores above VIX-only")

oc = v["opportunity_cost"]
ck(len(oc["scenarios"]) == 3, "three opportunity-cost scenarios")
ck([s["correction_pct"] for s in oc["scenarios"]] == [5, 10, 15], "5/10/15 scenarios")
ck(all(s["score_improvement"] >= 0 for s in oc["scenarios"]),
   "waiting never produces a nonsensical negative improvement")
ck(all("status" in s for s in oc["scenarios"]), "each scenario states whether waiting is needed")
ck(all("already reached" in s["status"] for s in oc["scenarios"]
       if s["correction_pct"] / 100 <= oc["current_drawdown_pct"] / 100 + 1e-9),
   "thresholds already exceeded are labelled as reached")
ck(all(0 <= s["historical_freq_pct"] <= 100 for s in oc["scenarios"]), "freq in range")

ck(len(d["analogues"]) == 5, "five analogues")
ck(all(0 <= a["similarity"] <= 100 for a in d["analogues"]), "similarity in range")
ck(len(d["historical_validation"]) >= 5, "historical validation episodes")
ck(len(d["formula_disclosure"]) == 6, "six formula disclosures")
for k, f in d["formula_disclosure"].items():
    for key in ("definition", "formula", "reason", "direction", "example"):
        ck(key in f and f[key], f"formula {k} has {key}")
ck(len(d["disclosures"]) >= 6, "disclosures present")

# Inheritance checks must survive a DAILY REFRESH: v6 is a frozen snapshot,
# whereas v7 recomputes from whatever data is current. Verify the formula is
# inherited structurally, and that the two layers are reproducible from the
# same frames v7 actually used.
from pipeline import entry_opportunity_engine as V6  # noqa: E402
from pipeline import entry_quality_engine as V5      # noqa: E402
from pipeline import ops_price as OP                 # noqa: E402

v6 = json.loads((ROOT / "data" / "evidence_v6.json").read_text(encoding="utf-8"))
ck(pd.Timestamp(d["meta"]["as_of"]) >= pd.Timestamp(v6["meta"]["as_of"]),
   "v7 data is at least as fresh as the v6 snapshot")
ck(X.FEAR_WEIGHTS == V6.FEAR_WEIGHTS, "fear weights inherited unchanged from v6")
ck(X.W_VAL == V6.W_VAL and X.W_OPP == V6.W_FEAR and X.W_RISK == V6.W_RISK,
   "30/50/20 layer weights inherited unchanged from v6")
ck(X.VOO_DD_BONUS == V6.VOO_DD_BONUS and X.QQQ_DD_BONUS == V6.QQQ_DD_BONUS,
   "drawdown bonus tables inherited unchanged from v6")
ck(X.VIX_KNOTS == V6.VIX_KNOTS, "nonlinear VIX knots inherited unchanged")

adj, raw = OP.load_frames(as_of=d["meta"]["as_of"])
pit, _ = V5.build_macro_pit(raw.index)
daily = V5.build_daily_components(raw, pit)
today = pd.Timestamp(raw.index.max())
mc = V5.expanding_pct(daily["mcclellan_neg"], daily.index, 252)
ck(abs(d["valuation"]["score"] - V6.valuation_components(daily, today)["score"]) < 0.06,
   "valuation reproducible from v6 components on current data")
ck(abs(d["opportunity"]["score"] - V6.fear_components(daily, today, mc)["score"]) < 0.06,
   "opportunity reproducible from v6 fear on current data")
ck(today == pd.Timestamp(d["meta"]["as_of"]),
   "as_of equals the dynamically resolved frame cutoff")

print(f"v7 engine: {len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAIL:", f)
if FAIL:
    raise SystemExit(1)
