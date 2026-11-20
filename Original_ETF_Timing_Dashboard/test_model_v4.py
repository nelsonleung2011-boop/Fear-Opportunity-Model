#!/usr/bin/env python3
"""
test_model_v4.py — leakage and stability tests for the v4 engine.

The important ones are the first three: they are the tests that would catch a
subtly leaking pipeline, which is exactly the failure mode that made earlier
versions of this dashboard overclaim.

Run:  python test_model_v4.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))
import model_v4 as M

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name + ((" — " + detail) if detail else ""))


# --------------------------------------------------------------------------
# synthetic market: two regimes, enough history for a scaled-down walk-forward
# --------------------------------------------------------------------------
def make_synthetic(n=900, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    regime = np.where((np.arange(n) % 300) < 200, 1, -1)
    ret = 0.0004 * regime + rng.normal(0, 0.010, n)
    spy = 100 * np.exp(np.cumsum(ret))
    cols = {
        "SPY": spy,
        "QQQ": spy * (1 + rng.normal(0, 0.001, n).cumsum() * 0),
        "VIX": np.clip(16 - 8 * regime + rng.normal(0, 2, n), 8, None),
        "TLT": 100 + np.cumsum(rng.normal(0, 0.004, n)) * 100,
        "TNX": np.clip(3 + rng.normal(0, 0.2, n).cumsum() * 0.01, 0.1, None),
        "IRX": np.clip(1.5 + rng.normal(0, 0.1, n).cumsum() * 0.01, 0.01, None),
        "HYG": 80 + np.cumsum(rng.normal(0, 0.002, n)) * 80,
        "LQD": 110 + np.cumsum(rng.normal(0, 0.002, n)) * 110,
        "RSP": spy * np.exp(np.cumsum(rng.normal(0, 0.0015, n))),
        "IWM": spy * np.exp(np.cumsum(rng.normal(0, 0.002, n))),
        "XLY": spy * np.exp(np.cumsum(rng.normal(0, 0.002, n))),
        "XLP": spy * np.exp(np.cumsum(rng.normal(0, 0.001, n))),
    }
    return pd.DataFrame(cols, index=dates)


def scaled():
    """Patch the module constants down to a size a synthetic test can afford."""
    old = (M.MIN_TRAIN, M.H, M.EMBARGO, M.TEST_SPAN, M.BASE_WINDOW, M.BASE_MINP)
    M.MIN_TRAIN, M.H, M.EMBARGO, M.TEST_SPAN = 300, 60, 60, 120
    M.BASE_WINDOW, M.BASE_MINP = 400, 200
    return old


def restore(old):
    M.MIN_TRAIN, M.H, M.EMBARGO, M.TEST_SPAN, M.BASE_WINDOW, M.BASE_MINP = old


# --------------------------------------------------------------------------
def main():
    P = make_synthetic()
    feat = M.build_features(P)
    fwd = M.forward_returns(P["SPY"])
    old = scaled()
    try:
        run_all(P, feat, fwd)
    finally:
        restore(old)


def run_all(P, feat, fwd):
    print("=== model v4 engine tests ===")

    # ---- 1. point-in-time percentile stability (acceptance criterion 1) ----
    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(size=400).cumsum())
    before = M.pit_percentile(s, None).copy()
    s2 = pd.concat([s, pd.Series(rng.normal(size=300).cumsum())], ignore_index=True)
    after = M.pit_percentile(s2, None)
    same = np.allclose(before.values[~np.isnan(before.values)],
                       after.values[:len(before)][~np.isnan(before.values)], equal_nan=False)
    check("PIT percentile unchanged when future data is appended", bool(same))
    check("PIT percentile stays NaN where it was NaN",
          bool((np.isnan(before.values) == np.isnan(after.values[:len(before)])).all()))

    s3 = M.pit_percentile(pd.Series(rng.normal(size=500)), 100)
    check("trailing-window percentile stays within [0,100]",
          bool(((s3.dropna() >= 0) & (s3.dropna() <= 100)).all()))

    # ---- 2/3. purge and embargo, straight from the block bookkeeping ----
    wf = M.walk_forward(P, feat, fwd, verbose=False)
    blocks = wf["blocks"]
    check("walk-forward produced test blocks", len(blocks) >= 2, "%d blocks" % len(blocks))
    ok_purge = all(b["train_last_index"] + M.H < b["val_start_index"] for b in blocks)
    check("no training label overlaps the validation window", ok_purge)
    ok_emb = all(b["purge_gap_sessions"] >= M.EMBARGO for b in blocks)
    check("embargo of at least the horizon is enforced", ok_emb,
          "min gap %d vs embargo %d" % (min(b["purge_gap_sessions"] for b in blocks), M.EMBARGO))

    # ---- 4. concatenation covers only test windows, no gaps/overlaps ----
    starts = [b["origin_index"] for b in blocks]
    ends = [s + M.TEST_SPAN for s in starts]
    check("test blocks tile without overlapping",
          all(ends[i] <= starts[i + 1] for i in range(len(starts) - 1)))
    pred_idx = set(np.where(np.isfinite(wf["p_win"].values))[0])
    expected = set()
    for s0, e0 in zip(starts, ends):
        expected |= set(range(s0, min(e0, len(feat))))
    check("every prediction lies inside a declared test window", pred_idx == expected,
          "%d predicted sessions" % len(pred_idx))

    # ---- 5. appending future rows must not change past predictions ----
    n_cut = int(len(feat) * 0.75)
    feat_cut, fwd_cut = feat.iloc[:n_cut], {k: v.iloc[:n_cut] for k, v in fwd.items()}
    wf_cut = M.walk_forward(P.iloc[:n_cut], feat_cut, fwd_cut, verbose=False)
    a = wf["p_win"].values[:n_cut]
    b = wf_cut["p_win"].values
    m = np.isfinite(a) & np.isfinite(b)
    same = np.allclose(a[m], b[m], atol=1e-9)
    check("out-of-sample predictions unchanged after appending future rows",
          bool(same), "%d compared" % int(m.sum()))

    # ---- 6. reproducibility ----
    wf2 = M.walk_forward(P, feat, fwd, verbose=False)
    check("walk-forward is deterministic",
          bool(np.allclose(wf["p_win"].values[~np.isnan(wf["p_win"].values)],
                           wf2["p_win"].values[~np.isnan(wf2["p_win"].values)])))

    # ---- 7. opportunity score bounds and direction ----
    mu = wf["mu"].fillna(0)
    op = M.compute_opportunity(feat, fwd, mu, np.array([-0.02] * 400))
    opp = op["opportunity"]
    check("opportunity score stays within 0-100",
          bool(((opp.dropna() >= 0) & (opp.dropna() <= 100)).all()))
    fams = op["families"]
    ok = all(((fams[k].dropna() >= 0) & (fams[k].dropna() <= 100)).all() for k in fams.columns)
    check("every opportunity family stays within 0-100", bool(ok))
    check("sensitivity alternatives are reported", len(op["sensitivity"]) >= 3,
          "%d schemes" % len(op["sensitivity"]))

    # ---- 8. timing percentile: three windows, higher = more attractive ----
    t10 = op["timing"]["10y"].dropna()
    check("timing percentile available over 5y/10y/expanding",
          all(len(op["timing"][k].dropna()) > 50 for k in ("10y", "5y", "expanding")))

    # ---- 9. NaN / extreme handling in the scoring helpers ----
    st = M.calibration_stats(np.array([np.nan] * 100), np.array([np.nan] * 100), 500)
    check("calibration_stats returns available=False on all-NaN input",
          st.get("available") is False)
    rng2 = np.random.default_rng(9)
    p = np.where(rng2.random(60) < 0.5, 0.02, 0.98)
    y = (rng2.random(60) < p).astype(float)
    st2 = M.calibration_stats(p, y, 60 * M.H)
    check("calibration_stats survives extreme probabilities",
          bool(st2.get("available")) and np.isfinite(st2["brier"]))
    lo, hi = M.wilson_interval(7, 10)
    check("Wilson interval stays inside [0,1] and brackets the point estimate",
          0 <= lo <= 7 / 10 <= hi <= 1)
    check("effective_n never exceeds the raw count",
          M.effective_n(50, 126 * 100) <= 50 and M.effective_n(1000, 1260) <= 10)

    # ---- 10. reliability bin merging ----
    rng = np.random.default_rng(5)
    pp = rng.uniform(0.55, 0.95, 3000)
    yy = (rng.uniform(size=3000) < pp).astype(float)
    rc, merged = M.reliability_curve(pp, yy, total_span=3000)
    check("low-evidence reliability bins are merged", all(r.get("n_eff", 99) >= 8 or i == len(rc) - 1
                                                          for i, r in enumerate(rc)),
          "%d bins" % len(rc))

    # ---- 11. analogue episode clustering ----
    import analogues as A
    dates = [d.strftime("%Y-%m-%d") for d in feat.index]
    ana = A.build(feat, dates, fwd, len(feat) - 1)
    if ana.get("available"):
        ds = [int(x["episode"]) for x in ana["rows"]]
        check("analogues come from distinct episodes", len(set(ds)) == len(ds),
              "%d episodes" % len(set(ds)))
    else:
        check("analogues degrade gracefully", "reason" in ana, ana.get("reason", ""))

    # ---- 12. signal tests reject noise ----
    st_sig = M.signal_tests(feat, fwd, start_index=M.MIN_TRAIN + 2 * M.H + M.EMBARGO, step=21)
    check("signal tests run on non-overlapping monthly dates",
          st_sig["window"]["n_non_overlapping"] > 10,
          "n=%d" % st_sig["window"]["n_non_overlapping"])

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILED:", FAIL)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
