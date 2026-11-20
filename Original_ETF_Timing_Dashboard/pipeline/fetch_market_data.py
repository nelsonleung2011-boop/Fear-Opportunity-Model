#!/usr/bin/env python3
"""
Server-side market data fetcher for the ETF Timing Dashboard.

Runs in GitHub Actions (or locally). Two providers:
  1. Yahoo Finance chart API  (primary)
  2. Stooq CSV                (fallback when Yahoo fails / returns non-JSON)

Writes raw per-symbol CSV files into data/raw/ with a provenance record.
All validation happens in validate_data.py, downstream.
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error
import datetime as dt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# raw provider symbol -> (Yahoo symbol, Stooq symbol, friendly name)
SYMBOLS = {
    "SPY":  ("SPY",  "spy.us",  "SPDR S&P 500"),
    "QQQ":  ("QQQ",  "qqq.us",  "Invesco QQQ"),
    "VIX":  ("^VIX", None,      "CBOE VIX"),
    "TLT":  ("TLT",  "tlt.us",  "iShares 20+Y Treasury"),
    "TNX":  ("^TNX", None,      "CBOE 10Y Yield"),
    "IRX":  ("^IRX", None,      "CBOE 13W Yield"),
    "HYG":  ("HYG",  "hyg.us",  "iShares High Yield Credit"),
    "LQD":  ("LQD",  "lqd.us",  "iShares IG Credit"),
    "RSP":  ("RSP",  "rsp.us",  "S&P 500 Equal Weight"),
    "IWM":  ("IWM",  "iwm.us",  "iShares Russell 2000"),
    # Sector SPDRs
    "XLK":  ("XLK",  "xlk.us",  "Technology"),
    "XLF":  ("XLF",  "xlf.us",  "Financials"),
    "XLV":  ("XLV",  "xlv.us",  "Health Care"),
    "XLY":  ("XLY",  "xly.us",  "Cons Discretionary"),
    "XLP":  ("XLP",  "xlp.us",  "Cons Staples"),
    "XLI":  ("XLI",  "xli.us",  "Industrials"),
    "XLE":  ("XLE",  "xle.us",  "Energy"),
    "XLB":  ("XLB",  "xlb.us",  "Materials"),
    "XLRE": ("XLRE", "xlre.us", "Real Estate"),
    "XLU":  ("XLU",  "xlu.us",  "Utilities"),
    "XLC":  ("XLC",  "xlc.us",  "Communication Svcs"),
}

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def http_get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_yahoo(yahoo_symbol, start, end, retries=2):
    """
    Return (rows, reported_symbol, provider, basis) via Yahoo chart API, or raise.

    rows are (YYYY-MM-DD, close, adj_close). `adj_close` is dividend- and
    split-adjusted and is the series the model actually uses: using raw `close`
    would understate total return by roughly the dividend yield (~1.8%/yr for
    SPY), which biases every forward-return label downward.
    """
    import urllib.parse as up
    enc = up.quote(yahoo_symbol, safe="")
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/" + enc
        + "?period1=%d&period2=%d&interval=1d&events=div%%2Csplit" % (start, end)
    )
    last = None
    for a in range(retries + 1):
        try:
            raw = http_get(url)
            txt = raw.decode("utf-8", "ignore")
            if not txt.lstrip().startswith("{"):
                raise ValueError("non-JSON response (HTML/blocked)")
            j = json.loads(txt)
            res = j.get("chart", {}).get("result")
            if not res:
                err = j.get("chart", {}).get("error") or {}
                raise ValueError("Yahoo error: %s" % err.get("description", "no result"))
            r0 = res[0]
            reported = r0.get("meta", {}).get("symbol")
            ts = r0.get("timestamp") or []
            ind = r0.get("indicators", {}) or {}
            closes = (ind.get("quote") or [{}])[0].get("close") or []
            adj = []
            if ind.get("adjclose"):
                adj = (ind.get("adjclose") or [{}])[0].get("adjclose") or []
            if len(adj) != len(closes):
                adj = [None] * len(closes)
            rows = []
            has_adj = False
            for t, c, a_ in zip(ts, closes, adj):
                if c is None and a_ is None:
                    continue
                day = dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")
                if a_ is not None:
                    has_adj = True
                rows.append((day,
                             None if c is None else round(float(c), 6),
                             None if a_ is None else round(float(a_), 6)))
            if len(rows) < 30:
                raise ValueError("too few bars (%d)" % len(rows))
            basis = "adjusted" if has_adj else "close"
            return rows, reported, "Yahoo", basis
        except Exception as e:  # noqa: BLE001
            last = e
            if a < retries:
                time.sleep(0.4 * 2 ** a)
    raise last


def fetch_stooq(stooq_symbol, start, end):
    """Fallback provider. Stooq serves CSV; JS-challenge HTML is rejected."""
    import urllib.parse as up
    d1 = dt.datetime.utcfromtimestamp(start).strftime("%Y%m%d")
    d2 = dt.datetime.utcfromtimestamp(end).strftime("%Y%m%d")
    url = "https://stooq.com/q/d/l/?s=%s&i=d&d1=%s&d2=%s" % (up.quote(stooq_symbol), d1, d2)
    raw = http_get(url)
    txt = raw.decode("utf-8", "ignore")
    if "Date" not in txt.split("\n")[0] or "<html" in txt[:200].lower():
        raise ValueError("Stooq returned HTML challenge / no CSV")
    rows = []
    for line in txt.strip().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            v = round(float(parts[4]), 6)
            rows.append((parts[0], v, None))
        except ValueError:
            continue
    if len(rows) < 30:
        raise ValueError("Stooq returned too few bars")
    return rows, stooq_symbol, "Stooq", "close"


def main():
    end = int(time.time())
    start = int(dt.datetime(1999, 1, 1).replace(tzinfo=dt.timezone.utc).timestamp())
    provenance = {}
    for key, (ysym, ssym, name) in SYMBOLS.items():
        rows = None
        errs = []
        try:
            rows, reported, prov, basis = fetch_yahoo(ysym, start, end)
        except Exception as e:  # noqa: BLE001
            errs.append("Yahoo: %s" % e)
            if ssym:
                try:
                    rows, reported, prov, basis = fetch_stooq(ssym, start, end)
                except Exception as e2:  # noqa: BLE001
                    errs.append("Stooq: %s" % e2)
        if rows is None:
            provenance[key] = {"ok": False, "errors": errs, "name": name}
            print("FAIL %-5s %s" % (key, " | ".join(errs)))
            continue
        path = os.path.join(RAW_DIR, key + ".csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("date,close,adj_close\n")
            for day, c, a_ in rows:
                f.write("%s,%s,%s\n" % (
                    day,
                    "" if c is None else "%.6f" % c,
                    "" if a_ is None else "%.6f" % a_,
                ))
        provenance[key] = {
            "ok": True, "provider": prov, "provider_symbol": reported,
            "name": name, "rows": len(rows),
            "first": rows[0][0], "last": rows[-1][0],
            "price_basis": basis,
            "fetched_at": dt.datetime.utcnow().isoformat() + "Z",
            "errors": errs,
        }
        print("OK   %-5s %-6s %-8s %5d bars %s -> %s"
              % (key, prov, basis, len(rows), rows[0][0], rows[-1][0]))
    with open(os.path.join(RAW_DIR, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    ok = sum(1 for v in provenance.values() if v["ok"])
    print("\nFetched %d/%d symbols. Provenance written." % (ok, len(SYMBOLS)))
    return 0 if ok >= 6 else 1


if __name__ == "__main__":
    sys.exit(main())
