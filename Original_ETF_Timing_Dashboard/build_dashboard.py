#!/usr/bin/env python3
"""
One-shot rebuild of ETF_Timing_Dashboard.html.

Usage (from anywhere):
    python build_dashboard.py

Steps performed:
    1. Run the pipeline (pipeline/fetch_market_data.py + validate_data.py +
       build_market_data.py) -> writes data/*.json + dashboard_data.json with
       the redesigned probability-edge score, sector and backtest engines.
    2. Inject dashboard_data.json into dashboard_template.html at the
       __DASH_DATA__ placeholder -> writes ETF_Timing_Dashboard.html and
       an identical index.html (GitHub Pages + one-click launcher entry point).
    3. Extract inline JS and run `node --check` if Node is available.

Set SKIP_PIPELINE=1 to inject the existing dashboard_data.json without
re-fetching (used by the test harness for deterministic offline runs).

All paths are resolved relative to this script's folder, so the working
directory does not matter.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def p(name):
    return os.path.join(HERE, name)


def main():
    # Step 1: rebuild the embedded data payload via the fetch->validate->build pipeline
    if os.environ.get("SKIP_PIPELINE") == "1":
        print("[1/3] SKIP_PIPELINE=1 - using existing dashboard_data.json")
    else:
        print("[1/3] Running data pipeline (fetch -> validate -> build) ...")
        for step in ("fetch_market_data.py", "validate_data.py", "build_market_data.py"):
            print("      -> pipeline/%s" % step)
            subprocess.run([sys.executable, os.path.join(HERE, "pipeline", step)],
                           cwd=HERE, check=True)

    # Step 2: inject payload into template
    print("[2/3] Injecting payload into dashboard_template.html ...")
    payload = open(p("dashboard_data.json"), encoding="utf-8").read().strip()
    template = open(p("dashboard_template.html"), encoding="utf-8").read()
    if "__DASH_DATA__" not in template:
        raise RuntimeError("Template is missing the __DASH_DATA__ placeholder")
    html = template.replace("__DASH_DATA__", payload)
    out = p("ETF_Timing_Dashboard.html")
    open(out, "w", encoding="utf-8").write(html)
    leftover = re.findall(r"__[A-Z_]+__", html)
    if leftover:
        raise RuntimeError("Unresolved placeholders: %s" % leftover)
    print("      wrote %s (%.1f KB)" % (os.path.basename(out), os.path.getsize(out) / 1024))

    # Also publish an index.html copy: GitHub Pages serves this automatically
    # from the repository root, and the Windows launcher opens it at "/".
    index = p("index.html")
    open(index, "w", encoding="utf-8").write(html)
    print("      wrote %s (%.1f KB) for local server + GitHub Pages"
          % (os.path.basename(index), os.path.getsize(index) / 1024))

    # Step 3: JS syntax validation (optional; needs Node)
    print("[3/3] Validating inline JavaScript ...")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    check_js = p("_dash_check.tmp.js")
    open(check_js, "w", encoding="utf-8").write("\n".join(scripts))
    try:
        subprocess.run(["node", "--check", check_js], check=True)
        print("      JS syntax OK")
    except FileNotFoundError:
        print("      node not found - skipped (install Node to enable this check)")
    except subprocess.CalledProcessError:
        print("      WARNING: JS syntax check FAILED")
    finally:
        if os.path.exists(check_js):
            os.remove(check_js)

    print("\nDone. Open ETF_Timing_Dashboard.html in a browser.")


if __name__ == "__main__":
    main()
