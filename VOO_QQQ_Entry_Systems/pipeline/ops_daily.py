#!/usr/bin/env python3
"""V7 OPERATIONAL VERSION — one daily entry point.

    python pipeline/ops_daily.py

Pipeline:
  1. refresh prices + macro (isolated per source, cached data survives failures)
  2. recompute the v7 score with a dynamically resolved as-of date
  3. rank today's score against point-in-time month-end scores since 2000
  4. rebuild the explainability / data-provenance ledger
  5. evaluate alert rules and deliver email (dry-run without SMTP credentials)
  6. write ops status + rebuild the self-contained dashboard

Flags:
  --no-refresh       score only, using cached data (always safe)
  --offline          same as --no-refresh, labelled offline
  --dry-run          never send email; always archive to outbox
  --force-summary    force the weekly summary email (setup test)
  --as-of YYYY-MM-DD freeze the scoring date (replay / audit)
  --rebuild-history  force a full historical score backfill
  --no-build         skip the HTML rebuild

Idempotent: running twice on the same day does not duplicate history rows and
does not re-send alerts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "data" / "ops"
sys.path.insert(0, str(ROOT))

from pipeline import build_index_v7 as BUILDER       # noqa: E402
from pipeline import entry_decision_engine as V7     # noqa: E402
from pipeline import ops_alerts as AL               # noqa: E402
from pipeline import ops_data_audit as AUDIT         # noqa: E402
from pipeline import ops_history as HIST             # noqa: E402
from pipeline import ops_price as PRICE              # noqa: E402
from pipeline import ops_refresh as REFRESH          # noqa: E402

SETUP_NOTES = [
    "Run once per day, after US market close. A repeat run on the same day is safe.",
    "Emails are sent only when VOO/QQQ cross 80, the regime changes, the rare detector turns YES, or for one weekly summary.",
    "Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM and ALERT_TO to enable delivery; otherwise messages are archived in data/ops/outbox/ as dry-run.",
    "For STARTTLS on port 587 also set SMTP_STARTTLS=1. Credentials are read from environment variables only and are never written into the dashboard.",
    "Windows scheduler: powershell -ExecutionPolicy Bypass -File pipeline/register_daily_task.ps1",
    "Point-in-time safety: macro data is aligned by first-known date; expanding percentiles and risk probabilities use only prior observations.",
]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Daily v7 operational pipeline")
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-summary", action="store_true")
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--rebuild-history", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args(argv)
    OPS.mkdir(parents=True, exist_ok=True)

    offline = args.offline or args.no_refresh
    print("[ops] 1/6 refreshing data ...", flush=True)
    refresh = REFRESH.refresh_all(offline=offline)
    print(f"[ops]   {refresh['status']}: {refresh['sources_updated']} updated, "
          f"{refresh['sources_failed']} kept cached", flush=True)

    print("[ops] 2/6 recomputing score ...", flush=True)
    frames = PRICE.load_frames(as_of=args.as_of)
    payload = V7.run(frames=frames, as_of=args.as_of)
    as_of = pd.Timestamp(payload["meta"]["as_of"])

    print("[ops] 3/6 historical percentile ranking ...", flush=True)
    backfill = HIST.load_backfill(frames, as_of, rebuild=args.rebuild_history)
    pct = {"VOO": HIST.percentile_rank(payload["instruments"]["VOO"]["score"], backfill["voo"]),
           "QQQ": HIST.percentile_rank(payload["instruments"]["QQQ"]["score"], backfill["qqq"])}
    live = HIST.append_live(payload, pct["VOO"], pct["QQQ"])
    print(f"[ops]   VOO {payload['instruments']['VOO']['score']} = {pct['VOO']}th "
          f"| QQQ {payload['instruments']['QQQ']['score']} = {pct['QQQ']}th "
          f"({len(backfill)} historical month-ends)", flush=True)

    print("[ops] 4/6 building data provenance ledger ...", flush=True)
    audit = AUDIT.build(payload, as_of)

    print("[ops] 5/6 evaluating alerts ...", flush=True)
    state = AL.load_state()
    alerts, state = AL.evaluate(payload, state, force_summary=args.force_summary)
    sent = []
    for a in alerts:
        em = AL.render_email(a, payload, pct)
        res = AL.deliver(em, dry_run=args.dry_run)
        entry = {"time": a.get("time"), "type": a["type"], "priority": a["priority"],
                 "detail": a["detail"], "delivery": res["delivery"],
                 "reason": res.get("reason", ""), "subject": em["subject"]}
        AL.append_log(entry)
        sent.append({**entry, "fields": em["fields"]})
        print(f"[ops]   {a['type']} -> {res['delivery']}: {em['subject']}", flush=True)
    if not alerts:
        print("[ops]   no alert conditions met (no email sent)", flush=True)
    AL.save_state(state)

    archived = AL.last_outbox_email()
    preview = sent[-1]["fields"] if sent else archived.get("fields", {})
    cfg = AL.smtp_config()
    hist = HIST.combined_history()
    status = {
        "status": {"last_run": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "mode": "offline (cached)" if offline else "live refresh",
                   "status": "OK", "as_of": str(as_of.date()),
                   "refresh_status": refresh["status"],
                   "sources_updated": refresh["sources_updated"],
                   "sources_failed": refresh["sources_failed"]},
        "percentiles": pct,
        "rules": AL.rules_view(payload, state),
        "recent_alerts": AL.recent_log(25),
        # to_json writes NaN as null; to_dict would emit bare NaN (invalid JSON)
        # and break JSON.parse in the browser.
        "score_history": json.loads(hist.tail(45).to_json(orient="records")),
        "weekly_summary": {
            "week": state.get("last_weekly"),
            "text": (f"Week {state.get('last_weekly')}: VOO {payload['instruments']['VOO']['score']}/100 "
                     f"({pct['VOO']}th percentile), QQQ {payload['instruments']['QQQ']['score']}/100 "
                     f"({pct['QQQ']}th). Regime {payload['hero']['market_regime']}, "
                     f"detector {payload['instruments']['VOO']['detector']['state']}, "
                     f"deployment {payload['hero']['deployment']}. "
                     f"{len(alerts)} alert(s) generated this run."),
        },
        "email_config": {"configured": cfg["configured"],
                         "mode": "live" if cfg["configured"] and not args.dry_run else "dry-run"},
        "email_preview": {
            "subject": (sent[-1]["subject"] if sent
                        else archived.get("subject", "no email generated yet")),
            "latest_type": archived.get("type", ""),
            "date": str(as_of.date()),
            "scores": f"{payload['instruments']['VOO']['score']} / {payload['instruments']['QQQ']['score']}",
            "percentile": f"{pct['VOO']} / {pct['QQQ']}",
            "regime": payload["hero"]["market_regime"],
            "deployment": payload["hero"]["deployment"],
            "fields": preview,
        },
        "setup": {"command": "python pipeline/ops_daily.py", "notes": SETUP_NOTES},
        "history_meta": {"backfill_rows": len(backfill), "live_rows": len(live),
                         "first_backfill": str(backfill["date"].iloc[0]) if len(backfill) else None},
    }
    (OPS / "status.json").write_text(json.dumps(status, indent=1, ensure_ascii=False), encoding="utf-8")

    print("[ops] 6/6 rebuilding dashboard ...", flush=True)
    if not args.no_build:
        BUILDER.main()
    print(f"[ops] DONE | as-of {as_of.date()} | regime {payload['hero']['market_regime']} "
          f"| deployment {payload['hero']['deployment']} | {len(alerts)} email(s)", flush=True)
    return {"payload": payload, "status": status, "audit": audit, "alerts": sent, "percentiles": pct}


if __name__ == "__main__":
    main()
