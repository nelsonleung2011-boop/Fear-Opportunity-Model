#!/usr/bin/env python3
"""Alert rules, email rendering and delivery for the v7 operational system.

Trigger policy (deliberately narrow — no daily spam):

  threshold_crossed   VOO or QQQ crosses from <80 to >=80
  regime_change       Market Regime label changes (PANIC is urgent)
  detector_triggered  Rare Opportunity Detector transitions into YES
  weekly_summary      once per ISO week

A repeated run on the same day, an unchanged regime, a score that stays high
and a detector that stays YES all produce NO email. State is persisted in
alert_state.json, so running the pipeline twice is idempotent.

Email delivery is optional: with no SMTP_* environment variables the message is
written to data/ops/outbox/ as .html/.txt and marked dry-run. Nothing is sent.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "data" / "ops"
STATE = OPS / "alert_state.json"
OUTBOX = OPS / "outbox"
LOG = OPS / "alert_log.jsonl"

THRESHOLD = 80.0
RULE_ORDER = ["threshold_crossed", "regime_change", "detector_triggered", "weekly_summary"]
RULE_TRIGGERS = {
    "threshold_crossed": "VOO or QQQ crosses 80 from below",
    "regime_change": "Market Regime label changes (any change; PANIC is urgent)",
    "detector_triggered": "Rare Opportunity Detector transitions into YES",
    "weekly_summary": "First run of each ISO week",
}


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {"last": {}, "last_fired": {}, "last_weekly": None,
            "consecutive_yes": 0, "runs": 0}


def save_state(state: dict) -> None:
    OPS.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")


def snapshot(payload: dict) -> dict:
    return {
        "date": payload["meta"]["as_of"],
        "voo": float(payload["instruments"]["VOO"]["score"]),
        "qqq": float(payload["instruments"]["QQQ"]["score"]),
        "regime": payload["hero"]["market_regime"],
        "detector": payload["instruments"]["VOO"]["detector"]["state"],
        "deployment": payload["hero"]["deployment"],
    }


def evaluate(payload: dict, state: dict, now: dt.datetime | None = None,
             force_summary: bool = False) -> tuple[list[dict], dict]:
    """Pure state transition logic. Returns (alerts, updated_state)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    cur = snapshot(payload)
    prev = state.get("last") or {}
    st = dict(state)
    st.setdefault("last_fired", {})
    alerts: list[dict] = []

    if prev:
        for sym in ("voo", "qqq"):
            before, after = float(prev.get(sym, 0.0)), cur[sym]
            if before < THRESHOLD <= after:
                alerts.append({
                    "type": "threshold_crossed", "priority": "high",
                    "detail": f"{sym.upper()} {before:.1f} -> {after:.1f} crossed {THRESHOLD:.0f}",
                })
        if cur["regime"] != prev.get("regime"):
            alerts.append({
                "type": "regime_change",
                "priority": "urgent" if cur["regime"] == "PANIC" else "normal",
                "detail": f"Regime {prev.get('regime', 'n/a')} -> {cur['regime']}",
            })
        if cur["detector"] == "YES" and prev.get("detector") != "YES":
            alerts.append({
                "type": "detector_triggered", "priority": "urgent",
                "detail": f"Rare detector {prev.get('detector', 'n/a')} -> YES "
                          f"({payload['instruments']['VOO']['detector']['n_conditions']}/5 conditions)",
            })

    iso = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
    if force_summary or st.get("last_weekly") != iso:
        alerts.append({"type": "weekly_summary", "priority": "normal",
                       "detail": f"Weekly summary for {iso}"})
        st["last_weekly"] = iso

    st["consecutive_yes"] = (st.get("consecutive_yes", 0) + 1) if cur["detector"] == "YES" else 0
    st["runs"] = st.get("runs", 0) + 1
    st["last"] = cur
    st["last_evaluated"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    for a in alerts:
        st["last_fired"][a["type"]] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        a["time"] = st["last_evaluated"]
    return alerts, st


def rules_view(payload: dict, state: dict) -> list[dict]:
    cur = snapshot(payload)
    fired = state.get("last_fired", {})
    return [
        {"name": "VOO >= 80", "trigger": RULE_TRIGGERS["threshold_crossed"],
         "current": f"{cur['voo']:.1f}", "state": "YES" if cur["voo"] >= 80 else "NO",
         "last_fired": fired.get("threshold_crossed")},
        {"name": "QQQ >= 80", "trigger": "QQQ crosses 80 from below",
         "current": f"{cur['qqq']:.1f}", "state": "YES" if cur["qqq"] >= 80 else "NO",
         "last_fired": fired.get("threshold_crossed")},
        {"name": "Rare detector", "trigger": RULE_TRIGGERS["detector_triggered"],
         "current": cur["detector"], "state": cur["detector"],
         "last_fired": fired.get("detector_triggered")},
        {"name": "Regime change", "trigger": RULE_TRIGGERS["regime_change"],
         "current": cur["regime"], "state": "OK",
         "last_fired": fired.get("regime_change")},
        {"name": "Weekly summary", "trigger": RULE_TRIGGERS["weekly_summary"],
         "current": state.get("last_weekly") or "not sent", "state": "OK",
         "last_fired": fired.get("weekly_summary")},
    ]


def _lines(items: list[dict], key: str) -> str:
    if not items:
        return "none"
    return "; ".join(f"{x['name']} ({float(x.get(key, 0)):+.1f})" for x in items)


def email_fields(payload: dict, pct: dict) -> dict:
    v = payload["instruments"]["VOO"]
    ex = v["explain"]
    pos = [{"name": p["name"], "v": float(p["contribution"])} for p in ex["top_positive"]]
    neg = [{"name": p["name"], "v": float(p["delta_vs_neutral"])} for p in ex["top_negative"]]
    return {
        "Date": payload["meta"]["as_of"],
        "Current VOO Score": f"{v['score']:.1f}/100 ({v['label']})",
        "Current QQQ Score": f"{payload['instruments']['QQQ']['score']:.1f}/100 "
                             f"({payload['instruments']['QQQ']['label']})",
        "Historical Percentile": f"VOO {pct.get('VOO', 50.0):.1f}th / "
                                 f"QQQ {pct.get('QQQ', 50.0):.1f}th "
                                 f"(month-end entry dates since 2000)",
        "Top Positive Contributors": _lines(pos, "v"),
        "Top Negative Contributors": _lines(neg, "v"),
        "Current Regime": f"{payload['hero']['market_regime']} "
                          f"({payload['hero']['regime_confidence']} confidence)",
        "Historical Analogues": "; ".join(
            f"{a['episode']} {a['similarity']:.0f}%" for a in payload["analogues"][:5]),
        "Deployment Recommendation": f"{payload['hero']['deployment']} — "
                                     f"{payload['hero']['deployment_label']}",
    }


def render_email(alert: dict, payload: dict, pct: dict) -> dict:
    f = email_fields(payload, pct)
    label = alert["type"].replace("_", " ").upper()
    subject = (f"[ETF Entry] {label} — VOO {f['Current VOO Score'].split('/')[0]} "
               f"| {f['Current Regime'].split(' (')[0]} | {f['Date']}")
    rows = "".join(
        f'<tr><td style="padding:7px 12px;border-bottom:1px solid #e3e8ec;'
        f'color:#5a6a78;font-weight:600;width:210px">{html.escape(k)}</td>'
        f'<td style="padding:7px 12px;border-bottom:1px solid #e3e8ec;color:#16202b">'
        f'{html.escape(str(v))}</td></tr>' for k, v in f.items())
    body = (
        '<div style="font-family:Segoe UI,Arial,sans-serif;max-width:720px;margin:0 auto;'
        'background:#f4f6f8;padding:22px">'
        '<div style="background:#fff;border:1px solid #dce3e9;border-radius:6px;padding:22px">'
        '<div style="font-size:11px;letter-spacing:.12em;color:#1d6f8c;font-weight:700">'
        'ETF ENTRY DECISION SYSTEM · ALERT</div>'
        f'<h1 style="font-size:20px;margin:8px 0 4px">{html.escape(label)}</h1>'
        f'<p style="font-size:12px;color:#5a6a78;margin:0 0 16px">'
        f'{html.escape(alert["detail"])}</p>'
        f'<table style="width:100%;border-collapse:collapse;font-size:13px">{rows}</table>'
        '<p style="font-size:11px;color:#5a6a78;line-height:1.6;margin-top:18px">'
        'This is a cash-deployment aid for long-term investors, based on observable '
        'conditions rather than return forecasts. Not investment advice. '
        'Sent only on a threshold crossing, regime change, detector trigger or '
        'the weekly summary — never on an unchanged day.</p>'
        '</div></div>')
    text = "\n".join([label, alert["detail"], ""] + [f"{k}: {v}" for k, v in f.items()]
                     + ["", "Observable conditions, not a return forecast. Not investment advice."])
    return {"subject": subject, "html": body, "text": text, "fields": f,
            "type": alert["type"], "priority": alert["priority"], "detail": alert["detail"],
            "time": alert.get("time", "")}


def smtp_config() -> dict:
    host = os.environ.get("SMTP_HOST", "")
    user = os.environ.get("SMTP_USER", "")
    pwd = os.environ.get("SMTP_PASS", "")
    to = os.environ.get("ALERT_TO", "")
    configured = bool(host and user and pwd and to)
    return {"host": host, "port": int(os.environ.get("SMTP_PORT", "465")),
            "user": user, "password": pwd, "to": [x.strip() for x in to.split(",") if x.strip()],
            "from": os.environ.get("SMTP_FROM", user),
            "starttls": os.environ.get("SMTP_STARTTLS", "0") == "1",
            "configured": configured}


def deliver(email: dict, dry_run: bool = False) -> dict:
    """Always archive the message. Send only with a full SMTP configuration."""
    OUTBOX.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = OUTBOX / f"{stamp}_{email['type']}"
    base.with_suffix(".html").write_text(email["html"], encoding="utf-8")
    base.with_suffix(".txt").write_text(email["text"], encoding="utf-8")
    base.with_suffix(".json").write_text(json.dumps(
        {"subject": email["subject"], "type": email["type"], "priority": email["priority"],
         "detail": email["detail"], "fields": email["fields"]}, indent=1), encoding="utf-8")
    cfg = smtp_config()
    if dry_run or not cfg["configured"]:
        return {"delivery": "dry-run", "reason": "explicit dry-run" if dry_run else
                "SMTP_HOST / SMTP_USER / SMTP_PASS / ALERT_TO not all set",
                "outbox": str(base.with_suffix(".html"))}
    try:
        msg = EmailMessage()
        msg["Subject"] = email["subject"]
        msg["From"] = cfg["from"]
        msg["To"] = ", ".join(cfg["to"])
        msg.set_content(email["text"])
        msg.add_alternative(email["html"], subtype="html")
        if cfg["starttls"]:
            with smtplib.SMTP(cfg["host"], cfg["port"], timeout=25) as s:
                s.starttls()
                s.login(cfg["user"], cfg["password"])
                s.send_message(msg)
        else:
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=25) as s:
                s.login(cfg["user"], cfg["password"])
                s.send_message(msg)
        return {"delivery": "sent", "to": cfg["to"], "outbox": str(base.with_suffix(".html"))}
    except Exception as exc:
        return {"delivery": "error", "reason": type(exc).__name__ + ": " + str(exc)[:160],
                "outbox": str(base.with_suffix(".html"))}


def append_log(entry: dict) -> None:
    OPS.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def last_outbox_email() -> dict:
    """Most recent archived message, so the Ops tab can preview a real email
    even on a day when no alert fired."""
    if not OUTBOX.exists():
        return {}
    files = sorted(OUTBOX.glob("*.json"))
    if not files:
        return {}
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def recent_log(n: int = 20) -> list[dict]:
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text(encoding="utf-8").splitlines()[-n:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out
