"""Append-only decision log.

Every publish, edit and rejection is written here before anything leaves the
system. Three reasons this is not an afterthought:

* Accountability. If a wrong recommendation reaches 40,000 subscribers, the
  team needs to answer "who approved this, when, on what evidence" without
  reconstructing it from memory.
* Feedback. Editor overrides are the only free source of labelled data this
  system generates. A rejection with a reason is a training signal.
* Trust. An operator who can see what the system did last week is far more
  willing to let it act this week.

JSONL on disk is deliberate: it is append-only by construction, survives a
Streamlit restart, needs no database, and is trivially exportable. In
production this would be Postgres with the same shape.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from core.config import AUDIT_LOG_PATH

ACTIONS = ["PUBLISHED", "EDITED_AND_PUBLISHED", "REJECTED", "ESCALATED", "AUTO_BLOCKED"]


def _fingerprint(payload: dict[str, Any]) -> str:
    """Stable hash of the decision content, so a log line cannot be quietly
    edited afterwards without the hash disagreeing."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def record(
    action: str,
    reviewer: str,
    summary: str,
    payload: dict[str, Any],
    reason: str = "",
    ai_recommendation: str = "",
    human_changed_it: bool = False,
) -> dict[str, Any]:
    if action not in ACTIONS:
        raise ValueError(f"Unknown action {action!r}; expected one of {ACTIONS}")

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action": action,
        "reviewer": reviewer,
        "summary": summary,
        "reason": reason,
        "ai_recommendation": ai_recommendation,
        "human_changed_it": human_changed_it,
        "payload": payload,
    }
    entry["fingerprint"] = _fingerprint(entry)

    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return entry


def read_all(limit: int | None = None) -> list[dict[str, Any]]:
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    rows: list[dict[str, Any]] = []
    with open(AUDIT_LOG_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.reverse()
    return rows[:limit] if limit else rows


def stats() -> dict[str, Any]:
    """Aggregate the log into the numbers that tell you whether the human/AI
    boundary is drawn in the right place.

    Override rate is the key metric. Near zero means the human is rubber
    stamping and the review step is theatre. Very high means the AI is not
    good enough to be drafting. Somewhere in between means the split is
    working.
    """
    rows = read_all()
    if not rows:
        return {"total": 0}

    by_action: dict[str, int] = {}
    for r in rows:
        by_action[r["action"]] = by_action.get(r["action"], 0) + 1

    decided = [r for r in rows if r["action"] in
               ("PUBLISHED", "EDITED_AND_PUBLISHED", "REJECTED")]
    changed = [r for r in decided if r.get("human_changed_it")]

    return {
        "total": len(rows),
        "by_action": by_action,
        "decisions": len(decided),
        "override_rate": round(len(changed) / len(decided), 3) if decided else 0.0,
        "reviewers": sorted({r["reviewer"] for r in rows}),
        "first": rows[-1]["ts"],
        "last": rows[0]["ts"],
    }


def clear() -> None:
    """Reset the log. Exposed only for demo resets, never for production use."""
    if os.path.exists(AUDIT_LOG_PATH):
        os.remove(AUDIT_LOG_PATH)
