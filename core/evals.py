"""Evaluation harness for the verification agent.

The question this answers
-------------------------
Not "is the model accurate", which is the easy question, but "what reaches a
human as a confident wrong answer", which is the one that actually matters
when the output is published to 40,000 people.

Three outcomes are tracked separately, because they have very different costs:

* **Caught** - the agent's verdict was wrong AND it escalated. The safety net
  worked. Costs a minute of an editor's time.
* **Leaked** - the agent's verdict was wrong AND it did not escalate. This is
  the number that matters. It is a wrong claim presented confidently, and it
  is the only outcome that can reach a subscriber.
* **Over-escalated** - the verdict was right but escalated anyway. Pure cost:
  human time spent confirming something the AI already had correct.

Tuning the escalation thresholds trades Leaked against Over-escalated. There
is no setting that minimises both, and pretending otherwise is how these
systems get oversold. The sweep below makes the trade-off explicit so the team
can choose a point on it deliberately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from core.config import COSTS, EVAL_SET_PATH, POLICY, SCENARIO
from core.verify import BLOCKING, Evidence, Sample, aggregate


@dataclass
class EvalResult:
    n: int
    accuracy: float
    blocking_precision: float
    blocking_recall: float
    escalation_rate: float
    caught: int
    leaked: int
    over_escalated: int
    clean_pass: int
    per_case: list[dict[str, Any]] = field(default_factory=list)


def load_eval_set(path: str | None = None) -> list[dict[str, Any]]:
    with open(path or EVAL_SET_PATH, encoding="utf-8") as fh:
        return [c for c in json.load(fh) if not str(c.get("player", "")).startswith("_")]


def _verdict_for(case: dict[str, Any], now: datetime) -> Any:
    samples = [
        Sample(
            s["status"], s.get("confidence", 0.5), s.get("source_url"),
            (now - timedelta(days=s.get("days_old", 3))).strftime("%Y-%m-%d")
            if s.get("source_url") else None,
            s.get("quote"), s.get("reasoning", ""),
        )
        for s in case["samples"]
    ]
    evidence = [
        Evidence(e.get("title", ""), e.get("url", ""), e.get("snippet", ""),
                 (now - timedelta(days=e.get("days_old", 3))).strftime("%Y-%m-%d"))
        for e in case.get("evidence", [])
    ]
    return aggregate(samples, case["player"], case.get("team", ""), "IN", evidence, now)


def run(cases: list[dict[str, Any]] | None = None) -> EvalResult:
    cases = cases or load_eval_set()
    now = datetime.now(timezone.utc)

    tp = fp = fn = 0
    correct = escalated = caught = leaked = over = clean = 0
    per_case: list[dict[str, Any]] = []

    for case in cases:
        truth = case["ground_truth"]
        v = _verdict_for(case, now)

        is_correct = v.status == truth
        correct += is_correct
        escalated += v.escalate

        # Blocking-detection confusion matrix, judged on the surfaced verdict.
        pred_block = v.status in BLOCKING
        true_block = truth in BLOCKING
        tp += pred_block and true_block
        fp += pred_block and not true_block
        fn += (not pred_block) and true_block

        if not is_correct and v.escalate:
            outcome, caught = "caught", caught + 1
        elif not is_correct and not v.escalate:
            outcome, leaked = "leaked", leaked + 1
        elif is_correct and v.escalate:
            outcome, over = "over_escalated", over + 1
        else:
            outcome, clean = "clean_pass", clean + 1

        per_case.append({
            "player": case["player"],
            "note": case.get("note", ""),
            "ground_truth": truth,
            "predicted": v.status,
            "confidence": v.confidence,
            "agreement": v.agreement,
            "escalated": v.escalate,
            "reasons": v.escalation_reasons,
            "outcome": outcome,
        })

    n = len(cases)
    return EvalResult(
        n=n,
        accuracy=round(correct / n, 3) if n else 0.0,
        blocking_precision=round(tp / (tp + fp), 3) if (tp + fp) else 0.0,
        blocking_recall=round(tp / (tp + fn), 3) if (tp + fn) else 0.0,
        escalation_rate=round(escalated / n, 3) if n else 0.0,
        caught=caught,
        leaked=leaked,
        over_escalated=over,
        clean_pass=clean,
        per_case=per_case,
    )


def threshold_sweep(cases: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Sweep the confidence threshold to expose the leaked/over-escalated
    trade-off rather than asserting that the default is correct."""
    cases = cases or load_eval_set()
    original = POLICY.low_confidence
    rows: list[dict[str, Any]] = []
    try:
        for thr in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            POLICY.low_confidence = thr
            r = run(cases)
            rows.append({
                "confidence_threshold": thr,
                "leaked": r.leaked,
                "over_escalated": r.over_escalated,
                "escalation_rate": r.escalation_rate,
                "human_minutes": round(r.escalation_rate * len(cases) * 1.5, 1),
            })
    finally:
        POLICY.low_confidence = original
    return rows


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------

def cost_projection(players_checked: int, samples_per_player: int | None = None) -> dict[str, Any]:
    """Project the spend of running this at the stated team's real volume.

    Deliberately compares against the human baseline it replaces, because
    "$X per gameweek" is meaningless without knowing what X buys.
    """
    samples = samples_per_player or POLICY.self_consistency_samples

    searches = players_checked
    # Measured from live runs: roughly 1.1k input tokens per call (system
    # prompt plus five search snippets) and ~90 output tokens of JSON.
    in_tok = players_checked * samples * 1100
    out_tok = players_checked * samples * 90

    per_run = (
        searches * COSTS["usd_per_search"]
        + in_tok / 1000 * COSTS["usd_per_1k_input_tokens"]
        + out_tok / 1000 * COSTS["usd_per_1k_output_tokens"]
    )

    gws = SCENARIO["gameweeks_per_season"]
    manual_hours = players_checked * SCENARIO["manual_minutes_per_player"] / 60.0
    manual_cost = manual_hours * COSTS["usd_per_reviewer_hour"]

    return {
        "players_checked": players_checked,
        "samples_per_player": samples,
        "searches": searches,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "usd_per_run": round(per_run, 3),
        "usd_per_season": round(per_run * gws, 2),
        "manual_hours_per_run": round(manual_hours, 1),
        "manual_usd_per_run": round(manual_cost, 2),
        "manual_usd_per_season": round(manual_cost * gws, 2),
        "saving_per_season": round((manual_cost - per_run) * gws, 2),
    }
