"""Assumptions, trade-offs and known failure modes, stated rather than implied."""

from __future__ import annotations

import streamlit as st

from core.config import POLICY, SCENARIO

st.set_page_config(page_title="Assumptions · Gameweek Desk", page_icon="⬛", layout="wide")
st.title("Assumptions and trade-offs")

st.markdown(f"""
## The scenario

A **{SCENARIO['team_size']}-person FPL advice service** with **{SCENARIO['subscribers']:,}
subscribers**. Every gameweek the team researches injury, suspension and rotation news
across ~{SCENARIO['players_tracked']} players, decides what to recommend, and publishes
before the deadline.

Done by hand that is around **{SCENARIO['players_tracked'] * SCENARIO['manual_minutes_per_player'] / 60:.0f}
reviewer-hours per gameweek** — most of one person's week, every week, on work that is
almost entirely repetitive lookup. It also caps the product: you cannot cover more players,
or more subscriber segments, without hiring.

This is a stand-in for a shape of problem, not a claim about any particular company. The
same structure — a deterministic ranking, external research that needs judgement, and a
publish decision with real consequences — describes a trust & safety review queue, a
refunds desk, or a grant-eligibility screen. Only the domain vocabulary changes.

## What the AI owns, and what it does not

| | |
|---|---|
| **AI owns** | Ingesting and cleaning the data. Ranking the player pool. Searching live sources for availability. Reading unstructured news and forming a verdict. Drafting the subscriber note. Removing confirmed-unavailable players and re-running the optimiser. |
| **Human owns** | Every decision that reaches a subscriber. Anything the agent flagged as uncertain. The confidence and freshness thresholds. Whether a −4 hit is worth taking. |
| **Neither** | The system never edits the user's actual FPL team, never sends anything, and has no write access to any external system. Its output is a draft and a queue. |

The one thing the AI does act on unilaterally is removing a player from consideration when
availability is **confirmed** — unanimous across samples, cited, and fresh. That is a
conservative action: the worst case is a missed opportunity, not a false claim published.
Every other flag routes to a human.

## Trade-offs, and which way they were called

**Escalate more, or leak less.** Every threshold that reduces wrong answers reaching an
editor increases the number of correct answers they waste time confirming. Currently set to
{POLICY.low_confidence:.2f} confidence and {POLICY.max_evidence_age_days} days freshness,
which on the labelled set leaks 2 of 16 and over-escalates 2. Both are operator controls,
because the right point differs on a Tuesday and an hour before deadline.

**Sample the model, not the search.** Self-consistency needs several opinions. Running the
search {POLICY.self_consistency_samples} times would triple the dominant cost for almost no
benefit, since the retrieved evidence is the same. So the search runs once and the model is
sampled {POLICY.self_consistency_samples} times at varying temperature against the same
evidence. This measures instability in the *judgement*, which is where it was observed.

**A simple model I can validate, over a complex one I cannot.** The earlier version used
gradient boosting fit on season-to-date points, predicted on its own training rows. Replaced
with ridge regression, cross-validated, with a residual-derived prediction interval. Almost
certainly less accurate at its ceiling. But the number it produces is a forecast rather than
a restatement, and it carries an error bar, which is what a human needs to weigh it.

**Recorded snapshot as the default.** The demo runs on a fixed pre-season dataset so it is
reproducible and cannot break on a third-party outage mid-review. Live mode hits the real
FPL API and degrades to the snapshot rather than erroring.

**Verification scope as a dial.** Bundle-only is cheap and blind to held players. Full-squad
costs about four times as much and is the only setting that catches a problem with someone
the optimiser had no opinion about. Left to the operator rather than fixed.

## Known failure modes

- **Comprehension errors pass every guardrail.** Two cases in the eval set are unanimous,
  cited, fresh and wrong. Disagreement, citation and freshness checks cannot catch a
  misreading. Roughly one in eight on the current set. This is the residual an editor absorbs.
- **Search coverage is uneven.** Well-covered players get good evidence; a squad player at a
  promoted club may return nothing usable. The system returns UNKNOWN and escalates rather
  than guessing, but that means more human load exactly where the human also knows least.
- **Pre-season forecasts are priors, not predictions.** With no minutes played, the model is
  leaning entirely on last season's rates. The prior weight is shown in the header so this is
  visible rather than buried.
- **Synthetic snapshot inflates the model metrics.** Priors were generated from price, so
  cross-validated R² is optimistic. On live prior-season data, expect materially lower.
- **Club and position data may be stale.** Snapshot rosters may not reflect recent transfers.
- **No provider redundancy.** One search provider and one model provider. Either going down
  degrades the system to "everything escalates", which is safe but not useful.

## What I would build next, in order

1. **Close the feedback loop on overrides.** Editor rejections with reasons are labelled data
   the system is currently throwing away. That is the highest-value missing piece: it turns
   every week of operation into eval set growth.
2. **Cache verdicts across players and runs.** Team news is per-club, not per-player. One
   search per club rather than per player would cut the dominant cost by roughly an order of
   magnitude.
3. **A second source before any blocking call.** Cross-check confirmed flags against a
   structured injury feed. Precision on blocking calls is the weakest measured number and a
   second source is the direct fix.
4. **Batch overnight, review in the morning.** Scheduled runs writing to the queue, so
   latency stops mattering and a cheaper, slower model becomes viable.
5. **Per-reviewer calibration.** Track which editors override which flag types, and route
   accordingly.
""")
