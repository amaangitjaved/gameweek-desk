"""How well does the agent actually work, and what does it cost to find out."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import COSTS, POLICY, SCENARIO
from core.evals import cost_projection, load_eval_set, run, threshold_sweep

st.set_page_config(page_title="Evaluation · Gameweek Desk", page_icon="⬛", layout="wide")
st.title("Evaluation")
st.caption(
    "Accuracy is the easy question. The one that matters is what reaches a human "
    "as a confident wrong answer."
)

cases = load_eval_set()
result = run(cases)

st.subheader("Outcomes on the labelled set")
st.caption(
    f"{result.n} hand-labelled cases covering the failure modes actually observed: "
    "unstable verdicts, hallucinated suspensions, stale evidence, parse failures, "
    "and two cases designed to slip through."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clean pass", result.clean_pass, help="Correct and not escalated. The AI handled it alone.")
c2.metric("Caught", result.caught, help="Wrong, but escalated. The safety net worked.")
c3.metric("Leaked", result.leaked,
          help="Wrong AND not escalated. The only outcome that can reach a subscriber. "
               "This is the number that matters.")
c4.metric("Over-escalated", result.over_escalated,
          help="Right, but escalated anyway. Pure cost in human attention.")

d1, d2, d3 = st.columns(3)
d1.metric("Accuracy", f"{result.accuracy:.0%}")
d2.metric("Blocking precision", f"{result.blocking_precision:.0%}",
          help="Of the players the agent flagged as injured or suspended, how many really were.")
d3.metric("Blocking recall", f"{result.blocking_recall:.0%}",
          help="Of the players who really were unavailable, how many the agent caught.")

st.info(
    f"Precision of {result.blocking_precision:.0%} on blocking calls is the honest headline: "
    "roughly half the time the agent says 'injured or suspended', it is over-reading its "
    "sources. That is exactly why a confirmed block is the only thing the system acts on "
    "unilaterally, and why everything shaky is routed to a human instead.",
    icon="🎯",
)

st.divider()

st.subheader("Case by case")
df = pd.DataFrame(result.per_case)
df["reasons"] = df["reasons"].apply(lambda rs: " · ".join(rs) if rs else "")
st.dataframe(
    df.rename(columns={
        "player": "Player", "ground_truth": "Truth", "predicted": "Agent said",
        "confidence": "Conf.", "agreement": "Agreement", "escalated": "Escalated",
        "outcome": "Outcome", "reasons": "Why escalated", "note": "Case",
    })[["Player", "Case", "Truth", "Agent said", "Conf.", "Agreement",
        "Escalated", "Outcome", "Why escalated"]],
    use_container_width=True, hide_index=True,
    column_config={
        "Conf.": st.column_config.NumberColumn(format="%.2f"),
        "Agreement": st.column_config.NumberColumn(format="%.2f"),
    },
)

with st.expander("The two leaks, and why no threshold catches them"):
    st.markdown(
        """
**Palmer.** A training report mentions he is managing a minor knock. The agent reads
"took part in training" and returns AVAILABLE, unanimously, at 86% confidence, citing a
fresh club source. Every guardrail passes: it agrees with itself, it cites something, the
source is a day old. The verdict is simply wrong about a nuance in the text.

**Watkins.** Left out of a friendly amid transfer speculation. The agent reads squad
omission as rotation risk. Again unanimous, confident, well sourced.

Neither is caught by disagreement, citation or freshness checks, because none of those
things are missing. They are comprehension errors, not process errors. Tightening the
confidence threshold does not help — both sit above any threshold that leaves the system
usable.

The honest position is that this residual rate exists, it is roughly one in eight on this
set, and it is the reason an editor reads the note before it goes out. The system is built
to make that reading fast, not to make it unnecessary.
        """
    )

st.divider()

st.subheader("The escalation trade-off")
st.caption(
    "Every threshold that reduces leakage increases the human review burden. "
    "There is no setting that minimises both. The team picks a point on this curve."
)

sweep = pd.DataFrame(threshold_sweep(cases))
st.dataframe(
    sweep.rename(columns={
        "confidence_threshold": "Confidence threshold",
        "leaked": "Leaked", "over_escalated": "Over-escalated",
        "escalation_rate": "Escalation rate", "human_minutes": "Human minutes",
    }),
    use_container_width=True, hide_index=True,
)
st.line_chart(sweep.set_index("confidence_threshold")[["leaked", "over_escalated"]])
st.caption(
    f"Currently set to {POLICY.low_confidence:.2f}. Moving to 0.90 removes one leak and "
    "adds three unnecessary escalations — worth it an hour before the deadline, wasteful "
    "on a Tuesday. This is why the threshold is an operator control, not a constant."
)

st.divider()

st.subheader("Cost at real volume")
players = st.slider("Players checked per gameweek", 15, 600, SCENARIO["players_tracked"], 15)
samples = st.slider("Self-consistency samples per player", 1, 5, POLICY.self_consistency_samples)
proj = cost_projection(players, samples)

k1, k2, k3 = st.columns(3)
k1.metric("Per gameweek", f"${proj['usd_per_run']:.2f}")
k2.metric("Per season", f"${proj['usd_per_season']:,.0f}")
k3.metric("Manual equivalent", f"${proj['manual_usd_per_season']:,.0f}/season",
          help=f"{proj['manual_hours_per_run']} reviewer-hours per gameweek at "
               f"${COSTS['usd_per_reviewer_hour']}/hour.")

st.markdown(
    f"""
Checking **{players}** players at **{samples}** samples each costs
**${proj['usd_per_run']:.2f}** per gameweek — **{proj['searches']} searches** and
**{proj['input_tokens'] + proj['output_tokens']:,} tokens**.

Search dominates: at {COSTS['usd_per_search']:.3f} per query it is roughly
{proj['searches'] * COSTS['usd_per_search'] / max(proj['usd_per_run'], 1e-9):.0%} of the
bill, which is why self-consistency samples the *model* several times but the *search*
only once. Sampling the model is nearly free; sampling the search is not.

The manual baseline is **{proj['manual_hours_per_run']} reviewer-hours** per gameweek.
For a {SCENARIO['team_size']}-person team that is most of someone's week, every week.
    """
)
