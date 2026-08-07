"""Gameweek Desk — editorial review console.

The human/AI boundary, stated plainly:

  The AI owns  : gathering data, ranking players, researching availability
                 across live sources, drafting the subscriber note.
  The human owns: every decision that reaches a subscriber.

Nothing leaves this system without a named editor pressing publish. That is
not a limitation of the prototype, it is the design. The failure this guards
against is not the model being wrong occasionally — it will be — but a wrong
claim reaching 40,000 people with nobody in the loop who could have caught it.
"""

from __future__ import annotations

import streamlit as st

from core import store
from core.config import POLICY, SCENARIO
from core.data import load
from core.evals import cost_projection
from core.model import score_players
from core.optimizer import demo_squad
from core.pipeline import run_gameweek

st.set_page_config(page_title="Gameweek Desk", page_icon="⬛", layout="wide")

STATUS_STYLE = {
    "AVAILABLE": ("🟢", "Available"),
    "DOUBTFUL": ("🟡", "Doubtful"),
    "ROTATION_RISK": ("🟡", "Rotation risk"),
    "INJURED": ("🔴", "Injured"),
    "SUSPENDED": ("🔴", "Suspended"),
    "UNKNOWN": ("⚪", "Unknown"),
}


# --------------------------------------------------------------------------
# Cached data loading
# --------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_universe(source: str, horizon: int):
    res = load(source, horizon=horizon)
    scored, report = score_players(res.players, res.prior_weight, horizon)
    return res, scored, report


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Gameweek Desk")
    st.caption(
        f"Editorial console for a {SCENARIO['team_size']}-person FPL advice team "
        f"serving {SCENARIO['subscribers']:,} subscribers."
    )
    st.divider()

    source = st.radio(
        "Data source",
        ["snapshot", "live"],
        format_func=lambda s: "Recorded snapshot" if s == "snapshot" else "Live FPL API",
        help=(
            "The snapshot is a fixed pre-season dataset so the demo is reproducible. "
            "Live hits the real FPL API and falls back to the snapshot if it is "
            "unreachable, rather than failing."
        ),
    )
    horizon = st.slider("Fixture horizon (gameweeks)", 1, 8, POLICY.fixture_horizon)
    free_transfers = st.slider("Free transfers", 1, 5, 2)
    max_transfers = st.slider("Max transfers to consider", 1, 3, 2)

    scope = st.radio(
        "Verification scope",
        ["bundle", "squad"],
        format_func=lambda s: "Transfer bundle only" if s == "bundle" else "Full squad",
        index=1,
        help=(
            "Bundle-only checks the players moving in and out: a handful of searches, "
            "fast and cheap, but blind to a held player who has just been ruled out. "
            "Full squad checks all fifteen for roughly four times the cost. This is "
            "the cost/confidence dial and it is deliberately an operator decision."
        ),
    )

    reviewer = st.text_input("Reviewer", value="Amaan (editor)")

    st.divider()
    st.caption("Escalation policy")
    POLICY.low_confidence = st.slider(
        "Minimum confidence", 0.0, 1.0, POLICY.low_confidence, 0.05,
        help="Below this, a consequential verdict goes to a human instead of being acted on.",
    )
    POLICY.max_evidence_age_days = st.slider(
        "Evidence freshness (days)", 1, 60, POLICY.max_evidence_age_days,
        help="A blocking flag cannot rest on an article older than this.",
    )


res, universe, report = get_universe(source, horizon)
squad = demo_squad(universe)
bank = round(100.0 - float(squad["price"].sum()), 1)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Gameweek Desk")
st.caption(
    "AI drafts the recommendation and researches availability. "
    "An editor decides what reaches subscribers."
)

for w in res.warnings:
    st.warning(w, icon="⚠️")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Squad value", f"£{squad['price'].sum():.1f}m")
c2.metric("In the bank", f"£{bank:.1f}m")
c3.metric("Model CV R²", f"{report.cv_r2:.2f}", help="Out-of-fold, 5-fold cross-validation.")
c4.metric("Model MAE", f"{report.cv_mae:.2f}", help="Mean absolute error, points per 90, out of fold.")
c5.metric(
    "Prior weight", f"{res.prior_weight:.0%}",
    help="How much of the forecast comes from prior-season rates rather than "
         "current-season form. 100% means the season has not started.",
)

if report.notes:
    with st.expander("Model caveats", expanded=False):
        for n in report.notes:
            st.markdown(f"- {n}")

st.divider()


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

left, right = st.columns([1, 3])
with left:
    go = st.button("Run gameweek analysis", type="primary", use_container_width=True)
with right:
    st.caption(
        "Ranks the player pool, proposes the best legal transfer bundle, researches "
        "the availability of everyone affected, and revises the recommendation if "
        "anything is confirmed unavailable."
    )

if go:
    with st.spinner("Scoring pool, proposing transfers, verifying availability..."):
        st.session_state.run = run_gameweek(
            squad, universe, bank,
            free_transfers=free_transfers,
            max_transfers=max_transfers,
            scope=scope,
        )
    st.session_state.decisions = {}

run = st.session_state.get("run")

if run is None:
    st.info("Run the analysis to populate the review queue.", icon="▶️")
    st.subheader("Current squad")
    view = squad[["web_name", "team_name", "position", "price", "xp_horizon", "start_prob"]].copy()
    # ProgressColumn formats the raw value, so express the share as 0-100
    # rather than 0-1 or every row renders as "1%".
    view["start_prob"] = (view["start_prob"] * 100).round(0)
    st.dataframe(
        view.rename(columns={
            "web_name": "Player", "team_name": "Club", "position": "Pos",
            "price": "£m", "xp_horizon": f"xP ({horizon} GW)", "start_prob": "Expected minutes",
        }).sort_values("Pos"),
        use_container_width=True, hide_index=True,
        column_config={
            "Expected minutes": st.column_config.ProgressColumn(
                "Expected minutes", min_value=0, max_value=100, format="%.0f%%"),
            f"xP ({horizon} GW)": st.column_config.NumberColumn(format="%.1f"),
            "£m": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.stop()


# --------------------------------------------------------------------------
# Pipeline trace — how the AI got here
# --------------------------------------------------------------------------

st.subheader("What the system did")

t1, t2, t3 = st.columns(3)
with t1:
    st.markdown("**1. Model proposed**")
    if run.initial:
        for t in run.initial.transfers:
            st.markdown(f"- {t.out_name} → **{t.in_name}**  \n  `+{t.xp_gain:.1f} xP`")
    else:
        st.markdown("- No transfer cleared the threshold")
with t2:
    st.markdown("**2. Agent verified**")
    st.markdown(f"- {len(run.verdicts)} players researched")
    st.markdown(f"- {run.cost.searches} searches, {run.cost.llm_calls} model calls")
    if run.blocked_players:
        st.markdown(f"- 🔴 Blocked: **{', '.join(run.blocked_players)}**")
    if run.escalated_players:
        st.markdown(f"- ⚠️ Escalated: **{', '.join(run.escalated_players)}**")
with t3:
    st.markdown("**3. System revised**")
    if run.revised and run.final:
        for t in run.final.transfers:
            st.markdown(f"- {t.out_name} → **{t.in_name}**  \n  `+{t.xp_gain:.1f} xP`")
    elif run.final:
        st.markdown("- No revision needed")
    else:
        st.markdown("- Recommending **hold**")

for n in run.notes:
    st.info(n, icon="ℹ️")

st.divider()


# --------------------------------------------------------------------------
# Escalations — the human's actual queue
# --------------------------------------------------------------------------

escalated = [v for v in run.verdicts if v.escalate]
if escalated:
    st.subheader(f"Needs your judgement ({len(escalated)})")
    st.caption(
        "The agent reached these but would not stand behind them. Each one names "
        "why it stopped rather than presenting a confident answer it cannot support."
    )
    for v in escalated:
        icon, label = STATUS_STYLE.get(v.status, ("⚪", v.status))
        with st.container(border=True):
            a, b = st.columns([3, 1])
            with a:
                st.markdown(f"**{v.player}** · {v.team} · `{v.side}`")
                st.markdown(f"{icon} Agent's reading: **{label}**")
                for r in v.escalation_reasons:
                    st.markdown(f"- ⚠️ {r}")
            with b:
                st.metric("Confidence", f"{v.confidence:.0%}")
                st.metric("Agreement", f"{v.agreement:.0%}",
                          help="Share of independent samples that produced the same verdict.")

            if len(set(s.status for s in v.samples)) > 1:
                with st.expander("Why the samples disagreed"):
                    st.caption(
                        "The same question asked several times at different temperatures. "
                        "When these do not converge, the honest output is 'I do not know', "
                        "not the first answer."
                    )
                    for i, s in enumerate(v.samples, 1):
                        st.markdown(
                            f"**Sample {i}** — `{s.status}` at {s.confidence:.0%}  \n"
                            f"{s.reasoning}"
                            + (f"  \n> {s.quote}" if s.quote else "")
                            + (f"  \n[source]({s.source_url})" if s.source_url else
                               "  \n_no source cited_")
                        )

            if v.evidence:
                with st.expander(f"Evidence retrieved ({len(v.evidence)})"):
                    for e in v.evidence:
                        st.markdown(
                            f"**{e.title}**  \n{e.snippet}  \n"
                            f"`{e.published or 'date unknown'}` · [{e.url}]({e.url})"
                        )
    st.divider()


# --------------------------------------------------------------------------
# Review queue — draft notes awaiting publish
# --------------------------------------------------------------------------

st.subheader("Draft recommendation")

if run.final is None:
    st.warning(
        "The system is recommending **hold** this week. Nothing to publish.",
        icon="⏸️",
    )
else:
    verdict_by_player = {v.player.split(" (")[0]: v for v in run.verdicts}

    for idx, t in enumerate(run.final.transfers):
        in_base = t.in_name.split(" (")[0]
        out_base = t.out_name.split(" (")[0]
        vin = verdict_by_player.get(in_base)
        vout = verdict_by_player.get(out_base)
        key = f"{t.out_id}-{t.in_id}"
        decided = st.session_state.get("decisions", {}).get(key)

        with st.container(border=True):
            head, badge = st.columns([4, 1])
            with head:
                st.markdown(f"### {t.out_name} → {t.in_name}")
                st.caption(f"{t.position} · sell £{t.out_sell:.1f}m · buy £{t.in_cost:.1f}m")
            with badge:
                if decided:
                    st.success(decided["action"].replace("_", " ").title())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("xP gain", f"+{t.xp_gain:.1f}")
            m2.metric("80% band", f"{t.in_xp_low:.0f}–{t.in_xp_high:.0f}",
                      help="The model's own uncertainty. A gain smaller than this band "
                           "is not a meaningful edge.")
            m3.metric("Expected minutes", f"{t.in_start_prob:.0%}")
            if vin:
                icon, label = STATUS_STYLE.get(vin.status, ("⚪", vin.status))
                m4.metric("Availability", f"{icon} {label}")

            # Availability evidence for both sides
            for v, side_label in ((vin, "Incoming"), (vout, "Outgoing")):
                if not v:
                    continue
                icon, label = STATUS_STYLE.get(v.status, ("⚪", v.status))
                with st.expander(f"{side_label}: {v.player} — {icon} {label} "
                                 f"({v.confidence:.0%} confidence)"):
                    st.markdown(v.reasoning or "_No reasoning recorded._")
                    if v.quote:
                        st.markdown(f"> {v.quote}")
                    if v.source_url:
                        st.markdown(f"Source: [{v.source_url}]({v.source_url}) · `{v.as_of or 'date unknown'}`")
                    else:
                        st.caption("No source cited.")

            draft = (
                f"**Transfer of the week: {t.out_name} → {t.in_name}**\n\n"
                f"We're moving {out_base} on for {in_base} this week. "
                f"Our model has {in_base} at +{t.xp_gain:.1f} expected points over the "
                f"next {horizon} gameweeks, with an expected minutes share of "
                f"{t.in_start_prob:.0%}.\n\n"
                + (f"Availability check: {in_base} is "
                   f"{STATUS_STYLE.get(vin.status, ('', vin.status))[1].lower()}"
                   + (f" ({vin.quote})" if vin and vin.quote else "") + ".\n\n" if vin else "")
                + f"This leaves £{run.final.bank_after:.1f}m in the bank"
                + (f" and costs a -{run.final.hit_points:.0f} hit."
                   if run.final.hit_points else ".")
            )

            st.markdown("**Draft note to subscribers**")
            edited = st.text_area(
                "Draft", value=draft, height=190, key=f"draft-{key}",
                label_visibility="collapsed",
            )
            changed = edited.strip() != draft.strip()
            if changed:
                st.caption("✏️ Edited from the AI draft — this will be recorded.")

            b1, b2, b3, _ = st.columns([1, 1, 1, 3])
            if b1.button("Publish", key=f"pub-{key}", type="primary",
                         disabled=bool(decided), use_container_width=True):
                entry = store.record(
                    action="EDITED_AND_PUBLISHED" if changed else "PUBLISHED",
                    reviewer=reviewer,
                    summary=f"{t.out_name} → {t.in_name}",
                    reason="",
                    ai_recommendation=draft,
                    human_changed_it=changed,
                    payload={
                        "out": t.out_name, "in": t.in_name, "xp_gain": t.xp_gain,
                        "published_text": edited,
                        "availability": vin.to_dict() if vin else None,
                        "bank_after": run.final.bank_after,
                        "hit": run.final.hit_points,
                    },
                )
                st.session_state.setdefault("decisions", {})[key] = entry
                st.rerun()

            if b2.button("Reject", key=f"rej-{key}", disabled=bool(decided),
                         use_container_width=True):
                st.session_state.setdefault("pending_reject", set()).add(key)

            if b3.button("Escalate", key=f"esc-{key}", disabled=bool(decided),
                         use_container_width=True):
                entry = store.record(
                    action="ESCALATED", reviewer=reviewer,
                    summary=f"{t.out_name} → {t.in_name}",
                    reason="Referred to a second editor.",
                    ai_recommendation=draft,
                    payload={"out": t.out_name, "in": t.in_name},
                )
                st.session_state.setdefault("decisions", {})[key] = entry
                st.rerun()

            if key in st.session_state.get("pending_reject", set()) and not decided:
                reason = st.text_input(
                    "Why are you rejecting this? (recorded as a training signal)",
                    key=f"rr-{key}",
                )
                if st.button("Confirm rejection", key=f"cr-{key}"):
                    entry = store.record(
                        action="REJECTED", reviewer=reviewer,
                        summary=f"{t.out_name} → {t.in_name}",
                        reason=reason or "No reason given.",
                        ai_recommendation=draft, human_changed_it=True,
                        payload={"out": t.out_name, "in": t.in_name},
                    )
                    st.session_state.setdefault("decisions", {})[key] = entry
                    st.session_state["pending_reject"].discard(key)
                    st.rerun()

st.divider()


# --------------------------------------------------------------------------
# Run economics
# --------------------------------------------------------------------------

st.subheader("This run")
e1, e2, e3, e4 = st.columns(4)
e1.metric("Searches", run.cost.searches)
e2.metric("Model calls", run.cost.llm_calls)
e3.metric("Cost", f"${run.cost.usd:.3f}")

proj = cost_projection(SCENARIO["players_tracked"])
e4.metric(
    "At full scale", f"${proj['usd_per_run']:.2f}/GW",
    help=f"Checking all {SCENARIO['players_tracked']} tracked players. "
         f"The manual equivalent is {proj['manual_hours_per_run']} reviewer-hours, "
         f"about ${proj['manual_usd_per_run']:,.0f}.",
)

if run.cost.searches == 0:
    st.caption(
        "Running on recorded agent verdicts — no API keys configured, so no live "
        "spend. The orchestration, aggregation and escalation logic are the same "
        "code paths used against live APIs; only the two external calls are replayed."
    )
