"""Every decision, who made it, and whether they changed what the AI proposed."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core import store

st.set_page_config(page_title="Audit log · Gameweek Desk", page_icon="⬛", layout="wide")
st.title("Audit log")
st.caption("Append-only. Written before anything is published, not after.")

s = store.stats()

if not s.get("total"):
    st.info(
        "Nothing logged yet. Publish, edit or reject a recommendation on the main "
        "page and it will appear here.",
        icon="📋",
    )
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Entries", s["total"])
c2.metric("Decisions", s["decisions"])
c3.metric("Override rate", f"{s['override_rate']:.0%}",
          help="Share of decisions where the editor changed or rejected what the AI "
               "proposed. Near zero means the review step is theatre and the human is "
               "rubber-stamping. Very high means the AI is not good enough to be drafting. "
               "The useful range is in between.")
c4.metric("Reviewers", len(s["reviewers"]))

if s["decisions"] >= 3:
    r = s["override_rate"]
    if r < 0.05:
        st.warning(
            "Override rate is near zero. Either the AI is unusually good this week, or "
            "the editor has stopped reading. Worth checking which — a review step nobody "
            "engages with provides no safety at all.",
            icon="⚠️",
        )
    elif r > 0.6:
        st.warning(
            "Override rate above 60%. The AI is producing drafts that mostly get rewritten, "
            "which means it is adding work rather than removing it. Worth retuning before "
            "widening its remit.",
            icon="⚠️",
        )

st.divider()

rows = store.read_all()
flat = [{
    "When": r["ts"],
    "Action": r["action"].replace("_", " ").title(),
    "Reviewer": r["reviewer"],
    "Recommendation": r["summary"],
    "Human changed it": "Yes" if r.get("human_changed_it") else "No",
    "Reason": r.get("reason", ""),
    "Fingerprint": r.get("fingerprint", ""),
} for r in rows]

st.dataframe(pd.DataFrame(flat), width="stretch", hide_index=True)

st.divider()
st.subheader("Detail")
for r in rows[:25]:
    with st.expander(f"{r['ts']} · {r['action'].replace('_', ' ').title()} · {r['summary']}"):
        st.markdown(f"**Reviewer:** {r['reviewer']}")
        if r.get("reason"):
            st.markdown(f"**Reason given:** {r['reason']}")
        st.markdown(f"**Content hash:** `{r.get('fingerprint', '')}`")
        if r.get("human_changed_it"):
            st.markdown("**The editor changed the AI's draft before publishing.**")
        if r.get("ai_recommendation"):
            st.markdown("**What the AI drafted**")
            st.code(r["ai_recommendation"], language=None)
        published = (r.get("payload") or {}).get("published_text")
        if published and published.strip() != (r.get("ai_recommendation") or "").strip():
            st.markdown("**What actually went out**")
            st.code(published, language=None)
        st.markdown("**Payload**")
        st.json(r.get("payload", {}), expanded=False)

st.divider()
with st.expander("Reset log (demo only)"):
    st.caption("Exists so the demo can be replayed cleanly. There is no equivalent in production.")
    if st.button("Clear the audit log", type="secondary"):
        store.clear()
        st.rerun()
