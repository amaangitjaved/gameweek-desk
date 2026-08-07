# Gameweek Desk

An editorial review console for a small FPL advice team. A statistical model ranks the
player pool and proposes transfers; an LLM agent researches whether those players are
actually available to play; an editor decides what reaches subscribers.

**Nothing is published without a named human pressing publish.**

---

## The problem

A 6-person advice service with 40,000 subscribers has to research injury, suspension and
rotation news across ~600 players before every gameweek deadline. By hand that is about
25 reviewer-hours a week of repetitive lookup, and it caps the product — you cannot cover
more players or more segments without hiring.

## The system

```
FPL data ──▶ xP model ──▶ transfer optimiser ──▶ availability agent ──▶ review queue ──▶ publish
             (ridge,       (shared budget,        (search + LLM,         (editor
              CV'd)         legal squads)          self-consistency)      decides)
                                  ▲                        │
                                  └──── confirmed block ───┘
                                        (re-optimise)
```

The agent does not merely annotate the recommendation. A **confirmed** availability
problem — unanimous across samples, cited, and fresh — removes that player from the
candidate pool and the optimiser reruns. Anything less certain goes to a human with a
stated reason instead.

### What the AI owns vs the human

| AI | Human |
|---|---|
| Ingest, clean, rank | Every decision reaching a subscriber |
| Search live sources for availability | Anything the agent flagged as uncertain |
| Read unstructured news, form a verdict | Confidence and freshness thresholds |
| Draft the subscriber note | Whether a −4 hit is worth taking |
| Remove confirmed-unavailable players and re-run | |

The system has no write access to anything. Its output is a draft and a queue.

---

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Runs with no API keys, on recorded agent verdicts. The orchestration, aggregation and
escalation logic are the same code paths used against live APIs; only the two external
calls are replayed.

For live agent behaviour, add `.streamlit/secrets.toml` (see `secrets.toml.example`):

```toml
GROQ_API_KEY = "gsk_..."
SERPAPI_KEY  = "..."
```

Verify the invariants:

```bash
python test_smoke.py
```

Rebuild the demo snapshot:

```bash
python build_snapshot.py
```

---

## Deploying to Streamlit Community Cloud

1. Push this directory to a GitHub repo.
2. At [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo,
   branch, and `app.py` as the entrypoint.
3. Optionally add `GROQ_API_KEY` and `SERPAPI_KEY` under **Advanced settings → Secrets**.
   The app works without them.
4. Deploy. First build takes 2–3 minutes.

---

## Layout

```
app.py                     Review console (main page)
pages/1_Evaluation.py      Accuracy, leak rate, threshold sweep, cost model
pages/2_Audit_Log.py       Append-only decision log, override rate
pages/3_Assumptions.py     Assumptions, trade-offs, failure modes, roadmap
core/config.py             Policy thresholds and cost constants, all in one place
core/data.py               FPL ingest, cold-start handling, player resolution
core/model.py              Ridge xP model with cross-validation and error bars
core/optimizer.py          Constrained transfer search over a shared budget
core/verify.py             Availability agent: search, sample, aggregate, escalate
core/pipeline.py           End-to-end run including the re-optimise feedback loop
core/store.py              Append-only audit log
core/evals.py              Labelled eval harness and cost projection
test_smoke.py              80+ invariant checks
```

---

## What changed from the first prototype, and why

| Before | Problem | Now |
|---|---|---|
| LightGBM fit on `total_points`, predicted on its own training rows | In-sample reconstruction of points already scored, presented as a forecast. `ict_index` leaked the label. | Ridge regression, 5-fold CV, features exclude points-derived quantities, residual-based prediction interval |
| Trained on current-season totals | In August every total is 0 → constant target → constant prediction → **empty recommendations, silently** | Season maturity measured; prior-season per-90 rates weighted in, blend shown in the UI |
| Each transfer given the full bank independently | Two transfers double-spend; `bank_balance_after` read off the last move and was simply wrong | Bundle search over one shared budget, validated for legality |
| No squad constraints | Would propose a 4th Arsenal defender | 3-per-club, like-for-like positions, correct FPL sell-price rule |
| No hit accounting | "+1.2 xP" could be a net loss of 2.8 | −4 hits costed, and a hit must clear a margin to outrank a free transfer |
| Agent output free-text prose | Same player, same input, "suspended" then "injured/back" 60s apart | Structured JSON, sampled 3× at varying temperature, **disagreement escalates** |
| Verdicts with no source or date | Match reports read as current squad status | Citation and freshness required; uncited blocking claims are inadmissible |
| Only `transfers_in` checked | An injury to the player you are selling was invisible | Both sides checked; full-squad scope available |
| Warning appended to unchanged advice | Human had to notice the contradiction | Confirmed blocks re-run the optimiser |
| Workflow ended with no output node | Nothing was ever published, approved or logged | Publish/edit/reject gate with an append-only audit log |
| `localhost:5678` + `127.0.0.1:8000` | Not shareable | Single deployable app |

---

## Known limits

Stated in full on the Assumptions page in the app. The short version: comprehension errors
pass every guardrail (~1 in 8 on the labelled set), search coverage is uneven, pre-season
forecasts are priors rather than predictions, and the snapshot's synthetic priors inflate
the model metrics.
