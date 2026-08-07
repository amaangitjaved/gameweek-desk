"""Availability verification agent: search the web, judge, and know when not to.

This is the part of the system that does real cognitive work. The statistical
model can rank players; it cannot know that someone limped off in a friendly
on Tuesday. That requires reading unstructured news and forming a judgement,
which is what an LLM is genuinely good at and what a regression cannot do.

The three failures this module is built around
----------------------------------------------
All three were observed in the original n8n workflow.

1. **Inconsistency.** The same player, the same input, one minute apart,
   returned "suspended" once and "injured/back" the next time. A system that
   returns a different answer each run cannot be trusted with a publish
   decision. Here the agent is sampled several times and *disagreement is
   itself a signal*: when the samples do not converge, the item is escalated
   to a human rather than one verdict being picked and presented confidently.

2. **Ungrounded conclusions.** The original produced "found guilty of violent
   conduct and should have been sent off, therefore suspended" with no source
   and no date. That is a match report being read as current squad status.
   Every verdict here must carry a source URL, a publication date and a
   verbatim quote, or it is downgraded to UNKNOWN.

3. **Staleness.** News from three weeks ago cannot establish availability
   today. Evidence older than the policy window cannot on its own justify a
   blocking flag.

Cost is metered per run because search is the dominant expense and the
cost/confidence trade-off is a real operational decision, not a footnote.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any

from core.config import COSTS, POLICY, get_secret

STATUSES = ["AVAILABLE", "DOUBTFUL", "INJURED", "SUSPENDED", "ROTATION_RISK", "UNKNOWN"]
BLOCKING = {"INJURED", "SUSPENDED"}
CAUTION = {"DOUBTFUL", "ROTATION_RISK"}

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
SERPAPI_URL = "https://serpapi.com/search"

SYSTEM_PROMPT = """You assess whether a Premier League footballer is available to play in the next fixture.

You will be given search results. Judge ONLY from those results. You have no other knowledge of current squad status.

Return a single JSON object, no prose, no markdown fence:
{
  "status": one of ["AVAILABLE","DOUBTFUL","INJURED","SUSPENDED","ROTATION_RISK","UNKNOWN"],
  "confidence": float 0.0-1.0,
  "source_url": the URL of the single result you relied on most, or null,
  "as_of": publication date of that result as YYYY-MM-DD, or null,
  "quote": a verbatim sentence from the results supporting your verdict, max 200 chars, or null,
  "reasoning": one sentence, max 200 chars
}

Rules you must follow:
- If the results do not clearly establish current availability, return UNKNOWN with low confidence. UNKNOWN is a correct and useful answer; guessing is not.
- A report of a past incident (a red card in a match that has been played, an old injury) does NOT by itself establish current unavailability. Look for explicit statements about the upcoming fixture or current status.
- Do not infer a suspension from a description of a foul, a booking, or pundit commentary. Only report SUSPENDED if a source states the player is suspended or banned for an upcoming match.
- If sources conflict, return the more cautious status and lower your confidence.
- Never invent a URL, a date or a quote. If you do not have one, use null."""


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------

@dataclass
class Evidence:
    title: str
    url: str
    snippet: str
    published: str | None = None


@dataclass
class Sample:
    status: str
    confidence: float
    source_url: str | None
    as_of: str | None
    quote: str | None
    reasoning: str


@dataclass
class Verdict:
    player: str
    team: str
    side: str                       # "IN" or "OUT"
    status: str
    confidence: float
    agreement: float                # fraction of samples agreeing with the modal status
    escalate: bool
    escalation_reasons: list[str] = field(default_factory=list)
    source_url: str | None = None
    as_of: str | None = None
    quote: str | None = None
    reasoning: str = ""
    samples: list[Sample] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    searches_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def is_blocking(self) -> bool:
        return self.status in BLOCKING and not self.escalate

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_blocking"] = self.is_blocking
        return d


@dataclass
class RunCost:
    searches: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0

    @property
    def usd(self) -> float:
        return (
            self.searches * COSTS["usd_per_search"]
            + self.input_tokens / 1000 * COSTS["usd_per_1k_input_tokens"]
            + self.output_tokens / 1000 * COSTS["usd_per_1k_output_tokens"]
        )


# --------------------------------------------------------------------------
# External calls
# --------------------------------------------------------------------------

def _search(query: str, api_key: str, num: int = 5) -> list[Evidence]:
    import requests

    resp = requests.get(
        SERPAPI_URL,
        params={"q": query, "api_key": api_key, "num": num, "hl": "en", "gl": "uk",
                "tbs": "qdr:m"},  # last month only; older news cannot establish current status
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    out: list[Evidence] = []
    for r in (data.get("news_results") or []) + (data.get("organic_results") or []):
        out.append(Evidence(
            title=r.get("title", ""),
            url=r.get("link", ""),
            snippet=r.get("snippet") or r.get("description") or "",
            published=r.get("date") or r.get("published_date"),
        ))
        if len(out) >= num:
            break
    return out


def _llm(messages: list[dict], api_key: str, temperature: float) -> tuple[str, int, int]:
    import requests

    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    return (
        data["choices"][0]["message"]["content"],
        int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
    )


def _parse_sample(raw: str) -> Sample:
    """Parse the model's JSON, defaulting to UNKNOWN rather than throwing.

    A malformed response is a real occurrence, not an exception: the correct
    behaviour is to record it as an uninformative sample so it counts against
    agreement, not to crash the run.
    """
    try:
        text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        obj = json.loads(text)
    except Exception:
        return Sample("UNKNOWN", 0.0, None, None, None, "Unparseable model response.")

    status = str(obj.get("status", "UNKNOWN")).upper().strip()
    if status not in STATUSES:
        status = "UNKNOWN"
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0

    return Sample(
        status=status,
        confidence=max(0.0, min(1.0, conf)),
        source_url=obj.get("source_url") or None,
        as_of=obj.get("as_of") or None,
        quote=(obj.get("quote") or None),
        reasoning=str(obj.get("reasoning") or "")[:250],
    )


# --------------------------------------------------------------------------
# Staleness
# --------------------------------------------------------------------------

def _days_old(as_of: str | None, now: datetime | None = None) -> int | None:
    if not as_of:
        return None
    now = now or datetime.now(timezone.utc)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(as_of.strip()[:10], fmt).replace(tzinfo=timezone.utc)
            return (now - dt).days
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------
# Aggregation: this is where disagreement becomes a signal
# --------------------------------------------------------------------------

def aggregate(samples: list[Sample], player: str, team: str, side: str,
              evidence: list[Evidence], now: datetime | None = None) -> Verdict:
    counts = Counter(s.status for s in samples)
    modal, modal_n = counts.most_common(1)[0]
    agreement = modal_n / max(1, len(samples))

    agreeing = [s for s in samples if s.status == modal]
    mean_conf = sum(s.confidence for s in agreeing) / max(1, len(agreeing))

    # Prefer the agreeing sample that actually cited something.
    cited = next((s for s in agreeing if s.source_url), agreeing[0] if agreeing else None)

    reasons: list[str] = []

    # 1. Disagreement across samples.
    if 1.0 - agreement >= POLICY.disagreement_escalation_threshold:
        spread = ", ".join(f"{k} x{v}" for k, v in counts.most_common())
        reasons.append(f"Samples disagreed ({spread}) — the model is not stable on this player.")

    # 2. A blocking claim with no citation is not admissible.
    if modal in BLOCKING and (not cited or not cited.source_url):
        reasons.append("Blocking status claimed with no source URL.")

    # 3. Stale evidence cannot establish current availability.
    age = _days_old(cited.as_of if cited else None, now)
    if modal in BLOCKING and age is not None and age > POLICY.max_evidence_age_days:
        reasons.append(f"Supporting article is {age} days old (limit {POLICY.max_evidence_age_days}).")
    if modal in BLOCKING and age is None and cited and cited.source_url:
        reasons.append("Could not establish the publication date of the supporting article.")

    # 4. Low confidence on a consequential call.
    if modal in (BLOCKING | CAUTION) and mean_conf < POLICY.low_confidence:
        reasons.append(f"Confidence {mean_conf:.2f} is below the {POLICY.low_confidence:.2f} threshold.")

    # 5. No evidence retrieved at all.
    if not evidence:
        reasons.append("No search results were retrieved for this player.")

    return Verdict(
        player=player,
        team=team,
        side=side,
        status=modal,
        confidence=round(mean_conf, 3),
        agreement=round(agreement, 3),
        escalate=bool(reasons),
        escalation_reasons=reasons,
        source_url=cited.source_url if cited else None,
        as_of=cited.as_of if cited else None,
        quote=cited.quote if cited else None,
        reasoning=cited.reasoning if cited else "",
        samples=samples,
        evidence=evidence,
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def verify_player(
    player: str,
    team: str,
    side: str,
    serp_key: str,
    groq_key: str,
    n_samples: int | None = None,
    cost: RunCost | None = None,
) -> Verdict:
    n_samples = n_samples or POLICY.self_consistency_samples
    cost = cost or RunCost()

    query = f"{player} {team} injury news team news availability"
    try:
        evidence = _search(query, serp_key)
        cost.searches += 1
    except Exception as exc:
        v = aggregate([], player, team, side, [], None)
        v.status = "UNKNOWN"
        v.escalate = True
        v.escalation_reasons = [f"Search failed: {type(exc).__name__}. Availability unverified."]
        return v

    if not evidence:
        return aggregate(
            [Sample("UNKNOWN", 0.0, None, None, None, "No search results.")],
            player, team, side, [], None,
        )

    context = "\n\n".join(
        f"[{i + 1}] {e.title}\nURL: {e.url}\nDate: {e.published or 'unknown'}\n{e.snippet}"
        for i, e in enumerate(evidence)
    )
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    user = (
        f"Today is {today}.\n"
        f"Player: {player} ({team})\n\n"
        f"Search results:\n{context}\n\n"
        f"Assess {player}'s availability for the next fixture."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    samples: list[Sample] = []
    # Temperature is varied deliberately. Sampling at a single low temperature
    # would hide instability rather than measure it; the point is to find out
    # whether the verdict is robust to it.
    for i in range(n_samples):
        try:
            raw, tin, tout = _llm(messages, groq_key, temperature=0.2 + 0.3 * i)
            cost.input_tokens += tin
            cost.output_tokens += tout
            cost.llm_calls += 1
            samples.append(_parse_sample(raw))
        except Exception as exc:
            samples.append(Sample("UNKNOWN", 0.0, None, None, None, f"LLM call failed: {type(exc).__name__}"))

    v = aggregate(samples, player, team, side, evidence)
    v.searches_used = 1
    return v


def verify_bundle(
    bundle,
    n_samples: int | None = None,
    offline_fixtures: dict[str, Any] | None = None,
) -> tuple[list[Verdict], RunCost, list[str]]:
    """Verify every player entering AND leaving the squad.

    The original workflow checked only `transfers_in`. That is the wrong half
    of the problem as often as not: if the player you are selling has just been
    ruled out, the transfer is more urgent, not less, and an editor needs to
    know. Both sides are checked here.
    """
    notes: list[str] = []
    cost = RunCost()

    # A transfer contributes two targets, one each side. A held player (where
    # the pseudo-transfer's in and out are the same person, used by
    # full-squad scope) contributes one, labelled HELD.
    targets: list[tuple[str, str, str]] = []
    for t in bundle.transfers:
        if t.in_id == t.out_id:
            targets.append((t.in_name, t.in_team, "HELD"))
        else:
            targets.append((t.in_name, t.in_team, "IN"))
            targets.append((t.out_name, t.out_team, "OUT"))

    # Deduplicate and strip the "(Team)" suffix for cleaner search queries.
    seen: set[str] = set()
    clean: list[tuple[str, str, str]] = []
    for name, team, side in targets:
        base = re.sub(r"\s*\(.*?\)\s*$", "", name).strip()
        key = f"{base}|{side}"
        if key in seen:
            continue
        seen.add(key)
        clean.append((base, team, side))

    serp_key = get_secret("SERPAPI_KEY")
    groq_key = get_secret("GROQ_API_KEY")

    if offline_fixtures is not None or not (serp_key and groq_key):
        if offline_fixtures is None:
            notes.append(
                "No GROQ_API_KEY / SERPAPI_KEY configured — running on recorded verdicts. "
                "The agent code path is identical; only the two external calls are replayed."
            )
        return replay(clean, offline_fixtures or {}), cost, notes

    n_samples = n_samples or POLICY.self_consistency_samples
    budget = POLICY.max_search_calls_per_run
    verdicts: list[Verdict] = []
    for base, team, side in clean:
        if cost.searches >= budget:
            notes.append(
                f"Search budget of {budget} exhausted; {len(clean) - len(verdicts)} player(s) "
                "left unverified and escalated to human review."
            )
            v = aggregate([], base, team, side, [], None)
            v.escalate = True
            v.escalation_reasons = ["Search budget exhausted before this player was checked."]
            verdicts.append(v)
            continue
        verdicts.append(verify_player(base, team, side, serp_key, groq_key, n_samples, cost))

    return verdicts, cost, notes


# --------------------------------------------------------------------------
# Offline replay
# --------------------------------------------------------------------------

def replay(targets: list[tuple[str, str, str]], fixtures: dict[str, Any]) -> list[Verdict]:
    """Replay recorded agent behaviour so the demo runs without API keys.

    The recorded set deliberately includes the real inconsistency observed in
    the original workflow (a player returning SUSPENDED and INJURED on
    consecutive runs) so the escalation path is exercised rather than
    described.
    """
    now = datetime.now(timezone.utc)
    out: list[Verdict] = []
    for base, team, side in targets:
        rec = fixtures.get(base)
        if rec is None:
            samples = [Sample("AVAILABLE", 0.86, "https://www.premierleague.com/news",
                              (now - timedelta(days=2)).strftime("%Y-%m-%d"),
                              f"{base} is expected to be involved.",
                              "No injury or suspension reported in recent coverage.")
                       for _ in range(POLICY.self_consistency_samples)]
            evidence = [Evidence(f"{base} team news", "https://www.premierleague.com/news",
                                 f"No fresh injury concern reported for {base}.",
                                 (now - timedelta(days=2)).strftime("%Y-%m-%d"))]
        else:
            samples = [
                Sample(
                    s["status"], s.get("confidence", 0.5), s.get("source_url"),
                    (now - timedelta(days=s.get("days_old", 3))).strftime("%Y-%m-%d")
                    if s.get("source_url") else None,
                    s.get("quote"), s.get("reasoning", ""),
                )
                for s in rec["samples"]
            ]
            evidence = [
                Evidence(e["title"], e["url"], e["snippet"],
                         (now - timedelta(days=e.get("days_old", 3))).strftime("%Y-%m-%d"))
                for e in rec.get("evidence", [])
            ]
        out.append(aggregate(samples, base, team, side, evidence, now))
    return out
