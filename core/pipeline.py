"""End-to-end gameweek run: score -> propose -> verify -> revise.

The revise step is the point of the whole system. In the original workflow the
agent's injury warning was a string appended to an unchanged recommendation:
the model said "buy Dalot", the agent said "Dalot is injured", and the two
statements sat next to each other with nothing reconciling them. A human had
to notice the contradiction and work out the implication.

Here a confirmed blocking flag removes the player from the candidate pool and
the optimiser reruns. The system resolves what it can and escalates what it
cannot, which is the division of labour the whole design turns on.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.config import DATA_DIR, POLICY
from core.data import LoadResult
from core.optimizer import Bundle, optimise
from core.verify import RunCost, Verdict, verify_bundle

RECORDED_PATH = os.path.join(DATA_DIR, "recorded_verdicts.json")


@dataclass
class GameweekRun:
    initial: Bundle | None
    final: Bundle | None
    verdicts: list[Verdict]
    cost: RunCost
    notes: list[str] = field(default_factory=list)
    blocked_players: list[str] = field(default_factory=list)
    escalated_players: list[str] = field(default_factory=list)
    revised: bool = False
    alternatives: list[Bundle] = field(default_factory=list)

    @property
    def needs_human(self) -> bool:
        return bool(self.escalated_players) or self.final is None


def load_recorded() -> dict[str, Any]:
    with open(RECORDED_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: v for k, v in data.items() if not k.startswith("_")}


class _SquadBundle:
    """Adapter so a whole squad can be passed through the same verification
    path as a transfer bundle, without duplicating the orchestration."""

    def __init__(self, transfers, squad_rows):
        self.transfers = transfers
        self._squad_rows = squad_rows


def _squad_targets(squad: pd.DataFrame, bundle: Bundle | None):
    """Build a pseudo-bundle covering every squad member plus any incoming
    player, so full-squad verification reuses verify_bundle unchanged."""
    from core.optimizer import Transfer

    ts = []
    for _, row in squad.iterrows():
        ts.append(Transfer(
            out_id=int(row["id"]), in_id=int(row["id"]),
            out_name=f"{row['web_name']} ({row['team_name']})",
            in_name=f"{row['web_name']} ({row['team_name']})",
            out_team=row["team_name"], in_team=row["team_name"],
            position=row["position"], out_sell=float(row["price"]),
            in_cost=float(row["price"]), xp_out=float(row["xp_horizon"]),
            xp_in=float(row["xp_horizon"]), xp_gain=0.0,
            in_start_prob=float(row["start_prob"]),
            in_xp_low=float(row["xp_low"]), in_xp_high=float(row["xp_high"]),
        ))
    if bundle:
        ts.extend(bundle.transfers)
    return _SquadBundle(ts, squad)


def run_gameweek(
    squad: pd.DataFrame,
    universe: pd.DataFrame,
    bank: float,
    free_transfers: int,
    max_transfers: int,
    live_agent: bool = False,
    scope: str = "bundle",
) -> GameweekRun:
    """`scope` controls the cost/confidence trade-off directly.

    "bundle" checks only the players entering and leaving the squad: cheap,
    a handful of searches, but blind to a player you were not planning to move
    who has just been ruled out.

    "squad" checks all fifteen plus incoming: roughly four times the spend and
    latency, and the only setting that catches an availability problem in a
    player the optimiser had no opinion about. Which is correct depends on
    whether it is a quiet Tuesday or an hour before the deadline, so it is an
    operator decision rather than a constant.
    """
    fixtures = None if live_agent else load_recorded()

    bundles = optimise(
        squad, universe, bank=bank,
        free_transfers=free_transfers, max_transfers=max_transfers, top_k=5,
    )
    if not bundles:
        return GameweekRun(
            initial=None, final=None, verdicts=[], cost=RunCost(),
            notes=[
                f"No transfer clears the +{POLICY.min_xp_gain} xP threshold. "
                "Holding is the recommendation. This is a real outcome, not an error: "
                "most gameweeks for a well-built squad should end here."
            ],
        )

    initial = bundles[0]
    to_verify = _squad_targets(squad, initial) if scope == "squad" else initial
    verdicts, cost, notes = verify_bundle(to_verify, offline_fixtures=fixtures)

    # A blocking flag only counts if the agent is confident AND did not
    # escalate. Anything escalated goes to a human instead of being acted on
    # automatically -- the system does not get to quietly veto on a shaky call.
    blocked = [v for v in verdicts if v.is_blocking and v.side == "IN"]
    escalated = [v for v in verdicts if v.escalate]

    final = initial
    revised = False
    alternatives: list[Bundle] = []

    if blocked:
        blocked_names = {v.player for v in blocked}
        exclude_ids = {
            int(r["id"]) for _, r in universe.iterrows()
            if r["web_name"] in blocked_names
        }
        revised_bundles = optimise(
            squad, universe, bank=bank,
            free_transfers=free_transfers, max_transfers=max_transfers,
            exclude_ids=exclude_ids, top_k=3,
        )
        if revised_bundles:
            final = revised_bundles[0]
            alternatives = revised_bundles[1:]
            revised = True
            notes.append(
                f"{', '.join(sorted(blocked_names))} removed from the candidate pool on a "
                "confirmed availability flag; the optimiser reran against the "
                "reduced pool and this is the replacement recommendation."
            )
        else:
            final = None
            notes.append(
                f"{', '.join(sorted(blocked_names))} blocked on availability and no "
                "alternative clears the threshold. Recommending hold."
            )
    else:
        alternatives = bundles[1:3]

    # Verify the revised bundle too. Skipping this would mean the replacement
    # goes out unchecked, which is exactly the gap the system exists to close.
    if revised and final is not None:
        extra, extra_cost, extra_notes = verify_bundle(final, offline_fixtures=fixtures)
        seen = {(v.player, v.side) for v in verdicts}
        verdicts.extend(v for v in extra if (v.player, v.side) not in seen)
        cost.searches += extra_cost.searches
        cost.input_tokens += extra_cost.input_tokens
        cost.output_tokens += extra_cost.output_tokens
        cost.llm_calls += extra_cost.llm_calls
        notes.extend(extra_notes)
        escalated = [v for v in verdicts if v.escalate]

    return GameweekRun(
        initial=initial,
        final=final,
        verdicts=verdicts,
        cost=cost,
        notes=notes,
        blocked_players=sorted({v.player for v in blocked}),
        escalated_players=sorted({v.player for v in escalated}),
        revised=revised,
        alternatives=alternatives,
    )
