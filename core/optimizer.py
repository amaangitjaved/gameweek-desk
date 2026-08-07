"""Constrained transfer search over a shared budget.

What was wrong before
---------------------
The prototype looped over each squad player independently, gave each one the
full bank balance, took the best affordable replacement, then returned the top
N by gain. That produces recommendations that cannot actually be executed:

* Two transfers each spent the whole bank, so the pair was unaffordable.
* Both transfers could name the same incoming player.
* `bank_balance_after` was read off the last move in the list, so the number
  shown to the user was simply wrong.
* No 3-per-club limit, so it would happily suggest a fourth Arsenal defender.
* Sale proceeds used the current price, ignoring FPL's sell-price rule.
* No accounting for the -4 point hit on transfers beyond the free allowance,
  so a "+1.2 xP" recommendation could be a net loss of 2.8.

This module searches transfer *bundles* against a single shared budget and
validates the resulting squad, so anything it proposes is legal and executable.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import pandas as pd

from core.config import POLICY

HIT_COST = 4.0
# Tolerance for money comparisons. Prices are tenths of a million held as
# floats, so exact equality checks fail unpredictably.
EPS = 1e-6


# --------------------------------------------------------------------------
# FPL rules
# --------------------------------------------------------------------------

def sell_price(purchase_price: float, current_price: float) -> float:
    """FPL sell-price rule: you keep only half of any price rise, rounded down
    to the nearest 0.1m. Price falls are borne in full.
    """
    if current_price <= purchase_price:
        return round(current_price, 1)
    # Work in integer tenths. Doing this in floats silently loses a tenth:
    # 5.6 - 5.0 evaluates to 0.5999999999999996, which floors to 5 tenths
    # instead of 6 and undervalues the sale.
    rise_tenths = round((current_price - purchase_price) * 10)
    kept_tenths = rise_tenths // 2
    return round(purchase_price + kept_tenths / 10.0, 1)


@dataclass
class Transfer:
    out_id: int
    in_id: int
    out_name: str
    in_name: str
    out_team: str
    in_team: str
    position: str
    out_sell: float
    in_cost: float
    xp_out: float
    xp_in: float
    xp_gain: float
    in_start_prob: float
    in_xp_low: float
    in_xp_high: float


@dataclass
class Bundle:
    transfers: list[Transfer]
    gross_xp_gain: float
    hit_points: float
    net_xp_gain: float
    bank_after: float
    club_counts: dict[str, int] = field(default_factory=dict)
    rationale: str = ""

    @property
    def n(self) -> int:
        return len(self.transfers)


# --------------------------------------------------------------------------
# Legality
# --------------------------------------------------------------------------

def _club_counts_after(squad: pd.DataFrame, transfers: list[Transfer], pool: pd.DataFrame) -> dict[str, int]:
    out_ids = {t.out_id for t in transfers}
    counts: dict[str, int] = {}
    for _, row in squad[~squad["id"].isin(out_ids)].iterrows():
        counts[row["team_name"]] = counts.get(row["team_name"], 0) + 1
    for t in transfers:
        counts[t.in_team] = counts.get(t.in_team, 0) + 1
    return counts


def _is_legal(squad: pd.DataFrame, transfers: list[Transfer], pool: pd.DataFrame,
              bank: float) -> tuple[bool, str, float, dict[str, int]]:
    in_ids = [t.in_id for t in transfers]
    out_ids = [t.out_id for t in transfers]

    if len(set(in_ids)) != len(in_ids):
        return False, "duplicate incoming player", 0.0, {}
    if len(set(out_ids)) != len(out_ids):
        return False, "duplicate outgoing player", 0.0, {}
    if set(in_ids) & set(squad["id"]):
        return False, "incoming player already in squad", 0.0, {}

    proceeds = sum(t.out_sell for t in transfers)
    spend = sum(t.in_cost for t in transfers)
    bank_after = round(bank + proceeds - spend, 1)
    if bank_after < -EPS:
        return False, "insufficient funds across the bundle", bank_after, {}

    counts = _club_counts_after(squad, transfers, pool)
    over = [c for c, n in counts.items() if n > POLICY.max_per_club]
    if over:
        return False, f"more than {POLICY.max_per_club} players from {', '.join(over)}", bank_after, counts

    # Like-for-like keeps positional counts valid by construction; assert it.
    for t in transfers:
        if pool.loc[pool["id"] == t.in_id, "element_type"].iloc[0] != \
           squad.loc[squad["id"] == t.out_id, "element_type"].iloc[0]:
            return False, "position mismatch", bank_after, counts

    return True, "", bank_after, counts


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

def _make_transfer(out_row, in_row, purchase_prices: dict[int, float]) -> Transfer:
    out_purchase = purchase_prices.get(int(out_row["id"]), float(out_row["price"]))
    return Transfer(
        out_id=int(out_row["id"]),
        in_id=int(in_row["id"]),
        out_name=f"{out_row['web_name']} ({out_row['team_name']})",
        in_name=f"{in_row['web_name']} ({in_row['team_name']})",
        out_team=out_row["team_name"],
        in_team=in_row["team_name"],
        position=out_row["position"],
        out_sell=sell_price(out_purchase, float(out_row["price"])),
        in_cost=float(in_row["price"]),
        xp_out=float(out_row["xp_horizon"]),
        xp_in=float(in_row["xp_horizon"]),
        xp_gain=float(in_row["xp_horizon"] - out_row["xp_horizon"]),
        in_start_prob=float(in_row["start_prob"]),
        in_xp_low=float(in_row["xp_low"]),
        in_xp_high=float(in_row["xp_high"]),
    )


def optimise(
    squad: pd.DataFrame,
    universe: pd.DataFrame,
    bank: float,
    free_transfers: int = 1,
    max_transfers: int = 2,
    purchase_prices: dict[int, float] | None = None,
    exclude_ids: set[int] | None = None,
    candidates_per_position: int = 14,
    top_k: int = 5,
) -> list[Bundle]:
    """Return the best legal transfer bundles, ranked by net xP gain.

    `exclude_ids` is how the verification agent feeds back into the decision:
    a player flagged as injured or suspended is removed from the candidate
    pool and the search reruns, rather than the flag being a cosmetic warning
    stapled to an unchanged recommendation.
    """
    purchase_prices = purchase_prices or {}
    exclude_ids = exclude_ids or set()

    squad_ids = set(squad["id"].astype(int))
    pool = universe[
        ~universe["id"].isin(squad_ids) & ~universe["id"].isin(exclude_ids)
    ].copy()

    # Shortlist per position keeps the search tractable without meaningfully
    # affecting the optimum: a player outside the top ~14 by xP in their
    # position is not going to be the best available upgrade.
    shortlists: dict[int, pd.DataFrame] = {
        et: g.nlargest(candidates_per_position, "xp_horizon")
        for et, g in pool.groupby("element_type")
    }

    # --- single transfers -------------------------------------------------
    singles: list[Transfer] = []
    for _, out_row in squad.iterrows():
        cands = shortlists.get(int(out_row["element_type"]))
        if cands is None or cands.empty:
            continue
        for _, in_row in cands.iterrows():
            t = _make_transfer(out_row, in_row, purchase_prices)
            if t.xp_gain <= 0:
                continue
            singles.append(t)

    bundles: list[Bundle] = []

    def _consider(ts: list[Transfer]) -> None:
        ok, _reason, bank_after, counts = _is_legal(squad, ts, pool, bank)
        if not ok:
            return
        gross = sum(t.xp_gain for t in ts)
        hit = HIT_COST * max(0, len(ts) - free_transfers)
        net = gross - hit
        threshold = POLICY.min_xp_gain if hit == 0 else POLICY.min_xp_gain
        if net < threshold:
            return
        bundles.append(Bundle(
            transfers=list(ts),
            gross_xp_gain=round(gross, 2),
            hit_points=hit,
            net_xp_gain=round(net, 2),
            bank_after=bank_after,
            club_counts=counts,
        ))

    for t in singles:
        _consider([t])

    # --- multi-transfer bundles ------------------------------------------
    # Enumerate over the strongest singles only. Sorting first means the
    # combinations we examine are the ones plausibly in the optimum.
    if max_transfers >= 2:
        singles_sorted = sorted(singles, key=lambda t: t.xp_gain, reverse=True)[:120]
        for a, b in itertools.combinations(singles_sorted, 2):
            if a.out_id == b.out_id or a.in_id == b.in_id:
                continue
            _consider([a, b])

    if max_transfers >= 3:
        singles_sorted = sorted(singles, key=lambda t: t.xp_gain, reverse=True)[:45]
        for combo in itertools.combinations(singles_sorted, 3):
            ids_out = {t.out_id for t in combo}
            ids_in = {t.in_id for t in combo}
            if len(ids_out) < 3 or len(ids_in) < 3:
                continue
            _consider(list(combo))

    # Rank by net gain, but discount bundles that take a points hit. A hit is
    # an irreversible cost paid against a forecast with a wide error bar, so
    # it should have to clear a margin rather than win by a rounding
    # difference. Fewer transfers break ties.
    def _rank(b: Bundle) -> tuple[float, int]:
        effective = b.net_xp_gain - (POLICY.hit_margin if b.hit_points > 0 else 0.0)
        return (effective, -b.n)

    bundles.sort(key=_rank, reverse=True)

    # Deduplicate bundles that differ only in ordering.
    seen: set[frozenset] = set()
    unique: list[Bundle] = []
    for bd in bundles:
        key = frozenset((t.out_id, t.in_id) for t in bd.transfers)
        if key in seen:
            continue
        seen.add(key)
        unique.append(bd)
        if len(unique) >= top_k:
            break

    for bd in unique:
        bd.rationale = _explain(bd, free_transfers)
    return unique


def _explain(bundle: Bundle, free_transfers: int) -> str:
    parts = []
    for t in bundle.transfers:
        parts.append(
            f"{t.out_name} -> {t.in_name} ({t.position}): "
            f"+{t.xp_gain:.1f} xP over the horizon, "
            f"{t.in_start_prob * 100:.0f}% expected minutes share, "
            f"80% band {t.in_xp_low:.1f}-{t.in_xp_high:.1f}"
        )
    if bundle.hit_points:
        parts.append(
            f"Costs a -{bundle.hit_points:.0f} hit ({bundle.n} transfers, "
            f"{free_transfers} free), so net gain is {bundle.net_xp_gain:.1f} xP."
        )
    parts.append(f"Bank after: £{bundle.bank_after:.1f}m.")
    return " ".join(parts)


# A fixed 15 chosen so the demo exercises the interesting paths: it contains
# players the recorded verification set has opinions about, and it is
# deliberately a little suboptimal so there are genuine upgrades to propose.
DEMO_SQUAD_NAMES = [
    ("Roefs", "Sunderland"), ("Darlow", "Leeds"),
    ("Esteve", "Burnley"), ("Rodon", "Leeds"), ("Ballard", "Sunderland"),
    ("Kilman", "West Ham"), ("Burn", "Newcastle"),
    ("Salah", "Liverpool"), ("Palmer", "Chelsea"), ("Bowen", "West Ham"),
    ("Garner", "Everton"), ("Gruev", "Leeds"),
    ("Haaland", "Man City"), ("Nketiah", "Crystal Palace"), ("Brobbey", "Sunderland"),
]


def demo_squad(universe: pd.DataFrame) -> pd.DataFrame:
    """The curated demo squad, falling back to the generated one if a name
    cannot be resolved against the loaded universe."""
    rows = []
    for web, team in DEMO_SQUAD_NAMES:
        hit = universe[(universe["web_name"] == web) & (universe["team_name"] == team)]
        if hit.empty:
            return default_squad(universe)
        rows.append(hit.iloc[0].to_dict())
    return pd.DataFrame(rows)


def default_squad(universe: pd.DataFrame, budget: float = 100.0, seed: int = 3) -> pd.DataFrame:
    """A legal 15-player squad for the demo, built greedily by value.

    Deliberately built to be slightly suboptimal so the review queue has real
    upgrades to propose.
    """
    need = {1: 2, 2: 5, 3: 5, 4: 3}

    # Phase 1: cheapest legal squad. Starting from a guaranteed-feasible
    # point and spending up is robust; greedy-by-value from an empty squad is
    # not, because it commits the budget to premium players and then cannot
    # afford to fill the remaining positions.
    picked: dict[int, dict] = {}
    club: dict[str, int] = {}
    for _, row in universe.sort_values("price").iterrows():
        et = int(row["element_type"])
        if need.get(et, 0) <= 0:
            continue
        if club.get(row["team_name"], 0) >= POLICY.max_per_club:
            continue
        picked[int(row["id"])] = row.to_dict()
        need[et] -= 1
        club[row["team_name"]] = club.get(row["team_name"], 0) + 1
        if sum(need.values()) == 0:
            break

    if sum(need.values()) > 0:
        raise RuntimeError(
            f"Universe too small to build a legal squad. Unfilled: "
            f"{ {k: v for k, v in need.items() if v} }"
        )

    spend = sum(p["price"] for p in picked.values())

    # Phase 2: spend the remaining budget on the highest-xP legal upgrade
    # available, one swap at a time, leaving a small float in the bank so the
    # demo has a realistic non-zero balance to reason about.
    reserve_bank = 1.5
    for _ in range(40):
        headroom = budget - reserve_bank - spend
        if headroom <= 0:
            break

        best = None
        for out_id, out_p in picked.items():
            et = int(out_p["element_type"])
            out_price = float(out_p["price"])
            out_xp = float(out_p["xp_horizon"])
            club_wo = dict(club)
            club_wo[out_p["team_name"]] -= 1

            cands = universe[
                (universe["element_type"] == et)
                & (~universe["id"].isin(picked.keys()))
                & (universe["price"] <= out_price + headroom + EPS)
                & (universe["xp_horizon"] > out_xp)
            ]
            for _, cand in cands.iterrows():
                if club_wo.get(cand["team_name"], 0) >= POLICY.max_per_club:
                    continue
                gain = float(cand["xp_horizon"]) - out_xp
                if best is None or gain > best[0]:
                    best = (gain, out_id, cand.to_dict())

        if best is None:
            break

        _gain, out_id, in_p = best
        out_p = picked.pop(out_id)
        club[out_p["team_name"]] -= 1
        picked[int(in_p["id"])] = in_p
        club[in_p["team_name"]] = club.get(in_p["team_name"], 0) + 1
        spend += float(in_p["price"]) - float(out_p["price"])

    squad = pd.DataFrame(list(picked.values()))
    if len(squad) != 15:
        raise RuntimeError(f"default_squad built {len(squad)} players, expected 15.")
    return squad
