"""Data ingest with explicit cold-start handling.

The failure this module exists to prevent
-----------------------------------------
The original prototype trained on `total_points` from the live FPL API. In
early August every player's `total_points` is 0, so the target is constant,
the model predicts a constant, every transfer scores a gain of ~0, and the
system silently returns an empty recommendation list. It does not error. It
just quietly does nothing, which is the worst kind of failure.

Here, season maturity is measured first and drives an explicit blend between
prior-season rates and current-season form. The blend weight is surfaced in
the UI so a reviewer can see how much of a recommendation is evidence versus
prior.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.config import (
    FPL_BOOTSTRAP_URL,
    FPL_FIXTURES_URL,
    POSITIONS,
    SNAPSHOT_PATH,
)

# Below this many minutes played league-wide, we treat the season as
# insufficiently mature to trust current-season stats on their own.
PRESEASON_MINUTES_THRESHOLD = 90 * 3  # ~3 full matches per player, i.e. ~GW3


@dataclass
class LoadResult:
    players: pd.DataFrame
    teams: pd.DataFrame
    fixtures: pd.DataFrame
    source: str                 # "snapshot" | "live"
    season_maturity: float      # 0.0 = pre-season, 1.0 = fully mature
    prior_weight: float         # how much the model leans on prior-season data
    current_event: int | None
    warnings: list[str]
    meta: dict[str, Any]


def normalise_name(s: str) -> str:
    """Casefold, strip accents, collapse whitespace. Used for player matching."""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s)


def _load_snapshot() -> dict:
    with open(SNAPSHOT_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _fetch_live(timeout: int = 12) -> dict:
    import requests  # local import so snapshot mode has no network dependency

    headers = {"User-Agent": "gameweek-desk/1.0"}
    boot = requests.get(FPL_BOOTSTRAP_URL, headers=headers, timeout=timeout).json()
    fixtures = requests.get(FPL_FIXTURES_URL, headers=headers, timeout=timeout).json()
    boot["fixtures"] = fixtures
    boot.setdefault("meta", {"label": "Live FPL API", "synthetic": False})
    return boot


def _current_event(events: list[dict]) -> int | None:
    for key in ("is_current", "is_next"):
        for e in events:
            if e.get(key):
                return int(e["id"])
    return None


def _fixture_difficulty(fixtures: pd.DataFrame, current_event: int | None, horizon: int) -> dict[int, float]:
    """Mean opponent difficulty over the next `horizon` gameweeks, per team.

    Only forward-looking fixtures count. The original code averaged every
    home fixture across the entire season and ignored away fixtures entirely,
    which made the feature nearly meaningless.
    """
    if fixtures.empty:
        return {}

    df = fixtures.copy()
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df = df.dropna(subset=["event"])
    df["event"] = df["event"].astype(int)
    if current_event is not None:
        df = df[df["event"] >= current_event]

    home = df[["event", "team_h", "team_h_difficulty"]].rename(
        columns={"team_h": "team", "team_h_difficulty": "difficulty"})
    away = df[["event", "team_a", "team_a_difficulty"]].rename(
        columns={"team_a": "team", "team_a_difficulty": "difficulty"})

    up = pd.concat([home, away], ignore_index=True)
    up["difficulty"] = pd.to_numeric(up["difficulty"], errors="coerce")
    up["team"] = pd.to_numeric(up["team"], errors="coerce")
    up = up.dropna(subset=["team", "difficulty"])
    up["team"] = up["team"].astype(int)

    up = up.sort_values(["team", "event"]).groupby("team", as_index=False).head(horizon)
    return up.groupby("team")["difficulty"].mean().to_dict()


def load(source: str = "snapshot", horizon: int = 4) -> LoadResult:
    """Load player/team/fixture data and annotate it with season maturity.

    `source="live"` falls back to the snapshot on any failure rather than
    raising, because a demo that 500s when a third-party API blips is not a
    demo. The fallback is reported in `warnings` so it is never silent.
    """
    warnings: list[str] = []
    actual_source = source

    if source == "live":
        try:
            raw = _fetch_live()
        except Exception as exc:
            warnings.append(f"Live FPL API unavailable ({type(exc).__name__}); fell back to snapshot.")
            raw = _load_snapshot()
            actual_source = "snapshot"
    else:
        raw = _load_snapshot()

    players = pd.DataFrame(raw["elements"])
    teams = pd.DataFrame(raw["teams"])
    fixtures = pd.DataFrame(raw.get("fixtures", []))
    events = raw.get("events", [])

    team_map = dict(zip(teams["id"], teams["name"]))
    if "team_name" not in players.columns:
        players["team_name"] = players["team"].map(team_map)

    cur = _current_event(events)

    # ---- Season maturity -------------------------------------------------
    minutes = pd.to_numeric(players.get("minutes", 0), errors="coerce").fillna(0)
    median_minutes = float(minutes.median())
    maturity = float(min(1.0, median_minutes / (90 * 19)))  # 1.0 at ~half a season
    if median_minutes < PRESEASON_MINUTES_THRESHOLD:
        warnings.append(
            f"Pre-season / early season detected (median minutes played = {median_minutes:.0f}). "
            "Current-season statistics are not yet informative; recommendations lean on "
            "prior-season per-90 rates and will carry lower confidence."
        )

    # Prior weight decays as real minutes accumulate. At zero minutes the model
    # is 100% prior; by mid-season it is mostly current form.
    prior_weight = float(max(0.15, 1.0 - maturity))

    # ---- Derived columns -------------------------------------------------
    difficulty = _fixture_difficulty(fixtures, cur, horizon)
    players["difficulty"] = players["team"].map(difficulty).fillna(3.0)
    players["position"] = players["element_type"].map(POSITIONS)
    players["price"] = pd.to_numeric(players["now_cost"], errors="coerce").fillna(0) / 10.0

    players["full_name"] = (
        players["first_name"].astype(str).str.strip() + " " + players["second_name"].astype(str).str.strip()
    )
    players["web_name_norm"] = players["web_name"].map(normalise_name)
    players["full_name_norm"] = players["full_name"].map(normalise_name)
    players["team_name_norm"] = players["team_name"].map(normalise_name)

    # Prior columns are guaranteed present in the snapshot. When running live
    # they may be absent, so derive a conservative fallback from price.
    if "prior_points_per90" not in players.columns:
        warnings.append(
            "Live feed carries no prior-season per-90 rates; using a price-based "
            "prior. Backfill from the previous season's data before trusting "
            "August recommendations."
        )
        players["prior_points_per90"] = 2.0 + 0.28 * (players["price"] - 4.0).clip(lower=0)
        players["prior_minutes_share"] = (0.42 + 0.085 * (players["price"] - 4.0).clip(lower=0)).clip(0.1, 0.98)

    for col in ("minutes", "total_points", "ict_index", "selected_by_percent"):
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce").fillna(0)
        else:
            players[col] = 0.0

    return LoadResult(
        players=players,
        teams=teams,
        fixtures=fixtures,
        source=actual_source,
        season_maturity=maturity,
        prior_weight=prior_weight,
        current_event=cur,
        warnings=warnings,
        meta=raw.get("meta", {}),
    )


# --------------------------------------------------------------------------
# Player resolution
# --------------------------------------------------------------------------

@dataclass
class Resolution:
    matched: list[dict]
    missing: list[str]
    ambiguous: list[dict]


def resolve_players(players: pd.DataFrame, names: list[str]) -> Resolution:
    """Resolve free-text player names to unique rows.

    `web_name` is not unique: Dean Henderson (GK, Crystal Palace) and Jordan
    Henderson (MID, Brentford) both render as "Henderson". Silently taking the
    first match picks the wrong position and produces an illegal transfer, so
    ambiguity is returned to the caller for a human to settle rather than
    guessed at.
    """
    matched: list[dict] = []
    missing: list[str] = []
    ambiguous: list[dict] = []
    suffix_re = re.compile(r"^(?P<name>.*?)\s*\((?P<team>.*?)\)\s*$")

    for raw in names:
        key = str(raw).strip()
        if not key:
            continue
        m = suffix_re.match(key)
        name_key, team_key = (m.group("name").strip(), m.group("team").strip()) if m else (key, None)
        nk = normalise_name(name_key)

        if " " in nk:
            hits = players[players["full_name_norm"].eq(nk)]
            if hits.empty:
                mask = pd.Series(True, index=players.index)
                for tok in nk.split(" "):
                    mask &= players["full_name_norm"].str.contains(re.escape(tok), regex=True)
                hits = players[mask]
        else:
            hits = players[players["web_name_norm"].eq(nk)]
            if hits.empty:
                hits = players[players["full_name_norm"].str.contains(re.escape(nk), regex=True)]

        if team_key:
            hits = hits[hits["team_name_norm"].eq(normalise_name(team_key))]

        if hits.empty:
            missing.append(key)
        elif len(hits) > 1:
            ambiguous.append({
                "input": key,
                "options": hits[["id", "full_name", "web_name", "team_name", "position", "price"]]
                .to_dict("records"),
            })
        else:
            matched.append(hits.iloc[0].to_dict())

    return Resolution(matched=matched, missing=missing, ambiguous=ambiguous)


def cache_key(*parts: Any) -> str:
    return "|".join(str(p) for p in parts) + f"|{int(time.time() // 3600)}"
