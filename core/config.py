"""Central configuration and tunable policy thresholds.

Every number a reviewer might argue with lives here, not scattered through the
code. That is deliberate: in the panel walkthrough these are the dials that
express policy, and policy should be legible in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

# --------------------------------------------------------------------------
# Scenario assumptions (stated explicitly, per the brief)
# --------------------------------------------------------------------------

SCENARIO = {
    "team_size": 6,
    "subscribers": 40_000,
    "gameweeks_per_season": 38,
    "players_tracked": 600,
    "manual_minutes_per_player": 2.5,
}


# --------------------------------------------------------------------------
# Model / optimizer policy
# --------------------------------------------------------------------------

@dataclass
class Policy:
    """Thresholds governing what the system will and will not propose."""

    # Fixture horizon (gameweeks) used to weight opponent difficulty.
    fixture_horizon: int = 4

    # Minimum expected-points gain before a transfer is worth proposing at all.
    min_xp_gain: float = 0.6

    # A bundle that costs a -4 hit must beat the best hit-free bundle by this
    # margin before it is ranked above it. Without this, a hit bundle wins on
    # a 0.02 xP difference, which is far inside the model's error bar and is
    # false precision presented as a decision.
    hit_margin: float = 1.5

    # Squad legality
    max_per_club: int = 3
    squad_size: int = 15

    # Minutes model: players below this predicted-start probability are
    # treated as rotation risks and down-weighted rather than excluded, so a
    # human can still see and override the call.
    rotation_risk_threshold: float = 0.55

    # Verification agent
    max_search_calls_per_run: int = 24
    self_consistency_samples: int = 3
    # If sampled verdicts disagree at or above this rate, escalate to human
    # rather than surfacing a single confident-looking answer.
    disagreement_escalation_threshold: float = 0.34
    # Evidence older than this is treated as stale and cannot on its own
    # justify a blocking flag.
    max_evidence_age_days: int = 10

    # Confidence bands
    high_confidence: float = 0.80
    low_confidence: float = 0.50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


POLICY = Policy()


# --------------------------------------------------------------------------
# Cost model (published list prices; used for the live spend meter)
# --------------------------------------------------------------------------

COSTS = {
    # SerpAPI developer plan: 5,000 searches / month / $75 => $0.015 per search
    "usd_per_search": 0.015,
    # Groq llama-3.3-70b-versatile, approximate blended token price
    "usd_per_1k_input_tokens": 0.00059,
    "usd_per_1k_output_tokens": 0.00079,
    # Fully-loaded hourly cost of an ops reviewer, for the time-saved comparison
    "usd_per_reviewer_hour": 32.0,
}


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SNAPSHOT_PATH = os.path.join(DATA_DIR, "snapshot_2026_27.json")
EVAL_SET_PATH = os.path.join(DATA_DIR, "eval_set.json")
AUDIT_LOG_PATH = os.path.join(DATA_DIR, "audit_log.jsonl")

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
VALID_FORMATION_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
VALID_FORMATION_MAX = {1: 1, 2: 5, 3: 5, 4: 3}


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a secret from Streamlit secrets if present, else the environment.

    Kept import-safe so core modules can be unit tested without Streamlit.
    """
    try:
        import streamlit as st  # noqa: PLC0415

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)
