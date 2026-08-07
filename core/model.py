"""Expected-points model, with validation instead of assertion.

What was wrong before
---------------------
The prototype fit LightGBM on `total_points` using `[now_cost, ict_index,
selected_by_percent, difficulty]`, then predicted on the same rows it had just
trained on, and called the output "xP". Three problems:

1. No held-out data, so the reported quantity was an in-sample reconstruction
   of points already scored, not a forecast of points to come.
2. `ict_index` is a function of the same events that generate points, so the
   model was partly reading the answer off the feature vector.
3. No uncertainty. A single number with no error bar cannot support a
   human deciding whether to trust it.

What this does instead
----------------------
Ridge regression, closed form, with k-fold cross-validation and a residual
based prediction interval. Deliberately simple: the model is not the
interesting part of this system, the human/AI boundary is, and a simple model
I can validate beats a complex one I cannot. The CV metrics are shown in the
UI so the reviewer sees the model's accuracy rather than being told it.

Honest caveat, surfaced in-app: on the synthetic snapshot the priors were
generated from price, so CV R^2 is optimistic. On live data with real
prior-season rates, expect materially lower.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# FPL scoring constants by position
GOAL_POINTS = {1: 10, 2: 6, 3: 5, 4: 4}
CS_POINTS = {1: 4, 2: 4, 3: 1, 4: 0}
FEATURES = [
    "price", "difficulty", "prior_minutes_share", "selected_by_percent",
    "is_gk", "is_def", "is_mid", "is_fwd",
]


@dataclass
class ModelReport:
    cv_r2: float
    cv_mae: float
    residual_sd: float
    alpha: float
    n_train: int
    n_folds: int
    feature_weights: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Ridge regression, closed form
# --------------------------------------------------------------------------

def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    Xs, mu, sd = _standardise(X)
    n, p = Xs.shape
    A = Xs.T @ Xs + alpha * np.eye(p)
    w = np.linalg.solve(A, Xs.T @ (y - y.mean()))
    return w, float(y.mean()), mu, sd


def _ridge_predict(X: np.ndarray, w: np.ndarray, b: float, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mu) / sd) @ w + b


def _kfold_indices(n: int, k: int, seed: int = 7) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [f for f in np.array_split(idx, k) if len(f) > 0]


def _cross_validate(X: np.ndarray, y: np.ndarray, alpha: float, k: int = 5) -> tuple[float, float, float]:
    """Return (R^2, MAE, residual sd) computed strictly out of fold."""
    folds = _kfold_indices(len(y), k)
    preds = np.zeros_like(y, dtype=float)
    for f in folds:
        mask = np.ones(len(y), dtype=bool)
        mask[f] = False
        w, b, mu, sd = _ridge_fit(X[mask], y[mask], alpha)
        preds[f] = _ridge_predict(X[f], w, b, mu, sd)
    resid = y - preds
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, float(np.abs(resid).mean()), float(resid.std())


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------

def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    d = df.copy()
    for et, col in [(1, "is_gk"), (2, "is_def"), (3, "is_mid"), (4, "is_fwd")]:
        d[col] = (d["element_type"] == et).astype(float)
    return d[FEATURES].astype(float).to_numpy()


def _difficulty_multiplier(difficulty: pd.Series) -> pd.Series:
    """Map opponent difficulty (1 easy .. 5 hard) to an output multiplier.

    Centred on 3 so an average run of fixtures is neutral. The +/-18% span is
    a judgement call, not a fitted parameter, and is exposed as such.
    """
    return (1.0 + (3.0 - pd.to_numeric(difficulty, errors="coerce").fillna(3.0)) * 0.09).clip(0.6, 1.4)


def _minutes_model(df: pd.DataFrame) -> pd.Series:
    """Expected share of available minutes.

    Blends the prior-season share with current-season minutes once enough of
    the season has been played for the latter to mean anything. Kept separate
    from the points model because rotation risk and scoring rate are different
    questions, and a human reviewer treats them differently: a nailed 4-point
    player often beats an explosive one who might not start.
    """
    prior = pd.to_numeric(df.get("prior_minutes_share", 0.6), errors="coerce").fillna(0.6)
    mins = pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0)
    games = max(1.0, float(mins.max()) / 90.0)
    current = (mins / (games * 90.0)).clip(0, 1)
    w = float(min(1.0, games / 8.0))  # trust current-season minutes from ~GW8
    return ((1 - w) * prior + w * current).clip(0.05, 1.0)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def score_players(
    players: pd.DataFrame,
    prior_weight: float,
    horizon: int = 4,
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0),
) -> tuple[pd.DataFrame, ModelReport]:
    """Attach xP and an uncertainty band to every player.

    Returns the scored frame and a report of out-of-fold validation metrics.
    """
    df = players.copy()

    # Label: prior-season points per 90. Known before the current season
    # starts, so using it as a training target involves no leakage into the
    # forecast horizon. Features deliberately exclude any points-derived
    # quantity (notably ict_index, which the old model used).
    y = pd.to_numeric(df["prior_points_per90"], errors="coerce").fillna(0).to_numpy(dtype=float)
    X = _design_matrix(df)

    best = min(
        ((a, *_cross_validate(X, y, a)) for a in alpha_grid),
        key=lambda t: t[2],   # lowest out-of-fold MAE
    )
    alpha, cv_r2, cv_mae, resid_sd = best

    w, b, mu, sd = _ridge_fit(X, y, alpha)
    model_pp90 = _ridge_predict(X, w, b, mu, sd)

    # Blend the model's structural estimate with the observed prior rate.
    # In pre-season prior_weight is 1.0 and the observed rate dominates; as
    # real minutes accumulate the model's fixture-aware view takes over.
    blended_pp90 = prior_weight * y + (1 - prior_weight) * model_pp90
    blended_pp90 = np.clip(blended_pp90, 0.0, None)

    minutes_share = _minutes_model(df)
    diff_mult = _difficulty_multiplier(df["difficulty"])

    # Expected points for ONE gameweek, then across the horizon.
    xp_gw = pd.Series(blended_pp90, index=df.index) * minutes_share * diff_mult
    df["start_prob"] = minutes_share
    df["difficulty_multiplier"] = diff_mult
    df["xp_per_gw"] = xp_gw
    df["xp_horizon"] = xp_gw * horizon

    # Uncertainty: out-of-fold residual sd, widened when we are leaning on
    # priors (pre-season) and when the player is a rotation risk.
    uncertainty = resid_sd * (1.0 + 0.6 * prior_weight) * (1.0 + (1.0 - minutes_share))
    df["xp_sd"] = uncertainty * minutes_share * horizon
    df["xp_low"] = (df["xp_horizon"] - 1.28 * df["xp_sd"]).clip(lower=0)   # ~80% band
    df["xp_high"] = df["xp_horizon"] + 1.28 * df["xp_sd"]
    df["value_per_million"] = df["xp_horizon"] / df["price"].clip(lower=0.1)

    notes = []
    if prior_weight > 0.8:
        notes.append(
            "Pre-season: xP is driven almost entirely by prior-season per-90 rates. "
            "Treat the ranking as a prior, not a forecast, and widen your own error bars."
        )
    if bool(players.attrs.get("synthetic", True)):
        notes.append(
            "Snapshot priors were synthesised from price, so cross-validated R^2 is "
            "optimistic. On live prior-season data expect a materially lower figure."
        )

    report = ModelReport(
        cv_r2=cv_r2,
        cv_mae=cv_mae,
        residual_sd=resid_sd,
        alpha=alpha,
        n_train=len(y),
        n_folds=5,
        feature_weights={f: round(float(wi), 4) for f, wi in zip(FEATURES, w)},
        notes=notes,
    )
    return df, report
