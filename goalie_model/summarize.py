"""
summarize.py
------------
Post-processing utilities: tidy skill tables and derived metrics.

Public API
----------
summarize_skills(state, season_idx, centered)  -> (shooter_df, goalie_df)
compute_gsax(state, season_idx)                -> goalie_df
skill_trajectory(state, player_id, role, season_idx) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import json
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch

from .data_prep import ModelData, _safe_logit
from .fit import ModelState, _rebuild_model_data_from_state
from .irt_model import DynamicIRTModel


log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Player name lookup (NHL web API)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_player_names(
    player_ids: Iterable[int],
    cache_path: Optional[str] = None,
    delay: float = 0.5,
) -> Dict[int, str]:
    """
    Resolve a collection of NHL player IDs to full names.

    Uses the public NHL web API (no key required).  Results are
    optionally persisted to a local JSON cache so subsequent calls
    skip already-resolved IDs.

    Parameters
    ----------
    player_ids : iterable of int
        Player IDs to look up (duplicates are deduplicated).
    cache_path : str, optional
        Path to a JSON file used as a persistent name cache.  If the
        file exists it is loaded first; newly resolved names are
        appended and re-saved.
    delay : float
        Seconds to pause between HTTP requests (default 0.15 s).

    Returns
    -------
    dict mapping int player_id -> "First Last" string
    """
    unique_ids = list(dict.fromkeys(int(float(pid)) for pid in player_ids))

    # Load cache
    cache: Dict[int, str] = {}
    if cache_path:
        try:
            with open(cache_path) as fh:
                cache = {int(k): v for k, v in json.load(fh).items()}
        except FileNotFoundError:
            pass

    to_fetch = [pid for pid in unique_ids if pid not in cache]
    if to_fetch:
        log.info(
            "Fetching names for %d players from NHL API (%.2fs delay) …",
            len(to_fetch), delay,
        )
    for i, pid in enumerate(to_fetch):
        for attempt in range(4):
            try:
                r = requests.get(
                    f"https://api-web.nhle.com/v1/player/{pid}/landing",
                    timeout=10,
                )
                if r.status_code == 429:
                    wait = 10 * (2 ** attempt)
                    log.warning("Rate limited (429) on player %s — waiting %ds …", pid, wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                d = r.json()
                first = d.get("firstName", {}).get("default", "")
                last  = d.get("lastName",  {}).get("default", "")
                cache[pid] = f"{first} {last}".strip() or str(pid)
                break
            except requests.exceptions.HTTPError:
                raise
            except Exception as exc:
                log.warning("Could not fetch name for player %s: %s", pid, exc)
                cache[pid] = str(pid)
                break
        else:
            log.warning("Giving up on player %s after 4 attempts, using ID as name.", pid)
            cache[pid] = str(pid)
        if delay and i < len(to_fetch) - 1:
            time.sleep(delay)

    # Persist updated cache
    if cache_path and to_fetch:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump({str(k): v for k, v in cache.items()}, fh, indent=2)

    return {pid: cache.get(pid, str(pid)) for pid in unique_ids}


# ─────────────────────────────────────────────────────────────────────────────
# Internal: reconstruct model in eval mode from a ModelState
# ─────────────────────────────────────────────────────────────────────────────

def _model_from_state(state: ModelState) -> Tuple[DynamicIRTModel, ModelData]:
    data = _rebuild_model_data_from_state(state, state.config)
    model = DynamicIRTModel(data, state.config)
    model.load_state_dict_numpy(state.param_dict)
    model.eval()
    return model, data


# ─────────────────────────────────────────────────────────────────────────────
# Tidy skill tables
# ─────────────────────────────────────────────────────────────────────────────

def summarize_skills(
    state: ModelState,
    season_idx: int = 1,
    centered: Optional[bool] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return tidy DataFrames of shooter and goalie skills for one season.

    Each row is one (player, week) observation.

    Parameters
    ----------
    state : ModelState
        Fitted model state.
    season_idx : int
        0 = previous season, 1 = current season (default).
    centered : bool, optional
        Subtract weekly mean so that 0 = league average.
        Defaults to state.config.center_weekly.

    Returns
    -------
    shooter_df : pd.DataFrame
        Columns: shooter_id, week, theta, mu_theta
    goalie_df : pd.DataFrame
        Columns: goalie_id, week, phi, mu_phi
    """
    if centered is None:
        centered = state.config.center_weekly

    model, data = _model_from_state(state)

    T = data.weeks_per_season[season_idx]
    if T == 0:
        empty_s = pd.DataFrame(columns=["shooter_id", "week", "theta", "mu_theta"])
        empty_g = pd.DataFrame(columns=["goalie_id", "week", "phi", "mu_phi"])
        return empty_s, empty_g

    # θ: [N_shooters, T]
    theta_arr = model.get_shooter_skills(season_idx=season_idx, center=centered)
    mu_theta_arr = model.mu_theta.detach().cpu().numpy()

    shooter_rows = []
    for p, pid in enumerate(data.idx_to_shooter_id):
        for t in range(T):
            shooter_rows.append(
                {
                    "shooter_id": pid,
                    "week": t,
                    "theta": float(theta_arr[p, t]),
                    "mu_theta": float(mu_theta_arr[p]),
                }
            )
    shooter_df = pd.DataFrame(shooter_rows)

    # φ: [N_goalies, T]
    phi_arr = model.get_goalie_skills(season_idx=season_idx, center=centered)
    mu_phi_arr = model.mu_phi.detach().cpu().numpy()

    goalie_rows = []
    for q, qid in enumerate(data.idx_to_goalie_id):
        for t in range(T):
            goalie_rows.append(
                {
                    "goalie_id": qid,
                    "week": t,
                    "phi": float(phi_arr[q, t]),
                    "mu_phi": float(mu_phi_arr[q]),
                }
            )
    goalie_df = pd.DataFrame(goalie_rows)

    return shooter_df, goalie_df


# ─────────────────────────────────────────────────────────────────────────────
# Goals Saved Above xG (GSAx)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gsax(
    state: ModelState,
    season_idx: int = 1,
) -> pd.DataFrame:
    """
    Compute Goals Saved Above Expected xG for each goalie.

    For each shot i faced by goalie q:
        p_full_i  = sigmoid(β₀ + α·η_i^xG + θ[shooter, s, t] − φ[q, s, t])
        p_no_g_i  = sigmoid(β₀ + α·η_i^xG + θ[shooter, s, t])   [φ = 0]

    GSAx(q) = Σ_{i: goalie=q} (p_no_g_i − p_full_i)
            = expected extra goals allowed without φ vs with φ

    A positive GSAx means the goalie saved more goals than an average goalie would.

    Parameters
    ----------
    state : ModelState
    season_idx : int
        0 = previous season, 1 = current season.

    Returns
    -------
    pd.DataFrame
        Columns: goalie_id, shots_faced, goals_actual, goals_xg_only,
                 goals_model_predicted, gsax
    """
    model, data = _model_from_state(state)

    # Filter to given season
    s_mask = (data.season_idx == season_idx)
    if s_mask.sum() == 0:
        return pd.DataFrame(
            columns=[
                "goalie_id", "shots_faced", "goals_actual",
                "goals_xg_only", "goals_model_predicted", "gsax",
            ]
        )

    with torch.no_grad():
        logit_full = model.predict_logit(data)  # [N_shots]

        # Compute logit without goalie contribution
        theta_i = model.theta[data.shooter_idx, data.season_idx, data.week_idx]
        phi_j   = model.phi[data.goalie_idx,   data.season_idx, data.week_idx]
        xg = data.xg_logit
        logit_no_goalie = model.beta0 + model.alpha * xg + theta_i  # φ=0, actual θ

        season_arr = data.season_idx.numpy()
        season_mask_t = torch.tensor(season_arr == season_idx)
        # Shot-weighted league-mean θ for this season — use as baseline so
        # shooter_adj is centred at 0 across all goalies
        mean_theta_season = float(theta_i[season_mask_t].mean())
        mean_phi_season   = float(phi_j[season_mask_t].mean())
        logit_avg_shooter = model.beta0 + model.alpha * xg + mean_theta_season  # league-avg θ, φ=0

        p_full = torch.sigmoid(logit_full).numpy()
        p_no_goalie = torch.sigmoid(logit_no_goalie).numpy()
        # Raw xG probability (inverse-logit of stored xg_logit)
        p_xg_raw = torch.sigmoid(data.xg_logit).numpy()
        # IRT baseline: league-average shooter (shot-weighted mean θ), average goalie (φ=0)
        p_avg_shooter = torch.sigmoid(logit_avg_shooter).numpy()
        y = data.y.numpy()
        goalie_idx_arr = data.goalie_idx.numpy()
        season_mask = (season_arr == season_idx)

    rows = []
    for q, qid in enumerate(data.idx_to_goalie_id):
        mask = season_mask & (goalie_idx_arr == q)
        if mask.sum() == 0:
            continue
        shots_faced = int(mask.sum())
        goals_actual = float(y[mask].sum())
        goals_xg_only = float(p_xg_raw[mask].sum())   # expected goals from xG alone
        goals_model_pred = float(p_full[mask].sum())
        # gsax_raw: naive, ignores who shot on the goalie
        gsax_raw = round(goals_xg_only - goals_actual, 2)
        # shooter_adj: Σ(p_no_goalie − p_avg_shooter)
        #   compares actual shooter quality faced vs league-average shot (shot-weighted mean θ)
        #   positive → faced tougher-than-average shooters
        #   negative → faced easier-than-average shooters
        shooter_adj = round(float((p_no_goalie[mask] - p_avg_shooter[mask]).sum()), 2)
        # gsax: IRT-adjusted, credit for goalie's own φ skill only
        gsax = float((p_no_goalie[mask] - p_full[mask]).sum())

        rows.append(
            {
                "goalie_id": qid,
                "shots_faced": shots_faced,
                "goals_actual": goals_actual,
                "goals_xg_only": goals_xg_only,
                "gsax_raw": gsax_raw,
                "shooter_adj": shooter_adj,
                "goals_model_predicted": goals_model_pred,
                "gsax": gsax,
            }
        )

    df = pd.DataFrame(rows)
    if len(df):
        df["goalie_id"] = df["goalie_id"].apply(lambda x: int(float(x)) if x == x else x).astype("Int64")
        df = df.sort_values("gsax", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Single-player skill trajectory
# ─────────────────────────────────────────────────────────────────────────────

def skill_trajectory(
    state: ModelState,
    player_id: str,
    role: str = "goalie",
    season_idx: int = 1,
    centered: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Return the week-by-week skill trajectory for a single player.

    Parameters
    ----------
    state : ModelState
    player_id : str
        The raw player ID string (as stored in the data).
    role : 'goalie' or 'shooter'
    season_idx : int
        0 = previous season, 1 = current season.
    centered : bool, optional
        Center by weekly mean.

    Returns
    -------
    pd.DataFrame
        Columns: week, skill  (φ for goalies, θ for shooters)
        Returns empty DataFrame if player not found.
    """
    if centered is None:
        centered = state.config.center_weekly

    model, data = _model_from_state(state)
    T = data.weeks_per_season[season_idx]

    if role == "goalie":
        mapping = data.goalie_id_to_idx
        phi_arr = model.get_goalie_skills(season_idx=season_idx, center=centered)
        if player_id not in mapping:
            return pd.DataFrame(columns=["week", "skill"])
        q = mapping[player_id]
        skill_label = "phi"
        arr = phi_arr[q, :T]
    elif role == "shooter":
        mapping = data.shooter_id_to_idx
        theta_arr = model.get_shooter_skills(season_idx=season_idx, center=centered)
        if player_id not in mapping:
            return pd.DataFrame(columns=["week", "skill"])
        p = mapping[player_id]
        skill_label = "theta"
        arr = theta_arr[p, :T]
    else:
        raise ValueError(f"role must be 'goalie' or 'shooter', got: {role!r}")

    return pd.DataFrame({"week": np.arange(T), skill_label: arr})


# ─────────────────────────────────────────────────────────────────────────────
# Brier / log-loss diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(
    state: ModelState,
    season_idx: Optional[int] = None,
) -> dict:
    """
    Compute Brier score and log-loss for the fitted model on stored shots,
    and compare against an xG-only baseline (no player effects).

    Parameters
    ----------
    state : ModelState
    season_idx : int or None
        If provided, evaluate only shots from that season. None = all shots.

    Returns
    -------
    dict with keys: brier_model, logloss_model, brier_xg_only, logloss_xg_only
    """
    model, data = _model_from_state(state)

    with torch.no_grad():
        logit_full = model.predict_logit(data)
        p_model = torch.sigmoid(logit_full).numpy()
        p_xg_only = torch.sigmoid(data.xg_logit).numpy()
        y = data.y.numpy()
        s_arr = data.season_idx.numpy()

    if season_idx is not None:
        mask = s_arr == season_idx
        p_model = p_model[mask]
        p_xg_only = p_xg_only[mask]
        y = y[mask]

    eps = 1e-7
    brier_model = float(np.mean((p_model - y) ** 2))
    brier_xg = float(np.mean((p_xg_only - y) ** 2))
    logloss_model = float(
        -np.mean(y * np.log(p_model + eps) + (1 - y) * np.log(1 - p_model + eps))
    )
    logloss_xg = float(
        -np.mean(y * np.log(p_xg_only + eps) + (1 - y) * np.log(1 - p_xg_only + eps))
    )

    return {
        "brier_model": brier_model,
        "brier_xg_only": brier_xg,
        "logloss_model": logloss_model,
        "logloss_xg_only": logloss_xg,
        "n_shots": len(y),
        "season_idx": season_idx,
    }
