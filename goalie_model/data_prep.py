"""
data_prep.py
------------
Data loading, cleaning, indexing, and tensor construction for the
Dynamic Goalie-Shooter IRT model.

Public entry points
-------------------
prepare_shot_data(raw_df, config)  ->  ModelData
    Full pipeline from raw NHL PBP data to model-ready tensors.

extend_model_data(existing: ModelData, new_df, config)  ->  ModelData
    Append new shots while preserving existing player-index mappings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ──────────────────────────────────────────────────────────────────────────────
# ModelData container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelData:
    """
    All tensors and metadata required by DynamicIRTModel.

    Tensor shapes (N = number of shots)
    ------------------------------------
    xg_logit      float32  [N]  — logit-transformed xG probability
    y             float32  [N]  — binary outcome (1=goal, 0=save)
    shooter_idx   int64    [N]  — 0-based shooter index
    goalie_idx    int64    [N]  — 0-based goalie index
    season_idx    int64    [N]  — 0-based season index (0=prev, 1=curr)
    week_idx      int64    [N]  — 0-based week index *within* that season

    Scalars / lists
    ---------------
    n_shooters       int
    n_goalies        int
    n_seasons        int  (always 2)
    weeks_per_season list[int]  — [T_0, T_1]; number of weeks in each season

    Mappings
    --------
    shooter_id_to_idx  dict[str, int]
    goalie_id_to_idx   dict[str, int]
    idx_to_shooter_id  list[str]
    idx_to_goalie_id   list[str]
    season_labels      list[int]   — the two raw season-year integers

    Raw data (stored for warm-start updates)
    -----------------------------------------
    shots_df : pd.DataFrame  — cleaned shot-level frame used to build tensors
    """

    # ── Tensors ───────────────────────────────────────────────────────────────
    xg_logit: torch.Tensor
    y: torch.Tensor
    shooter_idx: torch.Tensor
    goalie_idx: torch.Tensor
    season_idx: torch.Tensor
    week_idx: torch.Tensor

    # ── Shape info ────────────────────────────────────────────────────────────
    n_shooters: int
    n_goalies: int
    n_seasons: int
    weeks_per_season: List[int]

    # ── Player mappings ───────────────────────────────────────────────────────
    shooter_id_to_idx: Dict[str, int]
    goalie_id_to_idx: Dict[str, int]
    idx_to_shooter_id: List[str]
    idx_to_goalie_id: List[str]
    season_labels: List[int]

    # ── Raw cleaned frame (for updates) ───────────────────────────────────────
    shots_df: pd.DataFrame = field(repr=False)

    @property
    def max_weeks(self) -> int:
        return max(self.weeks_per_season)

    @property
    def n_shots(self) -> int:
        return int(self.y.shape[0])

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ModelData(shots={self.n_shots:,}, shooters={self.n_shooters}, "
            f"goalies={self.n_goalies}, seasons={self.season_labels}, "
            f"weeks_per_season={self.weeks_per_season})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Numerically-stable logit transform."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _decode_strength(code) -> str:
    """
    Convert NHL situation_code integer (e.g. 1551, 1541, 1451) to a
    high-level strength label: EVEN, PP, PK, or OTHER.
    """
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "EVEN"
    away = (code // 100) % 10
    home = (code // 10) % 10
    if away == home:
        return "EVEN"
    elif away > home:
        return "PP"
    elif away < home:
        return "PK"
    return "OTHER"


def _assign_season(game_id: int, seasons: List[int]) -> Optional[int]:
    """
    Return 0-based season index (0=prev, 1=curr) from a numeric game_id.

    NHL game IDs encode the season start year in the leading 4 digits:
        2024 01 0001  →  2024-25 season  →  season_year = 2024
    """
    try:
        year = int(game_id) // 1_000_000
    except (ValueError, TypeError):
        return None
    if year == seasons[0]:
        return 0
    if year == seasons[1]:
        return 1
    return None


def _build_week_index(
    dates: pd.Series,
    method: str = "iso",
    season_start: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    Map a Series of dates to 0-based week indices within a season.

    Parameters
    ----------
    dates : Series of datetime-like
    method : 'iso' or 'custom'
    season_start : required when method='custom'
    """
    dt = pd.to_datetime(dates, errors="coerce")
    if method == "iso":
        # ISO week numbers can restart across year-boundaries (week 52/53→week 1)
        # so we compute (year-week) tuples and rank them to get contiguous indices.
        yw = dt.dt.isocalendar()[["year", "week"]].apply(
            lambda r: r["year"] * 100 + r["week"], axis=1
        )
        unique_sorted = sorted(yw.dropna().unique())
        yw_to_idx = {v: i for i, v in enumerate(unique_sorted)}
        return yw.map(yw_to_idx).fillna(-1).astype(int)
    elif method == "custom":
        if season_start is None:
            season_start = dt.min()
        days_since = (dt - season_start).dt.days.clip(lower=0)
        return (days_since // 7).fillna(-1).astype(int)
    else:
        raise ValueError(f"Unknown week_bin_method: '{method}'")


# ──────────────────────────────────────────────────────────────────────────────
# Core cleaning step (used by both prepare and extend)
# ──────────────────────────────────────────────────────────────────────────────

def _clean_raw(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Apply column-level cleaning to a raw NHL PBP frame:
      - Derive shooter_id (shooting or scoring player)
      - Derive goalie_id
      - Derive is_goal
      - Derive strength_state
      - Parse game_date
      - Apply strength-state filter

    Returns a new DataFrame with standardised column names.
    """
    df = df.copy()

    # ── Shooter ID ────────────────────────────────────────────────────────────
    if "shooter_id" not in df.columns:
        shooting = df.get("details.shootingPlayerId")
        scoring = df.get("details.scoringPlayerId")
        if shooting is not None and scoring is not None:
            df["shooter_id"] = shooting.fillna(scoring)
        elif shooting is not None:
            df["shooter_id"] = shooting
        elif scoring is not None:
            df["shooter_id"] = scoring
        else:
            raise KeyError(
                "Cannot find shooter columns. Expected 'details.shootingPlayerId' "
                "or 'details.scoringPlayerId' in raw data."
            )

    # ── Goalie ID ─────────────────────────────────────────────────────────────
    if "goalie_id" not in df.columns:
        if "details.goalieInNetId" in df.columns:
            df["goalie_id"] = df["details.goalieInNetId"]
        else:
            raise KeyError(
                "Cannot find goalie column. Expected 'details.goalieInNetId' "
                "or a pre-built 'goalie_id' column."
            )

    # ── Goal outcome ──────────────────────────────────────────────────────────
    if "is_goal" not in df.columns:
        if "shot_made" in df.columns:
            df["is_goal"] = df["shot_made"]
        else:
            raise KeyError("Cannot find outcome column. Expected 'shot_made' or 'is_goal'.")

    # ── xG ────────────────────────────────────────────────────────────────────
    if "xg" not in df.columns:
        if "xG" in df.columns:
            df["xg"] = df["xG"]
        else:
            raise KeyError("Cannot find xG column. Expected 'xG' or 'xg'.")

    # ── Strength state ────────────────────────────────────────────────────────
    if "strength_state" not in df.columns:
        if "situation_code" in df.columns:
            df["strength_state"] = df["situation_code"].apply(_decode_strength)
        else:
            df["strength_state"] = "EVEN"

    # ── Game date ─────────────────────────────────────────────────────────────
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    else:
        df["game_date"] = pd.NaT

    # ── Coerce IDs to string ──────────────────────────────────────────────────
    df["shooter_id"] = df["shooter_id"].astype(str)
    df["goalie_id"] = df["goalie_id"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    df["is_goal"] = df["is_goal"].astype(float)
    df["xg"] = df["xg"].astype(float)

    # ── Drop rows with missing required fields ────────────────────────────────
    required = ["shooter_id", "goalie_id", "xg", "is_goal", "game_id"]
    df = df.dropna(subset=required)
    df = df[
        ~df["shooter_id"].isin(["nan", "None", ""])
        & ~df["goalie_id"].isin(["nan", "None", ""])
    ]

    # ── Shootout filter ───────────────────────────────────────────────────────
    # Exclude shootout events — they are never counted in official GA/GF stats.
    if "periodDescriptor.periodType" in df.columns:
        df = df[df["periodDescriptor.periodType"] != "SO"]
    elif "period_number" in df.columns:
        df = df[df["period_number"] != 5]

    # ── Strength-state filter ─────────────────────────────────────────────────
    if config.strength_states is not None:
        df = df[df["strength_state"].isin(config.strength_states)]

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Construct integer indices within a cleaned frame
# ──────────────────────────────────────────────────────────────────────────────

def _build_indices(
    df: pd.DataFrame,
    config,
    shooter_id_to_idx: Optional[Dict[str, int]] = None,
    goalie_id_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, Dict, Dict, List, List]:
    """
    Add season_idx, week_idx, shooter_idx, goalie_idx columns to df.

    If player-index dictionaries are supplied (for warm-start updates) new
    players are appended at the end of the existing mapping.

    Returns
    -------
    df : annotated DataFrame
    shooter_id_to_idx, goalie_id_to_idx : updated dicts
    idx_to_shooter, idx_to_goalie : lists (index → id)
    """
    seasons = config.seasons

    # ── Season index ──────────────────────────────────────────────────────────
    def get_season_idx(gid):
        return _assign_season(gid, seasons)

    df["season_idx"] = df["game_id"].apply(get_season_idx)
    df = df[df["season_idx"].notna()].copy()
    df["season_idx"] = df["season_idx"].astype(int)

    # ── Week index (within each season) ───────────────────────────────────────
    week_idx_col = np.full(len(df), -1, dtype=int)
    for s_idx in range(len(seasons)):
        mask = df["season_idx"] == s_idx
        if mask.sum() == 0:
            continue
        sub = df.loc[mask, "game_date"]
        if config.week_bin_method == "custom":
            season_start = sub.min()
        else:
            season_start = None
        w = _build_week_index(sub, method=config.week_bin_method, season_start=season_start)
        week_idx_col[mask.values] = w.values

    df["week_idx"] = week_idx_col
    df = df[df["week_idx"] >= 0].reset_index(drop=True)

    # ── Player index mappings ─────────────────────────────────────────────────
    if shooter_id_to_idx is None:
        shooter_id_to_idx = {}
    if goalie_id_to_idx is None:
        goalie_id_to_idx = {}

    def _maybe_extend(id_series: pd.Series, mapping: Dict[str, int]) -> None:
        for pid in id_series.unique():
            if pid not in mapping:
                mapping[pid] = len(mapping)

    _maybe_extend(df["shooter_id"], shooter_id_to_idx)
    _maybe_extend(df["goalie_id"], goalie_id_to_idx)

    idx_to_shooter = [None] * len(shooter_id_to_idx)
    for pid, idx in shooter_id_to_idx.items():
        idx_to_shooter[idx] = pid

    idx_to_goalie = [None] * len(goalie_id_to_idx)
    for pid, idx in goalie_id_to_idx.items():
        idx_to_goalie[idx] = pid

    df["shooter_idx"] = df["shooter_id"].map(shooter_id_to_idx)
    df["goalie_idx"] = df["goalie_id"].map(goalie_id_to_idx)

    return df, shooter_id_to_idx, goalie_id_to_idx, idx_to_shooter, idx_to_goalie


# ──────────────────────────────────────────────────────────────────────────────
# Assemble ModelData from an annotated DataFrame
# ──────────────────────────────────────────────────────────────────────────────

def _assemble_model_data(
    df: pd.DataFrame,
    shooter_id_to_idx: Dict[str, int],
    goalie_id_to_idx: Dict[str, int],
    idx_to_shooter: List[str],
    idx_to_goalie: List[str],
    season_labels: List[int],
) -> ModelData:
    """Build ModelData tensors from an annotated DataFrame."""
    n_seasons = 2

    # Weeks per season (how many distinct week-indices exist for each season)
    weeks_per_season = []
    for s in range(n_seasons):
        mask = df["season_idx"] == s
        if mask.sum() == 0:
            weeks_per_season.append(0)
        else:
            weeks_per_season.append(int(df.loc[mask, "week_idx"].max()) + 1)

    xg_logit_np = _safe_logit(df["xg"].values)

    return ModelData(
        xg_logit=torch.tensor(xg_logit_np, dtype=torch.float32),
        y=torch.tensor(df["is_goal"].values, dtype=torch.float32),
        shooter_idx=torch.tensor(df["shooter_idx"].values, dtype=torch.long),
        goalie_idx=torch.tensor(df["goalie_idx"].values, dtype=torch.long),
        season_idx=torch.tensor(df["season_idx"].values, dtype=torch.long),
        week_idx=torch.tensor(df["week_idx"].values, dtype=torch.long),
        n_shooters=len(shooter_id_to_idx),
        n_goalies=len(goalie_id_to_idx),
        n_seasons=n_seasons,
        weeks_per_season=weeks_per_season,
        shooter_id_to_idx=shooter_id_to_idx,
        goalie_id_to_idx=goalie_id_to_idx,
        idx_to_shooter_id=idx_to_shooter,
        idx_to_goalie_id=idx_to_goalie,
        season_labels=season_labels,
        shots_df=df.reset_index(drop=True),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def prepare_shot_data(raw_df: pd.DataFrame, config) -> ModelData:
    """
    Full pipeline: raw NHL PBP DataFrame  →  ModelData.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw play-by-play frame with xG already computed.  Must contain
        (at minimum) the columns described in §2.1 of model_requirements.md.
    config : ModelConfig

    Returns
    -------
    ModelData
    """
    df = _clean_raw(raw_df, config)
    df, s2i, g2i, is_list, ig_list = _build_indices(df, config)
    return _assemble_model_data(df, s2i, g2i, is_list, ig_list, config.seasons)


def extend_model_data(
    existing: ModelData,
    new_df: pd.DataFrame,
    config,
) -> ModelData:
    """
    Append newly-ingested shots to an existing ModelData while preserving all
    previously assigned player-index mappings.

    Use this for weekly updates so that player indices remain stable between
    model fits (warm-starting requires consistent index ordering).

    Parameters
    ----------
    existing : ModelData
        The ModelData object produced by the most recent fit.
    new_df : pd.DataFrame
        Raw (or pre-cleaned) shot frame containing *only the new shots*.
    config : ModelConfig

    Returns
    -------
    ModelData
        Combined ModelData covering all shots (old + new) with stable indices.
    """
    cleaned_new = _clean_raw(new_df, config)

    # Combine with the previously cleaned frame
    combined = pd.concat([existing.shots_df, cleaned_new], ignore_index=True)

    # Re-index using the *existing* mappings so old players keep their indices
    _, s2i, g2i, is_list, ig_list = _build_indices(
        combined,
        config,
        shooter_id_to_idx=dict(existing.shooter_id_to_idx),
        goalie_id_to_idx=dict(existing.goalie_id_to_idx),
    )
    return _assemble_model_data(
        combined, s2i, g2i, is_list, ig_list, config.seasons
    )
