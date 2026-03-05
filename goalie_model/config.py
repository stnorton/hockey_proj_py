"""
config.py
---------
ModelConfig dataclass for the Dynamic Goalie-Shooter IRT model.

All hyper-parameters, data-filtering options, and optimization settings
live here so that every other module only needs to accept a ModelConfig.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelConfig:
    """
    Full configuration for the Dynamic Goalie-Shooter IRT model.

    Attributes
    ----------
    seasons : list[int]
        Exactly two integer season-start-year labels in chronological order,
        e.g. [2023, 2024].  The first is treated as the *previous* season
        (S=0 internally) and the second as the *current* season (S=1).
        If only one season of data is available, supply it as the second
        element and fill the first with None / a dummy value and set
        ``require_prev_season=False``.
    require_prev_season : bool
        If False, the model silently ignores the first season slot when no
        data is found for it.  Useful for bootstrapping on a single season.
    strength_states : list[str] or None
        Strength-state whitelist.  None keeps every shot.
        Common choices: ['EVEN'], ['EVEN','PP','PK'].
        Must match the values produced by decode_strength() in data_prep.

    week_bin_method : str
        'iso'    — ISO-8601 calendar week (Monday-Sunday).
        'custom' — 7-day bins anchored to the first game date of the season.

    prior_sigma_mu : float
        Std dev of Normal prior on career-baseline skills (μ_θ, μ_φ).
    prior_sigma_season : float
        Std dev of Normal prior on season-start deviation from baseline.
    prior_tau : float
        Std dev of the weekly random-walk step for both θ and φ.
    prior_sigma_beta0 : float
        Std dev of weakly-informative Normal prior on β₀.
    prior_sigma_alpha : float
        Std dev of Normal prior on α (xG calibration coefficient).
    prior_mean_alpha : float
        Mean of the Normal prior on α.  Default 1.0 (trust pre-fitted xG).
    sum_to_zero_weight : float
        Weight on the soft sum-to-zero penalty (∑_p θ[p,s,t])² applied per
        (season, week).  0 disables it; ~0.5 gives gentle centering.

    lr : float
        Adam learning rate for both full fits and weekly updates.
    max_epochs_full : int
        Maximum Adam iterations for a full two-season fit from scratch.
    max_epochs_update : int
        Maximum Adam iterations for a weekly warm-start update.
    patience : int
        Early-stopping: stop if no improvement for this many iterations.
    batch_size : int or None
        Mini-batch size for the Bernoulli log-likelihood term.
        None = use full data every step (fine for ≤100 k shots).
    update_window_k : int
        During weekly updates, freeze all parameters belonging to weeks
        earlier than (T_current - k) in the current season.  Previous-season
        parameters are always frozen during updates.
    lbfgs_steps : int
        L-BFGS refinement steps applied after Adam converges (0 = skip).

    center_weekly : bool
        Subtract the weekly cross-player mean from θ and φ before
        reporting (so that 0 = league average for that week).
    output_dir : str
        Directory where model state files and summaries are written.
    """

    # ── Data / Filtering ─────────────────────────────────────────────────────
    seasons: List[int] = field(default_factory=lambda: [2023, 2024])
    require_prev_season: bool = False

    strength_states: Optional[List[str]] = None  # None  = all states

    # ── Time Binning ─────────────────────────────────────────────────────────
    week_bin_method: str = "iso"  # "iso" | "custom"

    # ── Prior Scales ─────────────────────────────────────────────────────────
    prior_sigma_mu: float = 1.0
    prior_sigma_season: float = 0.5
    prior_tau: float = 0.2
    prior_sigma_beta0: float = 10.0
    prior_sigma_alpha: float = 0.5
    prior_mean_alpha: float = 1.0
    sum_to_zero_weight: float = 0.5

    # ── Variance lower bounds (prevent MAP collapse to zero) ──────────────────
    tau_min: float = 0.01           # floor for learned random-walk step SD
    sigma_season_min: float = 0.05  # floor for learned season-start deviation SD

    # ── Optimization ─────────────────────────────────────────────────────────
    lr: float = 1e-2
    max_epochs_full: int = 5000
    max_epochs_update: int = 300
    patience: int = 150
    batch_size: Optional[int] = None
    update_window_k: int = 8
    lbfgs_steps: int = 20

    # ── Reporting ────────────────────────────────────────────────────────────
    center_weekly: bool = True
    output_dir: str = "model_output"

    # ── Serialization helpers ─────────────────────────────────────────────────
    def to_json(self, path: str | Path) -> None:
        """Write config to a JSON file."""
        Path(path).write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        """Load config from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def __post_init__(self) -> None:
        if len(self.seasons) != 2:
            raise ValueError(
                f"'seasons' must contain exactly 2 elements (prev, current). "
                f"Got: {self.seasons}"
            )
        if self.week_bin_method not in ("iso", "custom"):
            raise ValueError(
                f"week_bin_method must be 'iso' or 'custom'. Got: {self.week_bin_method}"
            )
