"""
fit.py
------
MAP optimisation routines for DynamicIRTModel.

Public API
----------
fit_full_map(data, config)                    -> ModelState
update_map_with_new_week(state, new_df, config) -> ModelState
save_model_state(state, path)
load_model_state(path)                        -> ModelState
"""

from __future__ import annotations

import logging
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from .config import ModelConfig
from .data_prep import ModelData, extend_model_data
from .irt_model import DynamicIRTModel

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ModelState — the serialisable artefact produced by every fit
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelState:
    """
    Serialisable snapshot of a trained DynamicIRTModel.

    Attributes
    ----------
    config : ModelConfig
        The config used for this fit (preserved for reproducibility).
    param_dict : dict[str, np.ndarray]
        All model parameters as numpy arrays (from model.state_dict_numpy()).
    shooter_id_to_idx, goalie_id_to_idx : dicts
        Stable player → integer-index mappings.
    idx_to_shooter_id, idx_to_goalie_id : lists
        Reverse mappings.
    weeks_per_season : list[int]
        Number of weeks per season at the time of the fit.
    season_labels : list[int]
        The raw season-year integers (e.g. [2023, 2024]).
    shots_df : pd.DataFrame
        The cleaned, indexed shot history used to build the last fit.
        Stored so that extend_model_data() can append without re-cleaning all
        historical data.
    fit_metadata : dict
        Optional diagnostics: runtime, final loss, iteration count, etc.
    """
    config: ModelConfig
    param_dict: Dict[str, np.ndarray]
    shooter_id_to_idx: Dict[str, int]
    goalie_id_to_idx: Dict[str, int]
    idx_to_shooter_id: List[str]
    idx_to_goalie_id: List[str]
    weeks_per_season: List[int]
    season_labels: List[int]
    shots_df: pd.DataFrame
    fit_metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────

def save_model_state(state: ModelState, path: str | Path) -> None:
    """
    Pickle a ModelState to disk.

    The file is saved atomically (written to a .tmp file first) so a crash
    during saving does not corrupt the previous state.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    log.info("Model state saved → %s", path)


def load_model_state(path: str | Path) -> ModelState:
    """Load a pickled ModelState from disk."""
    with open(path, "rb") as f:
        state = pickle.load(f)
    log.info("Model state loaded ← %s", path)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_from_logistic(model: DynamicIRTModel, data: ModelData) -> None:
    """
    Warm-initialise β₀ and α from a simple logistic regression of
    is_goal ~ xg_logit.  All skill parameters remain at 0.
    """
    try:
        from sklearn.linear_model import LogisticRegression  # optional

        X = data.xg_logit.numpy().reshape(-1, 1)
        y = data.y.numpy()
        lr = LogisticRegression(
            C=1e6, solver="lbfgs", max_iter=200, random_state=0
        )
        lr.fit(X, y)
        model.alpha.data.fill_(float(lr.coef_[0, 0]))
        model.beta0.data.fill_(float(lr.intercept_[0]))
        log.info(
            "Logistic init: β₀=%.4f, α=%.4f",
            float(model.beta0),
            float(model.alpha),
        )
    except ImportError:
        log.warning(
            "scikit-learn not found; skipping logistic initialisation. "
            "Install scikit-learn for better convergence."
        )


def _zero_frozen_grads(
    model: DynamicIRTModel,
    data: ModelData,
    freeze_prev_season: bool = True,
    freeze_early_current: bool = True,
) -> None:
    """
    After loss.backward(), zero out gradients for parameters that should
    remain frozen during a weekly update.

    Freezing strategy
    -----------------
    * Previous season (s=0):  all θ and φ parameters.
    * Current season  (s=1):  weeks 0 … T_curr - K - 1  (first T_curr - K weeks).
    """
    cfg = model.config

    if model.theta.grad is None:
        return

    if freeze_prev_season and data.n_seasons > 1:
        model.theta.grad[:, 0, :] = 0.0
        model.phi.grad[:, 0, :]   = 0.0

    if freeze_early_current:
        T_curr = data.weeks_per_season[1] if len(data.weeks_per_season) > 1 else data.weeks_per_season[0]
        cutoff = max(0, T_curr - cfg.update_window_k)
        if cutoff > 0:
            model.theta.grad[:, -1, :cutoff] = 0.0
            model.phi.grad[:, -1, :cutoff]   = 0.0


def _make_batch_indices(
    n_shots: int, batch_size: Optional[int]
) -> Optional[torch.Tensor]:
    """Return a random mini-batch of indices or None for full-batch."""
    if batch_size is None or batch_size >= n_shots:
        return None
    return torch.randperm(n_shots)[:batch_size]


def _build_model_state(
    model: DynamicIRTModel,
    data: ModelData,
    config: ModelConfig,
    metadata: dict,
) -> ModelState:
    return ModelState(
        config=config,
        param_dict=model.state_dict_numpy(),
        shooter_id_to_idx=data.shooter_id_to_idx,
        goalie_id_to_idx=data.goalie_id_to_idx,
        idx_to_shooter_id=data.idx_to_shooter_id,
        idx_to_goalie_id=data.idx_to_goalie_id,
        weeks_per_season=list(data.weeks_per_season),
        season_labels=list(data.season_labels),
        shots_df=data.shots_df.copy(),
        fit_metadata=metadata,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core optimisation loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_adam(
    model: DynamicIRTModel,
    data: ModelData,
    config: ModelConfig,
    max_epochs: int,
    freeze_grads: bool = False,
) -> Dict[str, Any]:
    """
    Run Adam optimisation with early stopping.

    Returns a metadata dict with: final_loss, n_iters, runtime_seconds.
    """
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=config.lr * 0.01
    )

    best_loss = float("inf")
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()

        idx = _make_batch_indices(data.n_shots, config.batch_size)
        loss = model.loss(data, idx)
        loss.backward()

        if freeze_grads:
            _zero_frozen_grads(model, data)

        optimizer.step()
        scheduler.step()

        loss_val = loss.item()

        if loss_val < best_loss - 1e-4:
            best_loss = loss_val
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 100 == 0 or epoch == 1:
            log.info(
                "  epoch %4d/%d  loss=%.4f  best=%.4f  no_improve=%d",
                epoch, max_epochs, loss_val, best_loss, no_improve,
            )

        if no_improve >= config.patience:
            log.info("  Early stop at epoch %d (patience=%d)", epoch, config.patience)
            break

    runtime = time.time() - t0
    return {"final_loss": best_loss, "n_iters_adam": epoch, "runtime_adam": runtime}


def _run_lbfgs(
    model: DynamicIRTModel,
    data: ModelData,
    config: ModelConfig,
) -> Dict[str, Any]:
    """
    Optional L-BFGS refinement step after Adam.
    Saves model state before running; restores it if L-BFGS produces NaN/Inf.
    """
    if config.lbfgs_steps == 0:
        return {"n_iters_lbfgs": 0}

    # Snapshot good Adam parameters so we can roll back if L-BFGS diverges
    pre_lbfgs_state = {k: v.clone() for k, v in model.state_dict().items()}

    log.info("  Running %d L-BFGS steps …", config.lbfgs_steps)
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=config.lbfgs_steps,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    final_loss = [None]

    def closure():
        optimizer.zero_grad()
        loss = model.loss(data)
        loss.backward()
        final_loss[0] = loss.item()
        return loss

    optimizer.step(closure)

    loss_val = final_loss[0]
    if loss_val is None or math.isnan(loss_val) or math.isinf(loss_val):
        log.warning(
            "  L-BFGS produced NaN/Inf loss (%.4s) — restoring Adam best state.",
            loss_val,
        )
        model.load_state_dict(pre_lbfgs_state)
        return {"n_iters_lbfgs": config.lbfgs_steps, "final_loss_lbfgs": float("nan"), "lbfgs_nan": True}

    log.info("  L-BFGS done. loss=%.4f", loss_val)
    return {"n_iters_lbfgs": config.lbfgs_steps, "final_loss_lbfgs": loss_val}


# ─────────────────────────────────────────────────────────────────────────────
# Public: full fit
# ─────────────────────────────────────────────────────────────────────────────

def fit_full_map(data: ModelData, config: ModelConfig) -> ModelState:
    """
    Fit the dynamic IRT model from scratch on all shots in ``data``.

    Steps
    -----
    1. Initialise β₀ and α from logistic regression on xg_logit → is_goal.
    2. Adam optimisation with cosine LR schedule and early stopping.
    3. Optional L-BFGS refinement.
    4. Return ModelState (includes parameter dict, index maps, shot history).

    Parameters
    ----------
    data : ModelData
        Prepared shot data from ``prepare_shot_data()``.
    config : ModelConfig

    Returns
    -------
    ModelState
    """
    log.info(
        "fit_full_map: %d shots, %d shooters, %d goalies, %s",
        data.n_shots, data.n_shooters, data.n_goalies, data.weeks_per_season,
    )
    t_total = time.time()

    model = DynamicIRTModel(data, config)
    model.train()

    _init_from_logistic(model, data)

    meta: Dict[str, Any] = {}
    meta.update(_run_adam(model, data, config, config.max_epochs_full, freeze_grads=False))
    meta.update(_run_lbfgs(model, data, config))

    meta["runtime_total"] = time.time() - t_total
    log.info(
        "fit_full_map done in %.1f s  (Adam %d iters, L-BFGS %d steps)",
        meta["runtime_total"],
        meta.get("n_iters_adam", 0),
        meta.get("n_iters_lbfgs", 0),
    )

    return _build_model_state(model, data, config, meta)


# ─────────────────────────────────────────────────────────────────────────────
# Public: weekly warm-start update
# ─────────────────────────────────────────────────────────────────────────────

def update_map_with_new_week(
    state: ModelState,
    new_df: pd.DataFrame,
    config: Optional[ModelConfig] = None,
) -> ModelState:
    """
    Append newly-ingested shots and warm-start optimise the model.

    Strategy
    --------
    * Rebuilds ModelData by appending new shots to the stored shot history.
    * Preserves all existing player-index assignments.
    * Expands model tensors if new players or weeks have appeared.
    * Loads the previous MAP parameters as the warm start.
    * Re-optimises for ``config.max_epochs_update`` iterations with:
        - Previous season parameters frozen.
        - Only the last ``config.update_window_k`` weeks of the current season
          allowed to change (plus all global parameters and baselines).

    Parameters
    ----------
    state : ModelState
        Output of the most-recent fit_full_map() or update_map_with_new_week().
    new_df : pd.DataFrame
        Raw or pre-cleaned shot frame containing *only the new shots*.
    config : ModelConfig, optional
        If None, reuses state.config.

    Returns
    -------
    ModelState
        Updated state including the new shots in shots_df.
    """
    if config is None:
        config = state.config

    log.info("update_map_with_new_week: %d new shots", len(new_df))
    t_total = time.time()

    # ── Rebuild ModelData from stored history + new shots ─────────────────────
    prev_data = _rebuild_model_data_from_state(state, config)
    new_data = extend_model_data(prev_data, new_df, config)

    log.info(
        "  Extended data: %d → %d shots, weeks_per_season: %s → %s",
        prev_data.n_shots,
        new_data.n_shots,
        prev_data.weeks_per_season,
        new_data.weeks_per_season,
    )

    # ── Build new model and warm-start ────────────────────────────────────────
    model = DynamicIRTModel(new_data, config)

    # Copy previous parameters (expands if shape changed)
    model.expand_for_new_data(new_data)
    model.load_state_dict_numpy(state.param_dict)
    # After loading, re-expand in case new weeks were added
    model.expand_for_new_data(new_data)
    model.train()

    # ── Warm-start Adam with gradient freezing ────────────────────────────────
    meta: Dict[str, Any] = {}
    meta.update(
        _run_adam(
            model, new_data, config,
            max_epochs=config.max_epochs_update,
            freeze_grads=True,
        )
    )
    meta.update(_run_lbfgs(model, new_data, config))

    meta["runtime_total"] = time.time() - t_total
    log.info(
        "update_map_with_new_week done in %.1f s  (%d shots processed)",
        meta["runtime_total"],
        new_data.n_shots,
    )

    return _build_model_state(model, new_data, config, meta)


# ─────────────────────────────────────────────────────────────────────────────
# Internal: reconstruct a minimal ModelData from a saved state
# ─────────────────────────────────────────────────────────────────────────────

def _rebuild_model_data_from_state(
    state: ModelState, config: ModelConfig
) -> ModelData:
    """
    Reconstruct a ModelData object from the stored shots_df and mappings in
    ModelState, without running _clean_raw() again.

    This rebuilds the tensors from the already-indexed shots_df that was saved
    in the state, so index assignments are guaranteed to match param_dict.
    """
    from .data_prep import ModelData, _safe_logit
    import torch

    df = state.shots_df.copy()

    xg_logit_np = _safe_logit(df["xg"].values)

    return ModelData(
        xg_logit=torch.tensor(xg_logit_np, dtype=torch.float32),
        y=torch.tensor(df["is_goal"].values, dtype=torch.float32),
        shooter_idx=torch.tensor(df["shooter_idx"].values, dtype=torch.long),
        goalie_idx=torch.tensor(df["goalie_idx"].values, dtype=torch.long),
        season_idx=torch.tensor(df["season_idx"].values, dtype=torch.long),
        week_idx=torch.tensor(df["week_idx"].values, dtype=torch.long),
        n_shooters=len(state.shooter_id_to_idx),
        n_goalies=len(state.goalie_id_to_idx),
        n_seasons=2,
        weeks_per_season=list(state.weeks_per_season),
        shooter_id_to_idx=state.shooter_id_to_idx,
        goalie_id_to_idx=state.goalie_id_to_idx,
        idx_to_shooter_id=state.idx_to_shooter_id,
        idx_to_goalie_id=state.idx_to_goalie_id,
        season_labels=state.season_labels,
        shots_df=df,
    )
