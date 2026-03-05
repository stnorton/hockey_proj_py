"""
goalie_model
============
Dynamic Goalie-Shooter IRT model with weekly MAP updates.

Quick-start
-----------
>>> import pandas as pd
>>> from goalie_model import ModelConfig, prepare_shot_data, fit_full_map
>>> from goalie_model import save_model_state, load_model_state
>>> from goalie_model import summarize_skills, compute_gsax, evaluate_predictions

>>> # 1. Configure
>>> cfg = ModelConfig(seasons=[2023, 2024])

>>> # 2. Load and prepare data
>>> raw = pd.read_csv("ingest_scripts/nhl_pbp_2024_2025_with_xg.csv")
>>> data = prepare_shot_data(raw, cfg)

>>> # 3. Fit
>>> state = fit_full_map(data, cfg)
>>> save_model_state(state, "model_output/state_latest.pkl")

>>> # 4. Weekly update
>>> from goalie_model import update_map_with_new_week
>>> new_shots = pd.read_csv("new_week.csv")
>>> state = update_map_with_new_week(state, new_shots)
>>> save_model_state(state, "model_output/state_latest.pkl")

>>> # 5. Summarise
>>> shooter_df, goalie_df = summarize_skills(state)
>>> gsax_df = compute_gsax(state)
"""

from .config import ModelConfig
from .data_prep import ModelData, prepare_shot_data, extend_model_data
from .fit import (
    ModelState,
    fit_full_map,
    update_map_with_new_week,
    save_model_state,
    load_model_state,
)
from .irt_model import DynamicIRTModel
from .summarize import (
    summarize_skills,
    compute_gsax,
    skill_trajectory,
    evaluate_predictions,
    fetch_player_names,
)

__all__ = [
    # Config
    "ModelConfig",
    # Data
    "ModelData",
    "prepare_shot_data",
    "extend_model_data",
    # Model
    "DynamicIRTModel",
    # Fit
    "ModelState",
    "fit_full_map",
    "update_map_with_new_week",
    "save_model_state",
    "load_model_state",
    # Summarise
    "summarize_skills",
    "compute_gsax",
    "skill_trajectory",
    "evaluate_predictions",
    "fetch_player_names",
]
