"""Quick smoke test for the goalie_model package."""
import sys, warnings
sys.path.insert(0, r'c:\Users\Sean\Documents\Projects\hockey_proj_py')
warnings.filterwarnings('ignore')

import pandas as pd
from goalie_model import (
    ModelConfig, prepare_shot_data, fit_full_map,
    summarize_skills, compute_gsax, evaluate_predictions,
)

raw = pd.read_csv(
    r'c:\Users\Sean\Documents\Projects\hockey_proj_py\ingest_scripts\nhl_pbp_2024_2025_with_xg.csv',
    nrows=2000,
)
print(f'Raw rows: {len(raw)}')

cfg = ModelConfig(
    seasons=[2023, 2024],   # prev=2023-24 (no data), curr=2024-25
    require_prev_season=False,
    max_epochs_full=100,
    lbfgs_steps=0,
    patience=30,
)

data = prepare_shot_data(raw, cfg)
print(f'ModelData: {data}')

state = fit_full_map(data, cfg)
final_loss = state.fit_metadata['final_loss']
print(f'Fit done. Loss: {final_loss:.4f}')

shooter_df, goalie_df = summarize_skills(state)
print(f'Shooters: {len(shooter_df)} rows, Goalies: {len(goalie_df)} rows')

gsax = compute_gsax(state)
print(f'GSAx table: {len(gsax)} goalies')
print(gsax.head(3).to_string())

diag = evaluate_predictions(state)
brier_m = diag['brier_model']
brier_x = diag['brier_xg_only']
print(f'Brier model={brier_m:.5f}  xG-only={brier_x:.5f}')
print('SMOKE TEST PASSED')
