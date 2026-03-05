"""
run_model.py
------------
Entry-point for fitting (or updating) the Dynamic Goalie-Shooter IRT model.

Usage
-----
# Full two-season fit from scratch:
python run_model.py

# Weekly warm-start update with new shots:
python run_model.py --update --new-csv ingest_scripts/nhl_pbp_2025_2026_with_xg.csv

# Override default CSV paths or output dir:
python run_model.py --prev ingest_scripts/nhl_pbp_2024_2025_with_xg.csv \
                   --curr ingest_scripts/nhl_pbp_2025_2026_with_xg.csv \
                   --out  model_output
"""
import argparse
import logging
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from goalie_model import (
    ModelConfig,
    prepare_shot_data,
    fit_full_map,
    update_map_with_new_week,
    save_model_state,
    load_model_state,
    summarize_skills,
    compute_gsax,
    evaluate_predictions,
)
from goalie_model.summarize import fetch_player_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Strength-state configurations ──────────────────────────────────────────────────────
SITUATION_STRENGTH: dict = {
    "all":  None,       # all situations (5v5, PP, PK, 4v4, etc.)
    "even": ["EVEN"],   # even strength (5v5, 4v4, 3v3) only
    "pp":   ["PP"],     # power play only
}

DEFAULT_PREV_CSV = os.path.join(ROOT, "ingest_scripts", "nhl_pbp_2024_2025_with_xg.csv")
DEFAULT_CURR_CSV = os.path.join(ROOT, "ingest_scripts", "nhl_pbp_2025_2026_with_xg.csv")
DEFAULT_STATE    = os.path.join(ROOT, "model_output", "state_latest.pkl")
DEFAULT_OUT_DIR  = os.path.join(ROOT, "model_output")


def parse_args():
    p = argparse.ArgumentParser(description="Fit or update the Dynamic IRT model.")
    p.add_argument("--prev",    default=DEFAULT_PREV_CSV,
                   help="Path to previous-season xG CSV (2024-25).")
    p.add_argument("--curr",    default=DEFAULT_CURR_CSV,
                   help="Path to current-season xG CSV (2025-26).")
    p.add_argument("--out",     default=DEFAULT_OUT_DIR,
                   help="Directory to write model state and summaries.")
    p.add_argument("--update",  action="store_true",
                   help="Warm-start update from an existing state. "
                        "Requires --new-csv and a saved state at <out>/state_latest.pkl.")
    p.add_argument("--new-csv", default=None,
                   help="CSV of new shots (current season only) for --update mode.")
    p.add_argument("--epochs-full",   type=int, default=5000)
    p.add_argument("--epochs-update", type=int, default=300)
    p.add_argument("--lr",            type=float, default=0.01)
    p.add_argument("--no-lbfgs",      action="store_true",
                   help="Skip L-BFGS refinement step (faster but slightly less precise).")
    p.add_argument("--situation",
                   choices=list(SITUATION_STRENGTH),
                   default="all",
                   help="Strength-state filter: 'all'=all situations (default), 'even'=5v5, 'pp'=power play.")
    p.add_argument("--run-all",       action="store_true",
                   help="Fit all three situations (all, even, pp) sequentially.")
    return p.parse_args()


def _fit_situation(args: argparse.Namespace, situation: str) -> None:
    """Fit (or update) the IRT model for one strength-state situation."""
    logging.info("=" * 60)
    logging.info("SITUATION: %s  (strength_states=%s)", situation, SITUATION_STRENGTH[situation])
    logging.info("=" * 60)

    os.makedirs(args.out, exist_ok=True)

    cfg = ModelConfig(
        seasons=[2024, 2025],
        require_prev_season=False,
        strength_states=SITUATION_STRENGTH[situation],
        max_epochs_full=args.epochs_full,
        max_epochs_update=args.epochs_update,
        lr=args.lr,
        lbfgs_steps=0 if args.no_lbfgs else 20,
        output_dir=args.out,
    )

    state_path = os.path.join(args.out, f"state_{situation}.pkl")

    # ── Warm-start update mode ────────────────────────────────────────────────
    if args.update:
        if not os.path.exists(state_path):
            logging.error("No saved state found at %s. Run a full fit first.", state_path)
            return
        if not args.new_csv:
            logging.error("--new-csv is required in --update mode.")
            return

        logging.info("Loading existing state from %s", state_path)
        state = load_model_state(state_path)

        logging.info("Loading new shots from %s", args.new_csv)
        new_df = pd.read_csv(args.new_csv)

        state = update_map_with_new_week(state, new_df, cfg)
        save_model_state(state, state_path)

    # ── Full fit from scratch ────────────────────────────────────────────────
    else:
        frames = []

        if os.path.exists(args.prev):
            logging.info("Loading previous season: %s", args.prev)
            frames.append(pd.read_csv(args.prev))
        else:
            logging.warning("Previous-season file not found, skipping: %s", args.prev)

        if os.path.exists(args.curr):
            logging.info("Loading current season: %s", args.curr)
            frames.append(pd.read_csv(args.curr))
        else:
            logging.error("Current-season file not found: %s", args.curr)
            return

        raw = pd.concat(frames, ignore_index=True)
        logging.info("Combined dataset: %d rows", len(raw))

        logging.info("Preparing model data …")
        data = prepare_shot_data(raw, cfg)
        logging.info("%s", data)

        logging.info("Fitting MAP model (this may take several minutes) …")
        state = fit_full_map(data, cfg)
        save_model_state(state, state_path)
        logging.info("State saved → %s", state_path)

        # For backward compatibility: 'all' situation also writes state_latest.pkl
        if situation == "all":
            legacy = os.path.join(args.out, "state_latest.pkl")
            save_model_state(state, legacy)
            logging.info("Backward-compat copy → %s", legacy)

    # ── Summaries ─────────────────────────────────────────────────────────────
    logging.info("Writing summaries …")

    shooter_df, goalie_df = summarize_skills(state, season_idx=1)
    gsax_df = compute_gsax(state, season_idx=1)

    cache_path = os.path.join(args.out, "player_names_cache.json")
    goalie_names = fetch_player_names(
        gsax_df["goalie_id"].dropna().unique(), cache_path=cache_path
    )
    def _name(id_series):
        return id_series.map(lambda x: goalie_names.get(int(float(x)), str(x)) if x == x else None)

    gsax_df.insert(0,   "player_name", _name(gsax_df["goalie_id"]))
    goalie_df.insert(0, "player_name", _name(goalie_df["goalie_id"]))

    shooter_df.to_csv(os.path.join(args.out, f"shooter_skills_{situation}.csv"), index=False)
    goalie_df.to_csv(os.path.join(args.out,  f"goalie_skills_{situation}.csv"),  index=False)
    gsax_df.to_csv(os.path.join(args.out,    f"goalie_gsax_{situation}.csv"),    index=False)
    logging.info("Shooter skills → shooter_skills_%s.csv (%d rows)", situation, len(shooter_df))
    logging.info("Goalie skills  → goalie_skills_%s.csv  (%d rows)", situation, len(goalie_df))

    diag = evaluate_predictions(state, season_idx=1)
    logging.info(
        "Diagnostics (current season)  Brier: model=%.5f  xG-only=%.5f  |  "
        "LogLoss: model=%.5f  xG-only=%.5f  |  n=%d",
        diag["brier_model"], diag["brier_xg_only"],
        diag["logloss_model"], diag["logloss_xg_only"],
        diag["n_shots"],
    )

    if gsax_df is not None and len(gsax_df):
        top5 = gsax_df.dropna(subset=["gsax"]).head(5) if "gsax" in gsax_df.columns else gsax_df.head(5)
        logging.info("\nTop 5 goalies by GSAx (situation=%s):\n%s", situation, top5.to_string(index=False))


def main():
    args = parse_args()
    situations = ["all", "even", "pp"] if args.run_all else [args.situation]
    for sit in situations:
        _fit_situation(args, sit)


if __name__ == "__main__":
    main()
