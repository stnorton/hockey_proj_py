"""
build_data.py
-------------
Generate pre-aggregated CSVs + meta.json for the Streamlit dashboard.

Usage
-----
  python dashboard/build_data.py
  python dashboard/build_data.py --state model_output/state_latest.pkl \\
                                  --cache model_output/player_names_cache.json \\
                                  --out   dashboard/data \\
                                  --season-idx 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import math
from math import sqrt

import numpy as np
import pandas as pd
import requests
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from goalie_model import load_model_state, compute_gsax, summarize_skills
from goalie_model.fit import _rebuild_model_data_from_state
from goalie_model.irt_model import DynamicIRTModel
from goalie_model.summarize import fetch_player_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Situation labels ──────────────────────────────────────────────────────────────
SITUATION_LABELS: dict = {
    "all":  "All Situations",
    "even": "Even Strength (5v5)",
    "pp":   "Power Play",
}


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build dashboard data CSVs.")
    p.add_argument(
        "--state",
        default=None,
        help="Path to saved ModelState pickle. Auto-derived from --situation if not set.",
    )
    p.add_argument(
        "--cache",
        default=str(ROOT / "model_output" / "player_names_cache.json"),
        help="Path to player names JSON cache.",
    )
    p.add_argument(
        "--out",
        default=str(Path(__file__).parent / "data"),
        help="Base output directory. CSVs are written to <out>/<situation>/.",
    )
    p.add_argument(
        "--season-idx",
        type=int,
        default=1,
        help="Season index to export (0=prev, 1=current).",
    )
    p.add_argument(
        "--situation",
        choices=list(SITUATION_LABELS),
        default="all",
        help="Which situation to export: 'all', 'even' (5v5), or 'pp' (power play).",
    )
    p.add_argument(
        "--run-all",
        action="store_true",
        help="Build data for all three situations (all, even, pp) sequentially.",
    )
    p.add_argument(
        "--prev",
        default=str(ROOT / "ingest_scripts" / "nhl_pbp_2024_2025_with_xg.csv"),
        help="Path to previous-season xG CSV (used for hex map coordinate data).",
    )
    p.add_argument(
        "--curr",
        default=str(ROOT / "ingest_scripts" / "nhl_pbp_2025_2026_with_xg.csv"),
        help="Path to current-season xG CSV (used for hex map coordinate data).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Team + position lookup (separate cache file alongside name cache)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_player_info(
    player_ids: list[int],
    cache_dir: Path,
    delay: float = 0.2,
) -> dict[int, dict]:
    """
    Return {player_id: {"team": "TOR", "position": "C"}} for every ID.
    Reads/writes a player_info_cache.json alongside the name cache.
    """
    info_path = cache_dir / "player_info_cache.json"
    info_cache: dict[int, dict] = {}
    if info_path.exists():
        with open(info_path) as f:
            info_cache = {int(k): v for k, v in json.load(f).items()}

    unique_ids = list(dict.fromkeys(int(float(pid)) for pid in player_ids))
    to_fetch = [pid for pid in unique_ids if pid not in info_cache]

    if to_fetch:
        log.info("Fetching team/position for %d players from NHL API …", len(to_fetch))
        for i, pid in enumerate(to_fetch):
            for attempt in range(4):
                try:
                    r = requests.get(
                        f"https://api-web.nhle.com/v1/player/{pid}/landing",
                        timeout=10,
                    )
                    if r.status_code == 429:
                        wait = 15 * (2 ** attempt)
                        log.warning("429 on %s — waiting %ds …", pid, wait)
                        time.sleep(wait)
                        continue
                    r.raise_for_status()
                    d = r.json()
                    info_cache[pid] = {
                        "team":     d.get("currentTeamAbbrev", ""),
                        "position": d.get("position", ""),
                    }
                    break
                except Exception as exc:
                    log.warning("Failed info for %s: %s", pid, exc)
                    info_cache[pid] = {"team": "", "position": ""}
                    break
            if delay and i < len(to_fetch) - 1:
                time.sleep(delay)
            if (i + 1) % 100 == 0:
                log.info("  %d / %d done", i + 1, len(to_fetch))
                _save_json(info_path, {str(k): v for k, v in info_cache.items()})

        _save_json(info_path, {str(k): v for k, v in info_cache.items()})
        log.info("Player info cache saved → %s", info_path)

    return {pid: info_cache.get(pid, {"team": "", "position": ""}) for pid in unique_ids}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# FSAx computation (shooter analogue of GSAx)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fsax(
    model: DynamicIRTModel,
    data,
    season_idx: int,
) -> pd.DataFrame:
    """
    Goals Above Expected (finishing) per shooter for one season.

    For each shot i by shooter p:
        p_full_i       = sigmoid(β₀ + α·xg_i + θ[p,s,t] − φ[q,s,t])
        p_no_shooter_i = sigmoid(β₀ + α·xg_i           − φ[q,s,t])

    FSAx(p) = Σ_i (p_full_i − p_no_shooter_i)
            = goals scored above an average shooter facing the same goalies

    Returns DataFrame with columns:
        shooter_id, shots_taken, goals_actual, goals_xg_only,
        fsax_raw, goalie_difficulty_adj, fsax
    """
    season_arr    = data.season_idx.numpy()
    shooter_idx_arr = data.shooter_idx.numpy()
    goalie_idx_arr  = data.goalie_idx.numpy()
    s_mask = season_arr == season_idx

    with torch.no_grad():
        logit_full    = model.predict_logit(data)                              # β₀+α·xg+θ−φ
        phi_j         = model.phi[data.goalie_idx, data.season_idx, data.week_idx]
        logit_no_s    = model.beta0 + model.alpha * data.xg_logit - phi_j     # β₀+α·xg−φ  (θ=0)

        # Use shot-weighted league-mean φ as baseline so goalie_difficulty_adj
        # is centred at 0 across all shooters (not biased by φ≠0 intercept)
        s_mask_t = torch.tensor(season_arr == season_idx)
        mean_phi_season = float(phi_j[s_mask_t].mean())
        logit_avg_g = model.beta0 + model.alpha * data.xg_logit - mean_phi_season

        p_full        = torch.sigmoid(logit_full).numpy()
        p_no_s        = torch.sigmoid(logit_no_s).numpy()
        p_avg_goalie  = torch.sigmoid(logit_avg_g).numpy()   # IRT baseline: league-avg goalie
        p_xg_raw      = torch.sigmoid(data.xg_logit).numpy() # raw xG model (no IRT)
        y             = data.y.numpy()

    rows = []
    for p, pid in enumerate(data.idx_to_shooter_id):
        mask = s_mask & (shooter_idx_arr == p)
        if mask.sum() == 0:
            continue
        goals_actual   = float(y[mask].sum())
        goals_xg_only  = float(p_xg_raw[mask].sum())          # raw xG, no IRT
        fsax_raw       = round(goals_actual - goals_xg_only, 3)
        # IRT-adjusted: credit for shooter's own θ only
        fsax           = round(float((p_full[mask] - p_no_s[mask]).sum()), 4)
        # Goalie difficulty: actual goalies faced vs league-average goalie (shot-weighted mean φ)
        # Negative = faced better-than-average goalies; positive = easier goalies
        goalie_adj     = round(float((p_no_s[mask] - p_avg_goalie[mask]).sum()), 3)
        rows.append({
            "shooter_id":           int(float(pid)),
            "shots_taken":          int(mask.sum()),
            "goals_actual":         round(goals_actual, 1),
            "goals_xg_only":        round(goals_xg_only, 3),
            "fsax_raw":             fsax_raw,
            "goalie_difficulty_adj": goalie_adj,
            "fsax":                 fsax,
        })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("fsax", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Hex map computation
# ─────────────────────────────────────────────────────────────────────────────

HEX_SIZE_FEET = 15.0


def _hex_axial(x: float, y: float) -> tuple[int, int]:
    """Convert Cartesian (x, y) to rounded axial hex coordinates (q, r)."""
    q = (2 / 3) * x / HEX_SIZE_FEET
    r = (-1 / 3) * x / HEX_SIZE_FEET + (sqrt(3) / 3) * y / HEX_SIZE_FEET
    return round(q), round(r)


def _hex_center(q: int, r: int) -> tuple[float, float]:
    """Convert axial hex (q, r) to Cartesian centre (x, y) in feet."""
    x = HEX_SIZE_FEET * (3 / 2) * q
    y = HEX_SIZE_FEET * (sqrt(3) / 2 * q + sqrt(3) * r)
    return round(x, 2), round(y, 2)


def _decode_strength_code(code) -> str:
    try:
        code = int(float(code))
    except (TypeError, ValueError):
        return "EVEN"
    away = (code // 100) % 10
    home = (code // 10) % 10
    if away == home:
        return "EVEN"
    elif away > home:
        return "PP"
    return "PK"


def build_hex_data(
    raw_csv_paths: list[str],
    state,
    model: DynamicIRTModel,
    season_idx: int,
    strength_states,          # None = all, ["EVEN"] = even, ["PP"] = pp
    names: dict,
    info: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute per-hex shot deviation stats for shooters, goalies, and league.

    Returns
    -------
    hex_shooter, hex_goalie, hex_league DataFrames
    """
    # ── Load & combine raw shot CSVs ──────────────────────────────────────────
    frames = []
    for p in raw_csv_paths:
        if os.path.exists(p):
            frames.append(pd.read_csv(p, low_memory=False))
    if not frames:
        log.warning("No raw CSV paths found for hex map build; skipping.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # ── Column standardisation (mirrors data_prep._clean_raw) ─────────────────
    if "shooter_id" not in df.columns:
        shooting = df.get("details.shootingPlayerId")
        scoring  = df.get("details.scoringPlayerId")
        if shooting is not None and scoring is not None:
            df["shooter_id"] = shooting.fillna(scoring)
        elif shooting is not None:
            df["shooter_id"] = shooting
        else:
            df["shooter_id"] = scoring

    if "goalie_id" not in df.columns:
        df["goalie_id"] = df.get("details.goalieInNetId")

    if "is_goal" not in df.columns:
        df["is_goal"] = df.get("shot_made", 0)

    if "xg" not in df.columns:
        df["xg"] = df.get("xG", np.nan)

    if "strength_state" not in df.columns:
        sc = df.get("situation_code")
        if sc is not None:
            df["strength_state"] = sc.apply(_decode_strength_code)
        else:
            df["strength_state"] = "EVEN"

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    for col in ["shooter_id", "goalie_id", "game_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["is_goal"] = pd.to_numeric(df["is_goal"], errors="coerce")
    df["xg"]      = pd.to_numeric(df["xg"],      errors="coerce")

    # Drop rows missing required fields
    df = df.dropna(subset=["shooter_id", "goalie_id", "xg", "is_goal", "xCoord", "yCoord"])
    df = df[~df["shooter_id"].isin(["nan", "None", ""]) &
            ~df["goalie_id"].isin(["nan", "None", ""])]

    # ── Shootout exclusion ────────────────────────────────────────────────────
    if "periodDescriptor.periodType" in df.columns:
        df = df[df["periodDescriptor.periodType"] != "SO"]

    # ── Season filter: current season only ───────────────────────────────────
    seasons = state.config.seasons
    def _season_idx(gid):
        try:
            y = int(float(gid)) // 1_000_000
        except Exception:
            return None
        if y == seasons[0]:
            return 0
        if y == seasons[1]:
            return 1
        return None

    df["_season_idx"] = df["game_id"].apply(_season_idx)
    df = df[df["_season_idx"] == season_idx].copy()

    # ── Strength-state filter ─────────────────────────────────────────────────
    if strength_states is not None:
        df = df[df["strength_state"].isin(strength_states)]

    if df.empty:
        log.warning("No shots remain after season/strength filter for hex map.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ── Coordinate normalisation (all shots face right net) ───────────────────
    def _norm(row):
        x, y = float(row["xCoord"]), float(row["yCoord"])
        side = str(row.get("homeTeamDefendingSide", "left")).lower()
        return (-x, -y) if side == "right" else (x, y)

    df[["x_norm", "y_norm"]] = pd.DataFrame(
        df.apply(_norm, axis=1).tolist(), index=df.index, columns=["x_norm", "y_norm"]
    )
    df = df[df["x_norm"] > 0.0].copy()

    if df.empty:
        log.warning("No offensive-zone shots for hex map.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ── Hex assignment ────────────────────────────────────────────────────────
    hex_coords = df.apply(lambda r: pd.Series(
        _hex_axial(r["x_norm"], r["y_norm"]), index=["hex_q", "hex_r"]
    ), axis=1)
    df = pd.concat([df, hex_coords], axis=1)

    # ── IRT-adjusted p_adj per shot ───────────────────────────────────────────
    # Look up shooter mu_theta and goalie mu_phi from the model state
    param = state.param_dict
    # mu_theta: shape [n_shooters, n_seasons]
    mu_theta_arr = np.asarray(param.get("mu_theta", np.zeros((model.n_shooters, model.n_seasons))))
    mu_phi_arr   = np.asarray(param.get("mu_phi",   np.zeros((model.n_goalies,  model.n_seasons))))
    beta0 = float(np.asarray(param.get("beta0", 0.0)).item())
    alpha = float(np.asarray(param.get("alpha", 1.0)).item())

    s2i = state.shooter_id_to_idx
    g2i = state.goalie_id_to_idx

    def _safe_logit(p: float) -> float:
        p = max(1e-5, min(1 - 1e-5, p))
        return math.log(p / (1 - p))

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _p_adj(row):
        sid = str(row["shooter_id"])
        gid = str(row["goalie_id"])
        xg  = float(row["xg"])
        si  = s2i.get(sid, -1)
        gi  = g2i.get(gid, -1)
        theta = float(mu_theta_arr[si]) if si >= 0 else 0.0
        phi   = float(mu_phi_arr[gi])   if gi >= 0 else 0.0
        logit = beta0 + alpha * _safe_logit(xg) + theta - phi
        return _sigmoid(logit)

    log.info("Computing IRT-adjusted p_adj for %d hex-map shots …", len(df))
    df["p_adj"] = df.apply(_p_adj, axis=1)

    # ── League hex aggregate ──────────────────────────────────────────────────
    hex_league = (
        df.groupby(["hex_q", "hex_r"])
        .agg(
            shots  =("is_goal", "count"),
            goals  =("is_goal", "sum"),
            xg_sum =("xg",      "sum"),
        )
        .reset_index()
    )
    hex_league["goal_rate"] = (hex_league["goals"] / hex_league["shots"]).round(4)
    hex_league["xg_rate"]   = (hex_league["xg_sum"] / hex_league["shots"]).round(4)
    centres = hex_league.apply(
        lambda r: pd.Series(_hex_center(int(r["hex_q"]), int(r["hex_r"])),
                            index=["hex_x", "hex_y"]), axis=1
    )
    hex_league = pd.concat([hex_league, centres], axis=1)
    col_order = ["hex_q", "hex_r", "hex_x", "hex_y", "shots", "goals", "xg_sum",
                 "goal_rate", "xg_rate"]
    hex_league = hex_league[col_order]

    # ── Shooter hex aggregate ──────────────────────────────────────────────────
    hex_shooter_raw = (
        df.groupby(["shooter_id", "hex_q", "hex_r"])
        .agg(
            shots      =("is_goal", "count"),
            goals      =("is_goal", "sum"),
            xg_sum     =("xg",      "sum"),
            irt_xg_sum =("p_adj",   "sum"),
        )
        .reset_index()
    )
    hex_shooter_raw["deviation"]    = (hex_shooter_raw["goals"] - hex_shooter_raw["irt_xg_sum"]).round(4)
    hex_shooter_raw["dev_per_shot"] = (hex_shooter_raw["deviation"] / hex_shooter_raw["shots"]).round(4)
    hex_shooter_raw["shooter_id"]   = hex_shooter_raw["shooter_id"].apply(lambda x: int(float(x)))
    centres_s = hex_shooter_raw.apply(
        lambda r: pd.Series(_hex_center(int(r["hex_q"]), int(r["hex_r"])),
                            index=["hex_x", "hex_y"]), axis=1
    )
    hex_shooter_raw = pd.concat([hex_shooter_raw, centres_s], axis=1)
    hex_shooter_raw.insert(0, "player_name", hex_shooter_raw["shooter_id"].map(
        lambda pid: names.get(int(pid), str(pid))
    ))
    hex_shooter_raw.insert(1, "team", hex_shooter_raw["shooter_id"].map(
        lambda pid: info.get(int(pid), {}).get("team", "")
    ))
    hex_shooter = hex_shooter_raw[[
        "shooter_id", "player_name", "team",
        "hex_q", "hex_r", "hex_x", "hex_y",
        "shots", "goals", "xg_sum", "irt_xg_sum", "deviation", "dev_per_shot",
    ]]

    # ── Goalie hex aggregate ──────────────────────────────────────────────────
    hex_goalie_raw = (
        df.groupby(["goalie_id", "hex_q", "hex_r"])
        .agg(
            shots      =("is_goal", "count"),
            goals      =("is_goal", "sum"),
            xg_sum     =("xg",      "sum"),
            irt_xg_sum =("p_adj",   "sum"),
        )
        .reset_index()
    )
    # For goalies: positive = saves above expected (sign flip vs shooter)
    hex_goalie_raw["deviation"]    = (hex_goalie_raw["irt_xg_sum"] - hex_goalie_raw["goals"]).round(4)
    hex_goalie_raw["dev_per_shot"] = (hex_goalie_raw["deviation"] / hex_goalie_raw["shots"]).round(4)
    hex_goalie_raw["goalie_id"]    = hex_goalie_raw["goalie_id"].apply(lambda x: int(float(x)))
    centres_g = hex_goalie_raw.apply(
        lambda r: pd.Series(_hex_center(int(r["hex_q"]), int(r["hex_r"])),
                            index=["hex_x", "hex_y"]), axis=1
    )
    hex_goalie_raw = pd.concat([hex_goalie_raw, centres_g], axis=1)
    hex_goalie_raw.insert(0, "player_name", hex_goalie_raw["goalie_id"].map(
        lambda pid: names.get(int(pid), str(pid))
    ))
    hex_goalie_raw.insert(1, "team", hex_goalie_raw["goalie_id"].map(
        lambda pid: info.get(int(pid), {}).get("team", "")
    ))
    hex_goalie = hex_goalie_raw[[
        "goalie_id", "player_name", "team",
        "hex_q", "hex_r", "hex_x", "hex_y",
        "shots", "goals", "xg_sum", "irt_xg_sum", "deviation", "dev_per_shot",
    ]]

    return hex_shooter, hex_goalie, hex_league


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _build_one_situation(args: argparse.Namespace, situation: str) -> None:
    """Build dashboard CSVs for a single strength-state situation."""
    log.info("=" * 60)
    log.info("Building data for: %s", SITUATION_LABELS[situation])
    log.info("=" * 60)

    # ── Resolve state path ────────────────────────────────────────────────────
    if args.state is not None:
        state_path = str(args.state)
    else:
        candidate = str(ROOT / "model_output" / f"state_{situation}.pkl")
        fallback  = str(ROOT / "model_output" / "state_latest.pkl")
        if os.path.exists(candidate):
            state_path = candidate
        elif situation == "all" and os.path.exists(fallback):
            log.info("state_all.pkl not found; falling back to state_latest.pkl")
            state_path = fallback
        else:
            log.error(
                "No state file found for situation '%s'. Expected: %s\n"
                "Run: python run_model.py --situation %s",
                situation, candidate, situation,
            )
            return

    # ── Resolve output directory ──────────────────────────────────────────────
    out_dir   = Path(args.out) / situation
    cache_dir = Path(args.cache).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    season_idx = args.season_idx

    # ── Load & rebuild model ──────────────────────────────────────────────────
    log.info("Loading model state from %s", state_path)
    state = load_model_state(state_path)

    data  = _rebuild_model_data_from_state(state, state.config)
    model = DynamicIRTModel(data, state.config)
    model.load_state_dict_numpy(state.param_dict)
    model.eval()

    # ── Per-week shot counts for confidence bands ─────────────────────────────
    shots_s1 = state.shots_df[state.shots_df["season_idx"] == season_idx].copy()
    shots_s1["goalie_id_int"]  = shots_s1["goalie_id"].apply(lambda x: int(float(x)))
    shots_s1["shooter_id_int"] = shots_s1["shooter_id"].apply(lambda x: int(float(x)))

    goalie_wk_shots = (
        shots_s1.groupby(["goalie_id_int", "week_idx"])
        .size()
        .reset_index(name="shots_that_week")
        .rename(columns={"goalie_id_int": "goalie_id", "week_idx": "week"})
    )
    shooter_wk_shots = (
        shots_s1.groupby(["shooter_id_int", "week_idx"])
        .size()
        .reset_index(name="shots_that_week")
        .rename(columns={"shooter_id_int": "shooter_id", "week_idx": "week"})
    )

    # ── Skill summaries (weekly) ──────────────────────────────────────────────
    log.info("Computing weekly skill trajectories …")
    shooter_weekly_raw, goalie_weekly_raw = summarize_skills(state, season_idx=season_idx)

    goalie_weekly_raw["goalie_id"]   = goalie_weekly_raw["goalie_id"].apply(lambda x: int(float(x)))
    shooter_weekly_raw["shooter_id"] = shooter_weekly_raw["shooter_id"].apply(lambda x: int(float(x)))

    goalie_weekly = goalie_weekly_raw.merge(goalie_wk_shots, on=["goalie_id", "week"], how="left")
    goalie_weekly["shots_that_week"] = goalie_weekly["shots_that_week"].fillna(0).astype(int)

    shooter_weekly = shooter_weekly_raw.merge(shooter_wk_shots, on=["shooter_id", "week"], how="left")
    shooter_weekly["shots_that_week"] = shooter_weekly["shots_that_week"].fillna(0).astype(int)

    # Use the *learned* random-walk tau (clamped at tau_min during fitting)
    # instead of the prior_tau hyper-parameter, which is far too wide.
    tau_min = getattr(state.config, "tau_min", 0.01)
    tau_phi   = max(float(np.exp(state.param_dict.get("log_tau_phi",   np.log(0.2)))), tau_min)
    tau_theta = max(float(np.exp(state.param_dict.get("log_tau_theta", np.log(0.2)))), tau_min)
    log.info("Confidence-band tau:  phi=%.4f  theta=%.4f", tau_phi, tau_theta)

    for df, skill_col, lo_col, hi_col, tau in [
        (goalie_weekly,  "phi",   "phi_lo",   "phi_hi",   tau_phi),
        (shooter_weekly, "theta", "theta_lo", "theta_hi", tau_theta),
    ]:
        band = (2 * tau / (df["shots_that_week"].clip(lower=1) ** 0.5)).round(4)
        df[lo_col] = (df[skill_col] - band).round(4)
        df[hi_col] = (df[skill_col] + band).round(4)

    # ── GSAx (goalie summary) ─────────────────────────────────────────────────
    log.info("Computing GSAx …")
    gsax_df = compute_gsax(state, season_idx=season_idx)
    gsax_df["goalie_id"] = gsax_df["goalie_id"].apply(lambda x: int(float(x)))
    gsax_df["sv_pct"]    = (1 - gsax_df["goals_actual"] / gsax_df["shots_faced"]).round(4)
    gsax_df["xgsv_pct"]  = (1 - gsax_df["goals_xg_only"] / gsax_df["shots_faced"]).round(4)

    # ── FSAx (shooter summary) ────────────────────────────────────────────────
    log.info("Computing FSAx …")
    fsax_df = compute_fsax(model, data, season_idx)

    shooter_summary = (
        shooter_weekly
        .groupby("shooter_id")
        .agg(
            mu_theta    =("mu_theta", "first"),
            theta_latest=("theta",    "last"),
            theta_min   =("theta",    "min"),
            theta_max   =("theta",    "max"),
        )
        .reset_index()
    )
    shooter_summary = shooter_summary.merge(fsax_df, on="shooter_id", how="left")

    # ── Player names + team/position ──────────────────────────────────────────
    all_ids = (
        list(gsax_df["goalie_id"].unique())
        + list(shooter_summary["shooter_id"].unique())
    )
    log.info("Resolving %d player names (from cache, no API calls expected) …", len(all_ids))
    names = fetch_player_names(all_ids, cache_path=args.cache, delay=0)

    log.info("Fetching team / position for %d players …", len(all_ids))
    info  = fetch_player_info(all_ids, cache_dir)

    def _name(pid: int) -> str:  return names.get(int(float(pid)), str(pid))
    def _team(pid: int) -> str:  return info.get(int(float(pid)), {}).get("team", "")
    def _pos(pid: int)  -> str:  return info.get(int(float(pid)), {}).get("position", "")

    # ── Assemble final tables ─────────────────────────────────────────────────
    goalie_summary = gsax_df.copy()
    goalie_summary.insert(0, "player_name", goalie_summary["goalie_id"].map(_name))
    goalie_summary.insert(1, "team",        goalie_summary["goalie_id"].map(_team))
    goalie_summary.insert(2, "position",    "G")

    goalie_weekly.insert(0, "player_name", goalie_weekly["goalie_id"].map(_name))
    goalie_weekly.insert(1, "team",        goalie_weekly["goalie_id"].map(_team))

    shooter_summary.insert(0, "player_name", shooter_summary["shooter_id"].map(_name))
    shooter_summary.insert(1, "team",        shooter_summary["shooter_id"].map(_team))
    shooter_summary.insert(2, "position",    shooter_summary["shooter_id"].map(_pos))

    shooter_weekly.insert(0, "player_name", shooter_weekly["shooter_id"].map(_name))
    shooter_weekly.insert(1, "team",        shooter_weekly["shooter_id"].map(_team))

    # ── Write ─────────────────────────────────────────────────────────────────
    goalie_summary.to_csv(out_dir / "goalie_summary.csv",   index=False)
    goalie_weekly.to_csv(out_dir / "goalie_weekly.csv",     index=False)
    shooter_summary.to_csv(out_dir / "shooter_summary.csv", index=False)
    shooter_weekly.to_csv(out_dir / "shooter_weekly.csv",   index=False)

    # ── Hex map data ──────────────────────────────────────────────────────────
    log.info("Building hex map data …")
    raw_csvs = [args.prev, args.curr]
    strength_states = state.config.strength_states   # None / ["EVEN"] / ["PP"]
    hex_shooter_df, hex_goalie_df, hex_league_df = build_hex_data(
        raw_csvs, state, model, season_idx, strength_states, names, info
    )
    if not hex_shooter_df.empty:
        hex_shooter_df.to_csv(out_dir / "hex_shooter.csv", index=False)
        hex_goalie_df.to_csv(out_dir / "hex_goalie.csv",   index=False)
        hex_league_df.to_csv(out_dir / "hex_league.csv",   index=False)
        log.info("  hex_shooter: %d rows  |  hex_goalie: %d rows  |  hex_league: %d rows",
                 len(hex_shooter_df), len(hex_goalie_df), len(hex_league_df))
    else:
        log.warning("Hex map data was empty — CSV files not written.")

    season_label = (
        f"{state.config.seasons[season_idx]}-"
        f"{str(state.config.seasons[season_idx] + 1)[2:]}"
    )

    # Model params needed by the dashboard (H2H goals-per-100)
    beta0 = float(np.asarray(state.param_dict.get("beta0", 0.0)).item())
    alpha = float(np.asarray(state.param_dict.get("alpha", 1.0)).item())
    mean_xg_logit = float(data.xg_logit[data.season_idx == season_idx].mean())

    # Playoff start week = first week index beyond the regular-season data.
    # Weekly indices are 0-based; playoffs begin the week after the last data week.
    max_reg_week = int(goalie_weekly["week"].max()) if "week" in goalie_weekly.columns and len(goalie_weekly) > 0 else 26
    playoff_start_week = max_reg_week + 1

    meta = {
        "last_updated":          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "season":                season_label,
        "situation":             situation,
        "situation_label":       SITUATION_LABELS[situation],
        "n_goalies":             len(goalie_summary),
        "n_shooters":            len(shooter_summary),
        "n_goalie_weekly_rows":  len(goalie_weekly),
        "n_shooter_weekly_rows": len(shooter_weekly),
        "beta0":                 round(beta0, 6),
        "alpha":                 round(alpha, 6),
        "mean_xg_logit":         round(mean_xg_logit, 6),
        "playoff_start_week":    playoff_start_week,
    }
    _save_json(out_dir / "meta.json", meta)

    log.info("Done. Written to %s", out_dir)
    log.info("  goalie_summary:  %d rows", len(goalie_summary))
    log.info("  goalie_weekly:   %d rows", len(goalie_weekly))
    log.info("  shooter_summary: %d rows", len(shooter_summary))
    log.info("  shooter_weekly:  %d rows", len(shooter_weekly))


def main() -> None:
    args = parse_args()
    situations = ["all", "even", "pp"] if args.run_all else [args.situation]
    for sit in situations:
        _build_one_situation(args, sit)


if __name__ == "__main__":
    main()
