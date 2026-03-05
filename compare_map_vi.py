"""
compare_map_vi.py
-----------------
Compare MAP estimates (from run_model.py) against VI posterior
(from goalie_model/vi_model.py) to determine whether:

  1. tau_theta and tau_phi collapse under MAP but not under VI
     (if VI finds tau >> 0, the collapse is a MAP artifact, not a data issue)
  2. Player skill rankings are substantially different under the two methods
  3. McDavid's ranking changes with full uncertainty quantification

Usage:
    # Run VI from scratch then compare:
    python compare_map_vi.py

    # Use a cached VI state:
    python compare_map_vi.py --vi-state model_output/vi_state.pkl

    # Skip plots (terminal-only output):
    python compare_map_vi.py --no-plot

Dependencies:
    pip install pyro-ppl>=1.9
    pip install matplotlib  (for plots; optional)
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from goalie_model import ModelConfig, load_model_state
from goalie_model.data_prep import prepare_shot_data
from goalie_model.vi_model import (
    fit_svi,
    posterior_samples,
    save_vi_state,
    load_vi_state,
    restore_guide,
)

log = logging.getLogger(__name__)

DEFAULT_MAP_STATE = os.path.join(ROOT, "model_output", "state_latest.pkl")
DEFAULT_VI_STATE  = os.path.join(ROOT, "model_output", "vi_state.pkl")
DEFAULT_PREV_CSV  = os.path.join(ROOT, "ingest_scripts", "nhl_pbp_2024_2025_with_xg.csv")
DEFAULT_CURR_CSV  = os.path.join(ROOT, "ingest_scripts", "nhl_pbp_2025_2026_with_xg.csv")
DEFAULT_OUT_DIR   = os.path.join(ROOT, "model_output")

MCDAVID_ID = 8478402  # NHL player ID for Connor McDavid


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ci(arr: np.ndarray) -> str:
    """Format posterior mean ± 2SD  [2.5% – 97.5%] for a scalar array."""
    return (
        f"{arr.mean():.4f} ± {arr.std():.4f}"
        f"  [95% CI: {np.percentile(arr, 2.5):.4f} – {np.percentile(arr, 97.5):.4f}]"
    )


def _load_map_state(path: str):
    if not os.path.exists(path):
        sys.exit(f"MAP state not found at {path}.  Run run_model.py first.")
    log.info("Loading MAP state from %s", path)
    return load_model_state(path)


def _rebuild_data(map_state, cfg: ModelConfig):
    """
    Reconstruct ModelData from the shots stored in the MAP state, using
    the same player-index mappings so indices align with MAP parameters.
    """
    from goalie_model.data_prep import _assemble_model_data

    return _assemble_model_data(
        df=map_state.shots_df,
        shooter_id_to_idx=map_state.shooter_id_to_idx,
        goalie_id_to_idx=map_state.goalie_id_to_idx,
        idx_to_shooter=map_state.idx_to_shooter_id,
        idx_to_goalie=map_state.idx_to_goalie_id,
        season_labels=map_state.season_labels,
    )


def _extract_map_params(map_state) -> dict:
    """Pull key MAP parameter values out of the state dict."""
    p = map_state.param_dict
    return {
        "beta0":             float(p["beta0"]),
        "alpha":             float(p["alpha"]),
        "tau_theta_map":     float(np.exp(p["log_tau_theta"])),
        "tau_phi_map":       float(np.exp(p["log_tau_phi"])),
        "sigma_seas_theta":  float(np.exp(p["log_sigma_season_theta"])),
        "sigma_seas_phi":    float(np.exp(p["log_sigma_season_phi"])),
        "mu_theta":          np.array(p["mu_theta"]),   # [N_shooters]
        "mu_phi":            np.array(p["mu_phi"]),     # [N_goalies]
    }


def _get_mcdavid_idx(map_state) -> int | None:
    """Return shooter index for McDavid, or None if not in dataset."""
    return map_state.shooter_id_to_idx.get(MCDAVID_ID)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison logic
# ─────────────────────────────────────────────────────────────────────────────

def compare(
    map_state,
    vi_samples: dict,
    show_plots: bool = True,
    top_n: int = 15,
) -> None:
    """Print a structured comparison report to stdout."""

    map_p = _extract_map_params(map_state)
    shooter_id_to_idx = map_state.shooter_id_to_idx
    goalie_id_to_idx  = map_state.goalie_id_to_idx
    idx_to_shooter    = {v: k for k, v in shooter_id_to_idx.items()}
    idx_to_goalie     = {v: k for k, v in goalie_id_to_idx.items()}

    # ── Try to load player names if a name cache exists ──────────────────────
    name_cache_path = os.path.join(DEFAULT_OUT_DIR, "player_names_cache.json")
    player_names: dict = {}
    if os.path.exists(name_cache_path):
        import json
        with open(name_cache_path) as f:
            player_names = {int(k): v for k, v in json.load(f).items()}

    def name(pid: int) -> str:
        return player_names.get(pid, str(pid))

    sep = "=" * 70

    # ── Section 1: Variance parameters ───────────────────────────────────────
    print(f"\n{sep}")
    print("SECTION 1 — VARIANCE PARAMETERS (key tau collapse check)")
    print(sep)

    for param, map_key in [
        ("tau_theta",        "tau_theta_map"),
        ("tau_phi",          "tau_phi_map"),
        ("sigma_seas_theta", "sigma_seas_theta"),
        ("sigma_seas_phi",   "sigma_seas_phi"),
    ]:
        vi_arr = vi_samples.get(param)
        map_val = map_p.get(map_key, map_p.get(param))
        if vi_arr is None:
            print(f"  {param:22s}  MAP={map_val:.6f}  VI=N/A")
            continue

        vi_arr = np.array(vi_arr).flatten()
        verdict = ""
        if map_val < 1e-3 and vi_arr.mean() > 5 * map_val:
            verdict = "  *** MAP COLLAPSED — VI finds non-trivial value ***"
        elif abs(map_val - vi_arr.mean()) / (vi_arr.mean() + 1e-9) < 0.2:
            verdict = "  (MAP ≈ VI)"

        print(
            f"  {param:22s}  MAP={map_val:.6f}  VI={_ci(vi_arr)}{verdict}"
        )

    # ── Section 2: Global scalars ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("SECTION 2 — GLOBAL SCALARS (β₀, α)")
    print(sep)
    for param, map_key in [("beta0", "beta0"), ("alpha", "alpha")]:
        vi_arr = vi_samples.get(param)
        map_val = map_p[map_key]
        if vi_arr is None:
            print(f"  {param:10s}  MAP={map_val:.4f}  VI=N/A")
        else:
            vi_arr = np.array(vi_arr).flatten()
            print(f"  {param:10s}  MAP={map_val:.4f}  VI={_ci(vi_arr)}")

    # ── Section 3: McDavid ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SECTION 3 — McDavid (shooter index analysis)")
    print(sep)
    mc_idx = _get_mcdavid_idx(map_state)
    if mc_idx is None:
        print("  McDavid not found in dataset (ID 8478402).")
    else:
        map_mu = float(map_p["mu_theta"][mc_idx])
        vi_mu_all = vi_samples.get("mu_theta")
        if vi_mu_all is not None:
            vi_mu_all = np.array(vi_mu_all)  # [n_samples, N_shooters]
            if vi_mu_all.ndim == 1:
                vi_mu_mc = vi_mu_all[mc_idx]
                vi_mu_all_mean = vi_mu_all
            else:
                vi_mu_mc = vi_mu_all[:, mc_idx]       # [n_samples]
                vi_mu_all_mean = vi_mu_all.mean(axis=0)  # [N_shooters]

            # MAP rank (by mu_theta descending)
            map_rank = int((map_p["mu_theta"] > map_mu).sum()) + 1
            # VI rank (by posterior mean descending)
            vi_rank_per_sample = [
                int((vi_mu_all[i] > vi_mu_all[i, mc_idx]).sum()) + 1
                for i in range(vi_mu_all.shape[0])
            ]
            vi_rank_mean = float(np.mean(vi_rank_per_sample))
            vi_rank_sd   = float(np.std(vi_rank_per_sample))

            n_total = len(map_p["mu_theta"])
            print(f"  MAP  mu_theta = {map_mu:.4f}  | rank {map_rank} / {n_total}")
            print(
                f"  VI   mu_theta = {_ci(vi_mu_mc)}"
            )
            print(
                f"  VI   rank     = {vi_rank_mean:.0f} ± {vi_rank_sd:.0f}  / {n_total}"
            )
        else:
            print(f"  MAP  mu_theta = {map_mu:.4f}")
            print("  VI   mu_theta sample not available.")

    # ── Section 4: Top / bottom shooters (career baselines) ──────────────────
    print(f"\n{sep}")
    print(f"SECTION 4 — TOP {top_n} SHOOTERS BY CAREER BASELINE (mu_theta)")
    print(sep)
    n_shooters = len(map_p["mu_theta"])
    map_sort = np.argsort(map_p["mu_theta"])[::-1]

    vi_mu_theta = vi_samples.get("mu_theta")
    if vi_mu_theta is not None:
        vi_mu_theta = np.array(vi_mu_theta)
        if vi_mu_theta.ndim == 1:
            vi_mean = vi_mu_theta
            vi_sd   = np.zeros_like(vi_mean)
        else:
            vi_mean = vi_mu_theta.mean(axis=0)
            vi_sd   = vi_mu_theta.std(axis=0)
    else:
        vi_mean = None

    header = f"  {'Rank':>4}  {'Player':>12}  {'MAP μ_θ':>9}  {'VI μ_θ mean':>12}  {'VI μ_θ sd':>10}"
    print(header)
    print("  " + "-" * 60)
    for rank, idx in enumerate(map_sort[:top_n], 1):
        pid = idx_to_shooter.get(idx, idx)
        pname = name(pid)[:12]
        map_val = map_p["mu_theta"][idx]
        if vi_mean is not None:
            print(f"  {rank:>4}  {pname:>12}  {map_val:>+9.4f}  {vi_mean[idx]:>+12.4f}  {vi_sd[idx]:>10.4f}")
        else:
            print(f"  {rank:>4}  {pname:>12}  {map_val:>+9.4f}")

    print(f"\n  BOTTOM {top_n}:")
    print(header)
    print("  " + "-" * 60)
    for rank, idx in enumerate(map_sort[-top_n:][::-1], 1):
        pid = idx_to_shooter.get(idx, idx)
        pname = name(pid)[:12]
        map_val = map_p["mu_theta"][idx]
        if vi_mean is not None:
            print(f"  {rank:>4}  {pname:>12}  {map_val:>+9.4f}  {vi_mean[idx]:>+12.4f}  {vi_sd[idx]:>10.4f}")
        else:
            print(f"  {rank:>4}  {pname:>12}  {map_val:>+9.4f}")

    # ── Section 5: Rank correlation ───────────────────────────────────────────
    print(f"\n{sep}")
    print("SECTION 5 — RANK CORRELATION (MAP mu_theta vs VI posterior mean)")
    print(sep)
    if vi_mean is not None:
        from scipy.stats import spearmanr, pearsonr
        r_s, p_s = spearmanr(map_p["mu_theta"], vi_mean)
        r_p, p_p = pearsonr(map_p["mu_theta"],  vi_mean)
        print(f"  Spearman ρ = {r_s:.4f}  (p={p_s:.2e})")
        print(f"  Pearson  r = {r_p:.4f}  (p={p_p:.2e})")
        if r_s > 0.95:
            print("  → Rankings highly consistent. MAP is a good approximation.")
        elif r_s > 0.80:
            print("  → Rankings largely consistent but notable differences exist.")
        else:
            print("  → Rankings diverge substantially — MAP may be misleading.")
    else:
        print("  mu_theta samples not available — skipping rank correlation.")

    # ── Section 6: Goalie μ_φ ────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"SECTION 6 — TOP {top_n} GOALIES BY CAREER BASELINE (mu_phi)")
    print(sep)
    n_goalies = len(map_p["mu_phi"])
    g_sort = np.argsort(map_p["mu_phi"])[::-1]
    vi_mu_phi = vi_samples.get("mu_phi")
    if vi_mu_phi is not None:
        vi_mu_phi = np.array(vi_mu_phi)
        g_vi_mean = vi_mu_phi.mean(axis=0) if vi_mu_phi.ndim > 1 else vi_mu_phi
        g_vi_sd   = vi_mu_phi.std(axis=0)  if vi_mu_phi.ndim > 1 else np.zeros_like(g_vi_mean)
    else:
        g_vi_mean = None

    g_header = f"  {'Rank':>4}  {'Player':>12}  {'MAP μ_φ':>9}  {'VI μ_φ mean':>12}  {'VI μ_φ sd':>10}"
    print(g_header)
    print("  " + "-" * 60)
    for rank, idx in enumerate(g_sort[:top_n], 1):
        pid = idx_to_goalie.get(idx, idx)
        pname = name(pid)[:12]
        map_val = map_p["mu_phi"][idx]
        if g_vi_mean is not None:
            print(f"  {rank:>4}  {pname:>12}  {map_val:>+9.4f}  {g_vi_mean[idx]:>+12.4f}  {g_vi_sd[idx]:>10.4f}")
        else:
            print(f"  {rank:>4}  {pname:>12}  {map_val:>+9.4f}")

    # ── Section 7: Verdict ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SECTION 7 — OVERALL VERDICT")
    print(sep)

    tau_vi = vi_samples.get("tau_theta")
    if tau_vi is not None:
        tau_vi = np.array(tau_vi).flatten()
        tau_map = map_p["tau_theta_map"]
        ratio = tau_vi.mean() / (tau_map + 1e-10)
        if ratio > 100:
            print(
                f"  tau_theta: MAP={tau_map:.6f}  VI mean={tau_vi.mean():.4f}  (ratio {ratio:.0f}×)\n"
                f"  CONCLUSION: Tau collapse is a MAP artifact. VI finds meaningful weekly\n"
                f"  dynamics (tau ≈ {tau_vi.mean():.3f}). Fix tau clamping in MAP is validated."
            )
        elif ratio > 5:
            print(
                f"  tau_theta: MAP={tau_map:.6f}  VI mean={tau_vi.mean():.4f}  (ratio {ratio:.0f}×)\n"
                f"  CONCLUSION: Some tau collapse in MAP; VI is more informative."
            )
        else:
            print(
                f"  tau_theta: MAP={tau_map:.6f}  VI mean={tau_vi.mean():.4f}  (ratio {ratio:.1f}×)\n"
                f"  CONCLUSION: MAP and VI broadly agree on tau. MAP may be a reasonable\n"
                f"  approximation for variance parameters."
            )
    print(f"\n{sep}\n")

    # ── Optional plots ────────────────────────────────────────────────────────
    if not show_plots:
        return
    try:
        import matplotlib.pyplot as plt
        _make_plots(map_p, vi_samples, player_names, idx_to_shooter, idx_to_goalie)
    except ImportError:
        log.warning("matplotlib not available — skipping plots.")


def _make_plots(map_p, vi_samples, player_names, idx_to_shooter, idx_to_goalie):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("MAP vs VI Comparison", fontsize=13)

    # Plot 1: Posterior distribution of tau_theta and tau_phi
    ax = axes[0]
    for key, label, col in [
        ("tau_theta", "τ_θ (shooter walk)", "steelblue"),
        ("tau_phi",   "τ_φ (goalie walk)",  "tomato"),
    ]:
        arr = vi_samples.get(key)
        if arr is not None:
            arr = np.array(arr).flatten()
            ax.hist(arr, bins=40, density=True, alpha=0.6, label=label, color=col)
            map_val = map_p.get(f"{key}_map", map_p.get(key))
            if map_val is not None:
                ax.axvline(map_val, color=col, linestyle="--", linewidth=2,
                           label=f"MAP {label}={map_val:.5f}")
    ax.set_xlabel("Value")
    ax.set_title("VI Posterior for τ parameters\n(dashed = MAP point estimate)")
    ax.legend(fontsize=8)

    # Plot 2: MAP vs VI posterior mean for mu_theta (shooter career baselines)
    ax = axes[1]
    vi_mu = vi_samples.get("mu_theta")
    if vi_mu is not None:
        vi_mu = np.array(vi_mu)
        vi_mean = vi_mu.mean(axis=0) if vi_mu.ndim > 1 else vi_mu
        ax.scatter(map_p["mu_theta"], vi_mean, s=4, alpha=0.4, color="steelblue")
        lim_lo = min(map_p["mu_theta"].min(), vi_mean.min())
        lim_hi = max(map_p["mu_theta"].max(), vi_mean.max())
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1)
        ax.set_xlabel("MAP  μ_θ")
        ax.set_ylabel("VI posterior mean  μ_θ")
        ax.set_title("Shooter Career Baselines:\nMAP vs VI")

    # Plot 3: Posterior uncertainty widths for the top-50 shooters (by MAP rank)
    ax = axes[2]
    vi_mu = vi_samples.get("mu_theta")
    if vi_mu is not None and vi_mu.ndim > 1:
        vi_mu = np.array(vi_mu)
        vi_sd = vi_mu.std(axis=0)
        top50 = np.argsort(map_p["mu_theta"])[::-1][:50]
        labels = [
            player_names.get(idx_to_shooter.get(i, i), str(idx_to_shooter.get(i, i)))[:10]
            for i in top50
        ]
        ax.barh(range(50), vi_sd[top50][::-1], color="steelblue", alpha=0.7)
        ax.set_yticks(range(50))
        ax.set_yticklabels(labels[::-1], fontsize=6)
        ax.set_xlabel("Posterior SD of μ_θ (VI)")
        ax.set_title("Uncertainty (SD) for Top-50 Shooters")

    plt.tight_layout()
    out_path = os.path.join(DEFAULT_OUT_DIR, "map_vi_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Comparison plot saved to %s", out_path)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Compare MAP vs VI estimates.")
    p.add_argument("--map-state", default=DEFAULT_MAP_STATE)
    p.add_argument("--vi-state",  default=None,
                   help="Path to cached vi_state.pkl. Runs VI from scratch if not supplied.")
    p.add_argument("--prev",      default=DEFAULT_PREV_CSV)
    p.add_argument("--curr",      default=DEFAULT_CURR_CSV)
    p.add_argument("--epochs",    type=int, default=20_000,
                   help="SVI epochs (only used when fitting VI from scratch).")
    p.add_argument("--n-samples", type=int, default=300,
                   help="Posterior samples to draw for comparison.")
    p.add_argument("--no-plot",   action="store_true")
    args = p.parse_args()

    # Load MAP
    map_state = _load_map_state(args.map_state)

    # Build ModelData using same index mappings as MAP
    cfg = map_state.config
    try:
        data = _rebuild_data(map_state, cfg)
        log.info("ModelData rebuilt from MAP state: %s", data)
    except Exception as e:
        log.warning("Could not rebuild data from MAP state (%s). Trying CSV load.", e)
        from goalie_model.data_prep import prepare_shot_data
        import pandas as pd
        frames = []
        for csv in [args.prev, args.curr]:
            if os.path.exists(csv):
                frames.append(pd.read_csv(csv))
        if not frames:
            sys.exit("No data found.")
        combined = pd.concat(frames, ignore_index=True)
        data, _, _ = prepare_shot_data(combined, cfg)

    # Fit VI or load cached
    if args.vi_state and os.path.exists(args.vi_state):
        log.info("Loading cached VI state from %s", args.vi_state)
        vi_state_dict = load_vi_state(args.vi_state)
        model_fn, guide = restore_guide(data, cfg, vi_state_dict)
    else:
        log.info("No cached VI state found — fitting VI now (epochs=%d).", args.epochs)
        model_fn, guide, history = fit_svi(data, cfg, n_epochs=args.epochs)
        vi_path = os.path.join(DEFAULT_OUT_DIR, "vi_state.pkl")
        save_vi_state(guide, history, cfg, vi_path)

    # Draw posterior samples
    log.info("Drawing %d posterior samples …", args.n_samples)
    vi_samp = posterior_samples(
        model_fn, guide, n_samples=args.n_samples,
        return_sites=[
            "tau_theta", "tau_phi",
            "sigma_seas_theta", "sigma_seas_phi",
            "sigma_mu_theta", "sigma_mu_phi",
            "beta0", "alpha",
            "mu_theta", "mu_phi",
        ],
    )

    compare(map_state, vi_samp, show_plots=not args.no_plot)


if __name__ == "__main__":
    main()
