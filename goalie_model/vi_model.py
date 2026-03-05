"""
vi_model.py
-----------
Variational Inference (Pyro SVI) version of the Dynamic Goalie-Shooter IRT
model.  Uses a mean-field Normal AutoNormal guide over a non-centred
(whitened) parameterisation of the dynamic random walks.

Non-centred form avoids the funnel geometry that breaks mean-field VI on
hierarchical models by reparameterising skill as:

    skill[p, s, 0] = mu[p] + sigma_season * eps0[p, s],   eps0 ~ N(0, 1)
    skill[p, s, t] = skill[p, s, t-1] + tau * delta[p, s, t],  delta ~ N(0, 1)

so all leaf latents (eps0, delta) are unit Normals under the prior, and
AutoNormal assigns them independent N(loc_q, scale_q) guides.

Dependency (not in main requirements):
    pip install pyro-ppl>=1.9

Usage (standalone):
    python goalie_model/vi_model.py --out model_output --epochs 20000

Or import and call programmatically:
    from goalie_model.vi_model import fit_svi, posterior_samples
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO, Predictive
    from pyro.infer.autoguide import AutoNormal
    from pyro.optim import ClippedAdam
except ImportError:
    sys.exit(
        "\n[vi_model] pyro-ppl is required.\n"
        "Install with:  pip install pyro-ppl>=1.9\n"
    )

from .config import ModelConfig
from .data_prep import ModelData

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pyro model (non-centred parameterisation)
# ─────────────────────────────────────────────────────────────────────────────

def _make_model_fn(data: ModelData, cfg: ModelConfig):
    """
    Return a Pyro model function closed over data and config.

    All tensor indices and observed data are captured at construction time.
    The non-centred parameterisation whitens the random-walk skills so that
    AutoNormal can assign reasonable guides without manual tuning.
    """
    N_s       = data.n_shooters
    N_g       = data.n_goalies
    n_seasons = data.n_seasons
    max_T     = data.max_weeks or 1

    # Pre-extract tensors once; avoids re-building on every SVI step
    shooter_idx = data.shooter_idx
    goalie_idx  = data.goalie_idx
    season_idx  = data.season_idx
    week_idx    = data.week_idx
    xg_logit    = data.xg_logit
    y           = data.y.float()

    zeros_s_seas  = torch.zeros(N_s, n_seasons)
    zeros_g_seas  = torch.zeros(N_g, n_seasons)
    zeros_s_full  = torch.zeros(N_s, n_seasons, max_T)
    zeros_g_full  = torch.zeros(N_g, n_seasons, max_T)

    def model():
        # ── Positive scale hyperparameters ───────────────────────────────────
        sigma_mu_theta   = pyro.sample("sigma_mu_theta",   dist.HalfNormal(cfg.prior_sigma_mu))
        sigma_mu_phi     = pyro.sample("sigma_mu_phi",     dist.HalfNormal(cfg.prior_sigma_mu))
        sigma_seas_theta = pyro.sample("sigma_seas_theta", dist.HalfNormal(cfg.prior_sigma_season))
        sigma_seas_phi   = pyro.sample("sigma_seas_phi",   dist.HalfNormal(cfg.prior_sigma_season))
        tau_theta        = pyro.sample("tau_theta",        dist.HalfNormal(cfg.prior_tau))
        tau_phi          = pyro.sample("tau_phi",          dist.HalfNormal(cfg.prior_tau))

        # ── Global scalars ────────────────────────────────────────────────────
        beta0 = pyro.sample("beta0", dist.Normal(0.0, cfg.prior_sigma_beta0))
        alpha = pyro.sample("alpha", dist.Normal(cfg.prior_mean_alpha, cfg.prior_sigma_alpha))

        # ── Career baselines ──────────────────────────────────────────────────
        mu_theta = pyro.sample(
            "mu_theta",
            dist.Normal(torch.zeros(N_s), sigma_mu_theta.expand(N_s)).to_event(1),
        )
        mu_phi = pyro.sample(
            "mu_phi",
            dist.Normal(torch.zeros(N_g), sigma_mu_phi.expand(N_g)).to_event(1),
        )

        # ── Non-centred shooter random walk ───────────────────────────────────
        # eps0_theta[p, s] — season-start deviation (unit Normal)
        eps0_theta = pyro.sample(
            "eps0_theta",
            dist.Normal(zeros_s_seas, 1.0).to_event(2),
        )
        # delta_theta[p, s, t] — weekly step (unit Normal)
        delta_theta = pyro.sample(
            "delta_theta",
            dist.Normal(zeros_s_full, 1.0).to_event(3),
        )
        # Reconstruct centred skill:
        #   theta[p, s, t] = mu_theta[p] + sigma_season * eps0[p,s]
        #                    + tau * sum_{k=0}^{t} delta[p,s,k]
        theta_start = mu_theta.unsqueeze(1) + sigma_seas_theta * eps0_theta  # [N_s, S]
        theta = theta_start.unsqueeze(2) + tau_theta * torch.cumsum(delta_theta, dim=2)

        # ── Non-centred goalie random walk ────────────────────────────────────
        eps0_phi = pyro.sample(
            "eps0_phi",
            dist.Normal(zeros_g_seas, 1.0).to_event(2),
        )
        delta_phi = pyro.sample(
            "delta_phi",
            dist.Normal(zeros_g_full, 1.0).to_event(3),
        )
        phi_start = mu_phi.unsqueeze(1) + sigma_seas_phi * eps0_phi
        phi = phi_start.unsqueeze(2) + tau_phi * torch.cumsum(delta_phi, dim=2)

        # ── Bernoulli likelihood ──────────────────────────────────────────────
        theta_i = theta[shooter_idx, season_idx, week_idx]
        phi_i   = phi[goalie_idx,   season_idx, week_idx]
        logit_p = beta0 + alpha * xg_logit + theta_i - phi_i

        with pyro.plate("shots", y.shape[0]):
            pyro.sample("y", dist.Bernoulli(logits=logit_p), obs=y)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_svi(
    data: ModelData,
    cfg: ModelConfig,
    n_epochs: int = 20_000,
    lr: float = 0.005,
    log_every: int = 1_000,
) -> Tuple[object, object, List[float]]:
    """
    Fit the IRT model via Pyro SVI with an AutoNormal mean-field guide.

    Parameters
    ----------
    data       : ModelData from goalie_model.data_prep
    cfg        : ModelConfig
    n_epochs   : Number of SVI steps (default 20 000; more → better convergence)
    lr         : Learning rate for ClippedAdam
    log_every  : Print ELBO every this many epochs

    Returns
    -------
    model_fn   : The Pyro model function (needed for Predictive)
    guide      : Fitted AutoNormal guide
    elbo_history : list of -ELBO values (lower is better)
    """
    pyro.clear_param_store()
    model_fn = _make_model_fn(data, cfg)
    guide = AutoNormal(model_fn)

    optimizer = ClippedAdam({"lr": lr, "clip_norm": 5.0})
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO(num_particles=1))

    # Initialise guide parameters
    loss = float(svi.step())
    history = [loss]

    log.info("SVI started — %d epochs, lr=%.4f", n_epochs, lr)
    for epoch in range(1, n_epochs + 1):
        loss = float(svi.step())
        history.append(loss)
        if epoch % log_every == 0:
            log.info(
                "  SVI epoch %5d / %d  -ELBO = %-.0f",
                epoch, n_epochs, loss,
            )

    log.info("SVI done.  Final -ELBO = %.0f", history[-1])
    return model_fn, guide, history


# ─────────────────────────────────────────────────────────────────────────────
# Posterior sampling
# ─────────────────────────────────────────────────────────────────────────────

def posterior_samples(
    model_fn,
    guide,
    n_samples: int = 200,
    return_sites: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Draw n_samples from the fitted variational posterior.

    Returns a dict mapping parameter name → numpy array of shape
    [n_samples, *param_shape].  By default returns the scalar / vector
    parameters of interest for comparison with MAP; pass return_sites to
    customise.

    Default sites: tau_theta, tau_phi, sigma_seas_theta, sigma_seas_phi,
                   sigma_mu_theta, sigma_mu_phi, beta0, alpha,
                   mu_theta, mu_phi
    """
    if return_sites is None:
        return_sites = [
            "tau_theta", "tau_phi",
            "sigma_seas_theta", "sigma_seas_phi",
            "sigma_mu_theta", "sigma_mu_phi",
            "beta0", "alpha",
            "mu_theta", "mu_phi",
        ]
    predictive = Predictive(
        model_fn, guide=guide, num_samples=n_samples, return_sites=return_sites
    )
    with torch.no_grad():
        raw = predictive()
    return {k: v.squeeze().detach().cpu().numpy() for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_vi_state(
    guide,
    elbo_history: List[float],
    cfg: ModelConfig,
    path: str,
) -> None:
    """Save the Pyro param store and ELBO history to a pickle file."""
    param_snap = {
        name: val.detach().cpu().numpy()
        for name, val in pyro.get_param_store().items()
    }
    state = {
        "param_snap": param_snap,
        "elbo_history": elbo_history,
        "config": cfg,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=4)
    log.info("VI state saved to %s", path)


def load_vi_state(path: str) -> dict:
    """Load VI state saved by save_vi_state."""
    with open(path, "rb") as f:
        return pickle.load(f)


def restore_guide(
    data: ModelData,
    cfg: ModelConfig,
    vi_state: dict,
):
    """
    Restore a fitted AutoNormal guide from a saved state dict.

    Returns (model_fn, guide) ready for posterior_samples().
    """
    pyro.clear_param_store()
    model_fn = _make_model_fn(data, cfg)
    guide = AutoNormal(model_fn)

    # Initialise guide (required before loading params)
    with torch.no_grad():
        guide()

    # Restore param store values
    for name, val in vi_state["param_snap"].items():
        if name in pyro.get_param_store():
            pyro.get_param_store()[name].data.copy_(torch.tensor(val))

    return model_fn, guide


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_data(prev_csv: str, curr_csv: str, cfg: ModelConfig) -> Tuple[ModelData, dict, dict]:
    """Load CSVs and prepare ModelData (mirrors run_model.py logic)."""
    from .data_prep import prepare_shot_data

    frames = []
    if os.path.exists(prev_csv):
        log.info("Loading previous season: %s", prev_csv)
        frames.append(pd.read_csv(prev_csv))
    if os.path.exists(curr_csv):
        log.info("Loading current season: %s", curr_csv)
        frames.append(pd.read_csv(curr_csv))
    if not frames:
        raise FileNotFoundError("No CSV data found.")

    combined = pd.concat(frames, ignore_index=True)
    data, shooter_map, goalie_map = prepare_shot_data(combined, cfg)
    return data, shooter_map, goalie_map


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_PREV = os.path.join(ROOT, "ingest_scripts", "nhl_pbp_2024_2025_with_xg.csv")
    DEFAULT_CURR = os.path.join(ROOT, "ingest_scripts", "nhl_pbp_2025_2026_with_xg.csv")
    DEFAULT_OUT  = os.path.join(ROOT, "model_output")

    p = argparse.ArgumentParser(description="Fit IRT model via Pyro SVI.")
    p.add_argument("--prev",   default=DEFAULT_PREV)
    p.add_argument("--curr",   default=DEFAULT_CURR)
    p.add_argument("--out",    default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=20_000)
    p.add_argument("--lr",     type=float, default=0.005)
    args = p.parse_args()

    cfg = ModelConfig(
        seasons=[2024, 2025],
        require_prev_season=False,
        output_dir=args.out,
    )

    data, shooter_map, goalie_map = _build_data(args.prev, args.curr, cfg)
    log.info("Data loaded: %s", data)

    model_fn, guide, history = fit_svi(data, cfg, n_epochs=args.epochs, lr=args.lr)

    out_path = os.path.join(args.out, "vi_state.pkl")
    save_vi_state(guide, history, cfg, out_path)
    log.info("Done.  VI state → %s", out_path)

    # Quick summary of variance parameters
    samples = posterior_samples(model_fn, guide, n_samples=300)
    for name in ["tau_theta", "tau_phi", "sigma_seas_theta", "sigma_seas_phi"]:
        vals = samples[name]
        log.info(
            "  %-22s  mean=%.4f  sd=%.4f  [2.5%%=%.4f  97.5%%=%.4f]",
            name, vals.mean(), vals.std(),
            np.percentile(vals, 2.5), np.percentile(vals, 97.5),
        )


if __name__ == "__main__":
    main()
