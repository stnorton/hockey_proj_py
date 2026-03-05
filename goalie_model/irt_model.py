"""
irt_model.py
------------
PyTorch implementation of the Dynamic Goalie-Shooter IRT model.

The model
---------
For each shot i:
    logit(p_i) = β₀ + α·xg_logit_i + θ[s_i, p_i, t_i] − φ[s_i, q_i, t_i]
    y_i ~ Bernoulli(p_i)

Parameters
----------
β₀, α           — global intercept and xG calibration (scalars)
μ_θ[p]          — shooter career-average baseline  [N_shooters]
μ_φ[q]          — goalie career-average baseline   [N_goalies]
θ[p, s, t]      — shooter latent finishing skill   [N_shooters, S, max_T]
φ[q, s, t]      — goalie shot-stopping skill       [N_goalies,  S, max_T]
                  (positive φ reduces goal probability)
log_σ_*         — log of six positive scale hyper-parameters (see ModelConfig)

Scale parameters are stored in log-space to guarantee positivity.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_prep import ModelData

# ─────────────────────────────────────────────────────────────────────────────
# Numerical constants
# ─────────────────────────────────────────────────────────────────────────────
_LOG2PI_HALF = 0.5 * math.log(2.0 * math.pi)   # 0.5 ln(2π)
_LOG2 = math.log(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Log-PDF helpers (include constants so the posterior is on a proper scale)
# ─────────────────────────────────────────────────────────────────────────────

def _normal_logpdf(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Gaussian log-PDF, summed over all elements of x."""
    var = sigma ** 2
    return (-0.5 * ((x - mu) ** 2) / var - sigma.log() - _LOG2PI_HALF).sum()


def _halfnormal_logpdf_unconstrained(
    log_sigma: torch.Tensor, prior_scale: float
) -> torch.Tensor:
    """
    Log-prior for a scale parameter stored as log(σ).
    = log HalfNormal(σ | prior_scale) + log(σ)   [Jacobian of exp transform]
    """
    sigma = log_sigma.exp()
    return (
        _LOG2
        - math.log(prior_scale)
        - _LOG2PI_HALF
        - sigma ** 2 / (2 * prior_scale ** 2)
        + log_sigma  # Jacobian
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class DynamicIRTModel(nn.Module):
    """
    Dynamic IRT model for goalie shot-stopping and shooter finishing skill.

    Parameters are created to match the shape of the supplied ModelData.
    When the player roster grows (new season data ingested), call
    ``expand_for_new_data(new_data)`` to safely extend parameter tensors.

    Parameters
    ----------
    data : ModelData
        Fully-built model data (shapes only; tensors are not stored here).
    config : ModelConfig
    """

    def __init__(self, data: ModelData, config) -> None:
        super().__init__()

        self.n_shooters = data.n_shooters
        self.n_goalies = data.n_goalies
        self.n_seasons = data.n_seasons
        self.max_weeks = data.max_weeks or 1
        self.weeks_per_season = list(data.weeks_per_season)
        self.config = config

        # ── Global scalars ────────────────────────────────────────────────────
        self.beta0 = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.ones(1))

        # ── Career-average baselines ──────────────────────────────────────────
        self.mu_theta = nn.Parameter(torch.zeros(data.n_shooters))
        self.mu_phi = nn.Parameter(torch.zeros(data.n_goalies))

        # ── Dynamic skills: [players, seasons, weeks] ─────────────────────────
        self.theta = nn.Parameter(
            torch.zeros(data.n_shooters, data.n_seasons, self.max_weeks)
        )
        self.phi = nn.Parameter(
            torch.zeros(data.n_goalies, data.n_seasons, self.max_weeks)
        )

        # ── Log-scale hyper-parameters (log so σ > 0 always) ─────────────────
        # Initialised at the config's prior mode value.
        self.log_sigma_mu_theta = nn.Parameter(
            torch.tensor(math.log(config.prior_sigma_mu))
        )
        self.log_sigma_mu_phi = nn.Parameter(
            torch.tensor(math.log(config.prior_sigma_mu))
        )
        self.log_sigma_season_theta = nn.Parameter(
            torch.tensor(math.log(config.prior_sigma_season))
        )
        self.log_sigma_season_phi = nn.Parameter(
            torch.tensor(math.log(config.prior_sigma_season))
        )
        self.log_tau_theta = nn.Parameter(
            torch.tensor(math.log(config.prior_tau))
        )
        self.log_tau_phi = nn.Parameter(
            torch.tensor(math.log(config.prior_tau))
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass helper
    # ─────────────────────────────────────────────────────────────────────────

    def predict_logit(self, data: ModelData) -> torch.Tensor:
        """
        Return logit(p_i) for every shot in data.
        Shape: [N_shots]
        """
        # Gather per-shot skill values
        # theta/phi shape: [players, seasons, weeks]
        theta_i = self.theta[data.shooter_idx, data.season_idx, data.week_idx]
        phi_i = self.phi[data.goalie_idx, data.season_idx, data.week_idx]

        xg = data.xg_logit.to(theta_i.device)
        return self.beta0 + self.alpha * xg + theta_i - phi_i

    # ─────────────────────────────────────────────────────────────────────────
    # Log-likelihood
    # ─────────────────────────────────────────────────────────────────────────

    def _log_likelihood(
        self,
        data: ModelData,
        shot_indices: Optional[torch.Tensor] = None,
        n_total: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Bernoulli log-likelihood (summed, not averaged).

        If shot_indices is provided, compute on the mini-batch and scale
        up by (n_total / batch_size) to approximate the full-data likelihood.
        """
        logit_p = self.predict_logit(data)

        if shot_indices is not None:
            logit_p = logit_p[shot_indices]
            y = data.y[shot_indices].to(logit_p.device)
            scale = float(n_total) / float(len(shot_indices))
        else:
            y = data.y.to(logit_p.device)
            scale = 1.0

        # binary_cross_entropy_with_logits is numerically stable;
        # negate to get log-likelihood.
        nll = F.binary_cross_entropy_with_logits(logit_p, y, reduction="sum")
        return -nll * scale

    # ─────────────────────────────────────────────────────────────────────────
    # Log-prior
    # ─────────────────────────────────────────────────────────────────────────

    def _log_prior(self) -> torch.Tensor:
        """
        Sum of all log-prior terms.
        """
        cfg = self.config
        lp = torch.tensor(0.0)

        # ── Scale hyper-parameter HalfNormal priors ───────────────────────────
        lp = lp + _halfnormal_logpdf_unconstrained(self.log_sigma_mu_theta, cfg.prior_sigma_mu)
        lp = lp + _halfnormal_logpdf_unconstrained(self.log_sigma_mu_phi,   cfg.prior_sigma_mu)
        lp = lp + _halfnormal_logpdf_unconstrained(self.log_sigma_season_theta, cfg.prior_sigma_season)
        lp = lp + _halfnormal_logpdf_unconstrained(self.log_sigma_season_phi,   cfg.prior_sigma_season)
        lp = lp + _halfnormal_logpdf_unconstrained(self.log_tau_theta, cfg.prior_tau)
        lp = lp + _halfnormal_logpdf_unconstrained(self.log_tau_phi,   cfg.prior_tau)

        # Resolved positive scales (clamped to prevent MAP collapse to near-zero)
        sig_mu_theta    = self.log_sigma_mu_theta.exp()
        sig_mu_phi      = self.log_sigma_mu_phi.exp()
        sig_seas_theta  = self.log_sigma_season_theta.exp().clamp(min=cfg.sigma_season_min)
        sig_seas_phi    = self.log_sigma_season_phi.exp().clamp(min=cfg.sigma_season_min)
        tau_theta       = self.log_tau_theta.exp().clamp(min=cfg.tau_min)
        tau_phi         = self.log_tau_phi.exp().clamp(min=cfg.tau_min)

        # ── β₀ and α priors ───────────────────────────────────────────────────
        lp = lp + _normal_logpdf(self.beta0, torch.zeros_like(self.beta0),
                                 torch.tensor(cfg.prior_sigma_beta0))
        lp = lp + _normal_logpdf(self.alpha, torch.tensor(cfg.prior_mean_alpha),
                                 torch.tensor(cfg.prior_sigma_alpha))

        # ── Career-average baselines: N(0, σ_μ) ──────────────────────────────
        lp = lp + _normal_logpdf(self.mu_theta, torch.zeros_like(self.mu_theta), sig_mu_theta)
        lp = lp + _normal_logpdf(self.mu_phi,   torch.zeros_like(self.mu_phi),   sig_mu_phi)

        # ── Season-start prior: θ[p,s,0] ~ N(μ_θ[p], σ_season) ──────────────
        for s in range(self.n_seasons):
            lp = lp + _normal_logpdf(
                self.theta[:, s, 0],
                self.mu_theta,
                sig_seas_theta,
            )
            lp = lp + _normal_logpdf(
                self.phi[:, s, 0],
                self.mu_phi,
                sig_seas_phi,
            )

        # ── Weekly random-walk prior: θ[p,s,t] ~ N(θ[p,s,t-1], τ_θ) ─────────
        for s in range(self.n_seasons):
            T = self.weeks_per_season[s]
            if T < 2:
                continue
            diff_theta = self.theta[:, s, 1:T] - self.theta[:, s, 0:T - 1]
            lp = lp + _normal_logpdf(diff_theta, torch.zeros_like(diff_theta), tau_theta)

            diff_phi = self.phi[:, s, 1:T] - self.phi[:, s, 0:T - 1]
            lp = lp + _normal_logpdf(diff_phi, torch.zeros_like(diff_phi), tau_phi)

        return lp

    # ─────────────────────────────────────────────────────────────────────────
    # Soft sum-to-zero constraint
    # ─────────────────────────────────────────────────────────────────────────

    def _sum_to_zero_penalty(self) -> torch.Tensor:
        """
        Penalise non-zero weekly cross-player means.
        Penalty = w · Σ_{s,t} [ mean_p(θ[p,s,t])² + mean_q(φ[q,s,t])² ]
        """
        if self.config.sum_to_zero_weight == 0.0:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0)
        w = self.config.sum_to_zero_weight
        for s in range(self.n_seasons):
            T = self.weeks_per_season[s]
            if T == 0:
                continue
            theta_means = self.theta[:, s, :T].mean(dim=0)  # [T] — mean, not sum
            phi_means = self.phi[:, s, :T].mean(dim=0)      # [T] — mean, not sum
            penalty = penalty + w * (theta_means ** 2).sum()
            penalty = penalty + w * (phi_means ** 2).sum()
        return penalty

    # ─────────────────────────────────────────────────────────────────────────
    # Full MAP objective  (negative log-posterior)
    # ─────────────────────────────────────────────────────────────────────────

    def loss(
        self,
        data: ModelData,
        shot_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Negative log-posterior (to be minimised).

        loss = −log_likelihood − log_prior + sum_to_zero_penalty

        Parameters
        ----------
        data : ModelData
        shot_indices : optional 1-D long tensor
            Mini-batch indices into data shots.
        """
        n_total = data.n_shots
        ll = self._log_likelihood(data, shot_indices, n_total)
        lp = self._log_prior()
        penalty = self._sum_to_zero_penalty()
        return -(ll + lp) + penalty

    # ─────────────────────────────────────────────────────────────────────────
    # Expand tensors when new players appear (weekly updates)
    # ─────────────────────────────────────────────────────────────────────────

    def expand_for_new_data(self, new_data: ModelData) -> None:
        """
        Extend θ, φ, μ_θ, μ_φ tensors when new_data has more players or weeks
        than the model was originally built for.  Existing parameters are
        preserved; new entries are initialised to zero.

        Called automatically by fit.update_map_with_new_week when needed.
        """
        n_new_shooters = new_data.n_shooters
        n_new_goalies = new_data.n_goalies
        new_max_weeks = new_data.max_weeks

        # ── Expand shooter tensors if needed ──────────────────────────────────
        if n_new_shooters > self.n_shooters:
            extra = n_new_shooters - self.n_shooters
            self.mu_theta = nn.Parameter(
                torch.cat([self.mu_theta.data,
                           torch.zeros(extra, device=self.mu_theta.device)], dim=0)
            )
            self.theta = nn.Parameter(
                torch.cat([self.theta.data,
                           torch.zeros(extra, self.n_seasons, self.max_weeks,
                                       device=self.theta.device)], dim=0)
            )
            self.n_shooters = n_new_shooters

        # ── Expand goalie tensors if needed ───────────────────────────────────
        if n_new_goalies > self.n_goalies:
            extra = n_new_goalies - self.n_goalies
            self.mu_phi = nn.Parameter(
                torch.cat([self.mu_phi.data,
                           torch.zeros(extra, device=self.mu_phi.device)], dim=0)
            )
            self.phi = nn.Parameter(
                torch.cat([self.phi.data,
                           torch.zeros(extra, self.n_seasons, self.max_weeks,
                                       device=self.phi.device)], dim=0)
            )
            self.n_goalies = n_new_goalies

        # ── Expand week dimension if new season has more weeks ─────────────────
        if new_max_weeks > self.max_weeks:
            extra_weeks = new_max_weeks - self.max_weeks
            dev = self.theta.device
            self.theta = nn.Parameter(
                torch.cat([self.theta.data,
                           torch.zeros(self.n_shooters, self.n_seasons, extra_weeks,
                                       device=dev)], dim=2)
            )
            self.phi = nn.Parameter(
                torch.cat([self.phi.data,
                           torch.zeros(self.n_goalies, self.n_seasons, extra_weeks,
                                       device=dev)], dim=2)
            )
            self.max_weeks = new_max_weeks

        # Keep weeks-per-season metadata in sync
        self.weeks_per_season = list(new_data.weeks_per_season)

    # ─────────────────────────────────────────────────────────────────────────
    # Serialisation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def state_dict_numpy(self) -> dict:
        """Return all parameters as a dict of numpy arrays (for pickling)."""
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}

    def load_state_dict_numpy(self, numpy_dict: dict) -> None:
        """Load a numpy state dict produced by state_dict_numpy()."""
        tensor_dict = {k: torch.tensor(v) for k, v in numpy_dict.items()}
        self.load_state_dict(tensor_dict)

    # ─────────────────────────────────────────────────────────────────────────
    # Skill retrieval (detached, ready for reporting)
    # ─────────────────────────────────────────────────────────────────────────

    def get_shooter_skills(
        self, season_idx: int = 1, center: Optional[bool] = None
    ) -> np.ndarray:
        """
        Return θ[p, season_idx, :] as a numpy array of shape
        [N_shooters, T_s].

        Parameters
        ----------
        season_idx : 0=previous, 1=current
        center : subtract weekly mean? Defaults to config.center_weekly.
        """
        if center is None:
            center = self.config.center_weekly
        T = self.weeks_per_season[season_idx]
        arr = self.theta[:, season_idx, :T].detach().cpu().numpy()
        if center and T > 0:
            arr = arr - arr.mean(axis=0, keepdims=True)
        return arr

    def get_goalie_skills(
        self, season_idx: int = 1, center: Optional[bool] = None
    ) -> np.ndarray:
        """
        Return φ[q, season_idx, :] as a numpy array of shape
        [N_goalies, T_s].
        """
        if center is None:
            center = self.config.center_weekly
        T = self.weeks_per_season[season_idx]
        arr = self.phi[:, season_idx, :T].detach().cpu().numpy()
        if center and T > 0:
            arr = arr - arr.mean(axis=0, keepdims=True)
        return arr
