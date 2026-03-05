# IRT Model & Dashboard — Code Review and Improvement Plan

## Executive Summary

A thorough review of the Dynamic Goalie-Shooter IRT model, data pipeline, and Streamlit dashboard uncovered several interconnected issues. The most critical is **variance component collapse during MAP estimation**, which renders the "dynamic" weekly skills completely static. This cascading failure affects the McDavid ranking, the trajectory visualizations, and the interpretability of summary statistics.

---

## Table of Contents

1. [Critical: Variance Component Collapse (τ → 0)](#1-critical-variance-component-collapse)
2. [Critical: xG Miscalibration Inflates FSAx_raw](#2-critical-xg-miscalibration)
3. [High: McDavid Ranking — Root Cause Analysis](#3-high-mcdavid-ranking)
4. [High: Flat Trajectory Display](#4-high-flat-trajectory-display)
5. [Medium: Sum-to-Zero Penalty Uses sum() Not mean()](#5-medium-sum-to-zero-penalty)
6. [Medium: Confidence Bands Are Heuristic, Not Posterior](#6-medium-confidence-bands)
7. [Medium: No Strength-State Stratification](#7-medium-strength-state-stratification)
8. [Medium: Optimizer Didn't Converge](#8-medium-optimizer-convergence)
9. [Low: Dashboard Explanation & Framing](#9-low-dashboard-explanation)
10. [Proposed Explanations for Technical and Non-Technical Audiences](#10-audience-explanations)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Implementation Instructions for Sonnet](#12-implementation-instructions)

---

## 1. Critical: Variance Component Collapse

### What's happening

The MAP-estimated random-walk standard deviations have collapsed to near-zero:

| Parameter | Prior | Learned Value | Ratio |
|---|---|---|---|
| `tau_theta` (shooter weekly walk SD) | 0.2 | **5.3 × 10⁻⁷** | 0.0000027× |
| `tau_phi` (goalie weekly walk SD) | 0.2 | **9.3 × 10⁻⁵** | 0.00047× |
| `sigma_season_theta` (season start SD) | 0.5 | **1.6 × 10⁻⁴** | 0.00031× |
| `sigma_season_phi` (season start SD) | 0.5 | **1.9 × 10⁻⁴** | 0.00038× |

**Consequence**: θ and φ are identical across all weeks within a season. The weekly trajectory plots show flat lines. The "Dynamic" in "Dynamic IRT" is non-functional.

**Measured weekly ranges** (across 23 weeks of the 2025-26 season):
- Shooter θ: median per-player range = **0.0000017** (effectively zero)
- Goalie φ: median per-player range = **0.0000029** (effectively zero)

### Root cause

MAP estimation (posterior mode) is well-known to collapse variance components in hierarchical models. The posterior mode for a variance σ² with few observations per group can be at or near zero — the HalfNormal(0.2) prior is too weak to prevent this at the MAP.

### Recommended fixes (in priority order)

**Fix A — Clamp τ with a lower bound (simplest, do first)**:
```python
# In config.py, add:
tau_min: float = 0.01  # prevent collapse below this value

# In irt_model.py _log_prior(), after computing tau_theta/tau_phi:
tau_theta = torch.clamp(self.log_tau_theta.exp(), min=self.config.tau_min)
tau_phi   = torch.clamp(self.log_tau_phi.exp(),   min=self.config.tau_min)
```

**Fix B — Fix τ at a reasonable value (don't learn it)**:
Remove `log_tau_theta` and `log_tau_phi` as learnable parameters. Instead, use the config's `prior_tau` directly as a fixed constant. This is a standard approach when MAP can't reliably learn variance components.

**Fix C — Switch to variational inference or MCMC (longer-term)**:
Full Bayesian inference (e.g., NumPyro NUTS or Pyro SVI) properly integrates over uncertainty in variance parameters, avoiding mode collapse. This is the theoretically correct solution but requires significant refactoring.

---

## 2. Critical: xG Miscalibration

### What's happening

The Statsyuk xG model consistently over-predicts goal probability:

| Season | Mean xG | Actual Goal Rate | Over-prediction |
|---|---|---|---|
| 2024-25 | 6.62% | 5.17% | +28% |
| 2025-26 | 7.09% | 5.39% | +32% |

The IRT model partially compensates via `β₀ = -0.238` (shifts predictions downward) and `α = 0.978` (slightly shrinks xG). But the correction is imperfect — the IRT baseline (θ=φ=0) predicts 4.49% goal probability for an average shot, slightly undershooting the actual 5.3%.

### Impact on metrics

- **FSAx_raw** (goals − xG) is systematically negative for most players because xG is too high
- This makes "finishing above expected" look harder than it is
- Players who are average finishers appear below average in FSAx_raw

### Recommended fix

Add an explicit xG recalibration step in `data_prep.py` before the model sees the data. Fit a simple logistic regression or Platt scaling: `calibrated_xG = sigmoid(a · logit(xG) + b)` on a held-out set, or use isotonic regression. Store the calibration parameters and apply them before computing xG_logit.

Alternatively, since the IRT model already learns β₀ and α to recalibrate, the main change needed is to **use the model-recalibrated xG (not raw xG) when computing FSAx_raw and GSAx_raw** in the dashboard. Currently:
- `fsax_raw = goals_actual - goals_xg_only` uses raw xG sums
- This should instead use the IRT-recalibrated baseline: `goals_baseline = Σ sigmoid(β₀ + α · xg_logit_i)`

---

## 3. High: McDavid Ranking

### The finding

Connor McDavid ranks as one of the worst shooters by FSAx:
- 339 model shots, 32 goals, 40.37 raw xG → **FSAx_raw = -8.37**
- mu_theta = -0.032, FSAx (IRT) = -0.90
- At 5v5: 256 shots, 17 goals, 25.23 xG → converting at 67% of raw xG

### Is this correct?

**Partially**. McDavid genuinely under-performs his raw xG. His genius is creating high-quality chances (speed, positioning, playmaking), not necessarily finishing at a superhuman rate. This is a noted finding in hockey analytics.

**But the ranking is too extreme** because:

1. **Raw xG is miscalibrated high** (see §2). If McDavid's 40.37 xG were recalibrated to ~34.3 (using the 28-32% over-prediction), his FSAx_raw would drop from -8.37 to roughly -2.3 — still below average but not catastrophic.

2. **No strength-state stratification** (see §7). McDavid takes ~100 power play shots per season. PP dynamics are fundamentally different from 5v5, and pooling them biases the estimates.

3. **Empty-net shot exclusion affects him**. 68 of his 407 total shots had no goalie (empty net). These were removed from the model, taking 4 goals with them. This is correct methodology, but the dashboard should note this somewhere.

4. **Opponent quality adjustment is implicit, not explicit**. The model adjusts for goalie quality via φ, but with φ also nearly static, this adjustment is limited.

### Recommendations

- Recalibrate raw xG before computing FSAx_raw (see §2)
- Add even-strength-only metrics alongside all-situations metrics
- Add a note in the dashboard explaining what FSAx measures and why elite playmakers can have negative FSAx
- Consider adding "shot volume" or "chance creation" metrics so McDavid's offensive contribution is visible

---

## 4. High: Flat Trajectory Display

### Problem

The weekly trajectory charts show effectively flat lines because τ ≈ 0 makes skills constant. Even after fixing the tau collapse, the dashboard display needs improvement.

### Current state

- Draisaitl θ (best shooter): ranges from 0.191574 to 0.191626 across 23 weeks (Δ = 0.00005)
- Vasilevskiy φ: ranges from 0.132252 to 0.132398 (Δ = 0.00015)

### Recommendations after fixing tau

**A) Show cumulative/rolling skill instead of instantaneous**: Rather than plotting the raw weekly θ[p,s,t], show a rolling-window view — e.g., 4-week rolling goal rate vs rolling xG rate — overlaid on the IRT skill line. This gives users context about what's driving the skill estimate.

**B) Show skill relative to percentiles**: Instead of raw φ/θ on the logit scale, convert to percentile rank among peers (e.g., "this goalie was in the 85th percentile this week"). Percentiles are more intuitive than logit-scale values.

**C) Convert to goals-per-60 or save-percentage scale**: Transform the logit-scale skill to a more hockey-interpretable unit. For goalies: "what save percentage would this goalie have on league-average shots?" For shooters: "what shooting percentage above/below average does this imply?"

**D) Add data density indicators**: The current `shots_that_week` tooltip is helpful, but weeks with 0 shots should be visually distinct (dimmed or dashed line). The confidence bands (once fixed) should widen dramatically for low-shot weeks.

---

## 5. Medium: Sum-to-Zero Penalty Uses sum() Not mean()

### Code location
`irt_model.py`, `_sum_to_zero_penalty()` method.

### Issue

```python
theta_means = self.theta[:, s, :T].sum(dim=0)  # [T] — this is SUM, not MEAN
penalty = penalty + w * (theta_means ** 2).sum()
```

The penalty penalizes `(Σ_p θ[p,s,t])²`, which scales with N_players². With 1535 shooters, even small non-zero means create a large penalty. This should use `mean(dim=0)` to make the penalty scale-invariant:

```python
theta_means = self.theta[:, s, :T].mean(dim=0)  # Scale-invariant
```

In practice, this bug is masked because the penalty IS effectively keeping means at zero (the measured penalty value is ~0.000). But it may be contributing to making the optimizer push θ values smaller than they should be.

---

## 6. Medium: Confidence Bands Are Heuristic

### Code location
`build_data.py`, lines ~270-280.

### Issue

```python
tau = getattr(state.config, "prior_tau", 0.2)  # Uses PRIOR tau, not learned tau
band = (2 * tau / (df["shots_that_week"].clip(lower=1) ** 0.5)).round(4)
```

This is a rough heuristic, not a proper posterior uncertainty estimate. Problems:
1. Uses the prior value (0.2), not the learned tau (~5e-7). The bands are therefore 375,000× wider than the actual model uncertainty would suggest.
2. The `1/√n` scaling is an approximation that doesn't account for the prior, the random-walk structure, or correlations between weeks.
3. For weeks with 0 shots, it uses 1 in the denominator, giving a band of ±0.4, which may be reasonable by accident.

### Recommendations

**Short-term**: Use the learned tau value. But since learned tau is collapsed (see §1), fix tau first.

**Medium-term**: Compute proper approximate posterior uncertainty. For a MAP estimate with known Hessian, the posterior covariance is the inverse Hessian. PyTorch can compute the diagonal of the Hessian numerically for the θ and φ parameters.

**Long-term**: With full Bayesian inference, posterior credible intervals come naturally from MCMC samples.

---

## 7. Medium: No Strength-State Stratification

### Current behavior

`config.strength_states = None` → all strength states (5v5, PP, PK, 4v3, etc.) are pooled.

### Problem

- Power play shots have systematically different dynamics (more space, different defensive positioning)
- Mixing PP and EV data means the model conflates "plays on the PP a lot" with "is a good finisher"
- Players like McDavid with heavy PP minutes face a biased xG baseline

### Recommendations

**Option A (simplest)**: Default to even-strength only (`strength_states = ["EVEN"]`) for the primary metrics, with PP/PK shown separately.

**Option B**: Add a situation_code feature to the linear predictor: `logit(p) = β₀ + α·xg + β_pp·I(PP) + θ - φ`. This lets the model learn a PP intercept shift.

**Option C**: Fit separate θ_EV and θ_PP skills per player. This is most informative but doubles the parameter count.

---

## 8. Medium: Optimizer Didn't Converge

### Evidence

- Adam ran for all 2000 iterations (= `max_epochs_full`) without early stopping
- L-BFGS brought loss from -770,658 to -1,274,530 — a massive 65% improvement
- This suggests Adam alone was far from converged when it stopped

### Recommendations

1. Increase `max_epochs_full` to 5000-10000, or remove the cap and rely solely on early stopping
2. Try a warm-restart learning rate schedule or increase patience
3. Consider running L-BFGS for more steps (current: 20, try 100-200)
4. Log and check for gradient norm explosion/vanishing during training

---

## 9. Low: Dashboard Explanation & Framing

### Current gaps

- No "methodology" or "about" page explaining what the IRT model is
- Metric tooltips are minimal
- FSAx/GSAx definitions appear only in column headers, not in context
- The "Head-to-Head" page compares φ and θ on the same axis, but their scales differ (goalie φ std = 0.072, shooter θ std = 0.027)

### Recommended additions

1. Add an "About / Methodology" page to the dashboard sidebar
2. Add expandable info boxes (`st.expander`) on each page explaining the primary metric
3. The goalie trajectory page shows two charts (φ and μ_φ) with the same confidence bands — the second chart (μ_φ) is the career-average baseline and should NOT have weekly confidence bands
4. Add a "Data Notes" section mentioning: empty-net shots excluded, all strength states included, xG source

---

## 10. Proposed Explanations for Technical and Non-Technical Audiences

### For Non-Technical Audiences (fans, broadcasters, front-office analysts)

**GSAx (Goals Saved Above Expected)**:
> "GSAx measures how many more (or fewer) goals a goalie prevented compared to what an average goalie would have allowed on the same shots. A GSAx of +5 means the goalie saved 5 more goals than average. We adjust for the quality of shooters faced — a goalie facing elite scorers gets more credit."

**FSAx (Finishing Skill Above Expected)**:
> "FSAx measures how many more (or fewer) goals a shooter scored compared to what an average shooter would have scored from the same positions and situations. An FSAx of +3 means the shooter converted 3 more goals than average. We adjust for the quality of goalies faced."

**Important caveat for FSAx**:
> "FSAx measures *finishing* — converting shots into goals. It does NOT measure a player's overall offensive contribution. A player like McDavid creates far more chances than he finishes himself. His negative FSAx means he converts slightly fewer of his shots than expected given how good his shooting positions are — not that he's a bad offensive player."

**Skill trajectory**:
> "The skill line shows how a player's performance changes week to week. A rising line means the player is performing better; a flat line means consistent performance. The shaded area shows our confidence — wider shading means we're less certain (usually because the player faced fewer shots that week)."

### For Technical Audiences (data scientists, hockey analytics community)

**Model overview**:
> "We use a Dynamic Item Response Theory (IRT) model with Bernoulli likelihood. Each shot's goal probability is modeled as: `logit(p) = β₀ + α·logit(xG) + θ[shooter,week] − φ[goalie,week]`, where xG comes from a pre-trained XGBoost model (AUC 0.80), θ represents shooter finishing skill above xG, and φ represents goalie save skill. Skills evolve via a random-walk prior: `θ[t] ~ N(θ[t-1], τ²)`. Parameters are estimated via MAP optimization (Adam + L-BFGS) in PyTorch."

**Key assumptions and limitations**:
> - xG is treated as given (not jointly estimated)
> - Player skills are a single scalar — no shot-type-specific skills
> - All strength states pooled (no PP/EV/PK separation)
> - MAP estimation, not full Bayesian — uncertainty estimates are approximate
> - Weekly dynamics require adequate τ; current MAP collapses τ → 0 (see known issues)

**GSAx formula**:
> `GSAx(q) = Σ_shots [sigmoid(β₀ + α·xg + θ[shooter] − 0) − sigmoid(β₀ + α·xg + θ[shooter] − φ[goalie])]`
> = expected goals without goalie skill minus expected goals with goalie skill.
> Shooter adjustment: we compare the actual shooter θ values a goalie faced against the shot-weighted league-average θ.

**FSAx formula**:
> `FSAx(p) = Σ_shots [sigmoid(β₀ + α·xg + θ[shooter] − φ[goalie]) − sigmoid(β₀ + α·xg + 0 − φ[goalie])]`
> = expected goals with shooter skill minus expected goals without shooter skill.

---

## 11. Implementation Roadmap

### Phase 1 — Fix Critical Issues (do first)

| # | Task | Files | Effort |
|---|---|---|---|
| 1.1 | Fix tau collapse — clamp or fix tau at configurable minimum | `config.py`, `irt_model.py` | Small |
| 1.2 | Use model-recalibrated xG baseline for FSAx_raw / GSAx_raw | `build_data.py`, `summarize.py` | Small |
| 1.3 | Fix sum-to-zero penalty to use mean() not sum() | `irt_model.py` | Trivial |
| 1.4 | Increase max_epochs_full and L-BFGS steps | `config.py`, `run_model.py` defaults | Trivial |
| 1.5 | Re-run the model after fixes 1.1-1.4 | `run_model.py` | ~5-10 min |
| 1.6 | Rebuild dashboard data | `build_data.py` | ~2 min |

### Phase 2 — Improve Dashboard Display

| # | Task | Files | Effort |
|---|---|---|---|
| 2.1 | Default to even-strength only metrics (add strength_states config) | `run_model.py`, `config.py` | Small |
| 2.2 | Convert θ/φ to hockey-interpretable units (percentile or save%/shooting%) | `build_data.py`, `app.py` | Medium |
| 2.3 | Fix confidence bands to use learned tau | `build_data.py` | Small |
| 2.4 | Add "About/Methodology" page to dashboard | `app.py` | Medium |
| 2.5 | Fix goalie trajectory page (don't show weekly bands on μ_φ chart) | `app.py` | Small |
| 2.6 | Add shot volume / chance creation metric for shooters | `build_data.py`, `app.py` | Medium |
| 2.7 | Add explanatory text and caveats per §10 | `app.py` | Medium |

### Phase 3 — Structural Improvements (optional)

| # | Task | Files | Effort |
|---|---|---|---|
| 3.1 | Add PP intercept to linear predictor | `irt_model.py`, `data_prep.py` | Medium |
| 3.2 | Add explicit xG recalibration step | `data_prep.py` | Medium |
| 3.3 | Switch to variational inference (NumPyro/Pyro) | Major refactor | Large |

---

## 13. Post-Implementation Review (Session 3)

### What Was Fixed

| Issue | Status | Notes |
|---|---|---|
| 1.1 Tau collapse | ✅ Fixed | Clamped at `tau_min=0.01`. Learned tau settles at clamp (0.0091), confirming MAP wants to push it lower. |
| 1.3 sum→mean penalty | ✅ Fixed | |
| 1.4 Optimizer convergence | ✅ Fixed | 5000 Adam epochs + 20 L-BFGS steps. L-BFGS NaN fallback added. |
| 7. Strength-state stratification | ✅ Fixed | Three models: all, even (5v5), power play. Dashboard toggle. |
| 9. Dashboard explanation | ✅ Fixed | Methodology tab, per-page ℹ️ expanders, H2H rework. |
| Duplicate player names (Sebastian Aho) | ✅ Fixed | `_canonical_weekly()` deduplicates by shots in trajectory pages + H2H. |
| Missing player names in cache | ✅ Fixed | Manually resolved Korpisalo, Montembeault, Ellis, Miner. |

### Remaining / New Issues Found

#### Issue 13.1 — Confidence bands use `prior_tau` (0.2), not learned tau (~0.009)

**File**: `build_data.py`, line ~309  
**Code**: `tau = getattr(state.config, "prior_tau", 0.2)`  
**Impact**: Bands are ~22× wider than the model's actual uncertainty. With the clamped tau = 0.01, the bands should use 0.01 (the effective random-walk SD), not 0.2.  
**Fix**: Extract the learned tau from `state.param_dict` instead of using the prior. After clamping, the effective tau is `max(exp(log_tau), config.tau_min)`:
```python
import numpy as np
tau_theta = max(np.exp(state.param_dict["log_tau_theta"]), state.config.tau_min)
tau_phi   = max(np.exp(state.param_dict["log_tau_phi"]),   state.config.tau_min)
```

#### Issue 13.2 — H2H goals-per-100 formula is wrong

**File**: `app.py`, H2H page  
**Current code**:
```python
league_avg = g_sum["goals_actual"].sum() / g_sum["shots_faced"].sum()
base_logit = log(league_avg / (1 - league_avg))
p_goal     = σ(base_logit + θ_shooter − φ_goalie)
```
**Problem**: This uses a generic `logit(league_avg)` as the baseline, but the IRT model uses `β₀ + α·xG_logit`, not `logit(league_avg)`. Since θ and φ are offsets from the IRT intercept `β₀`, the correct formula for an "average shot" (xG at league mean) is:
```python
p_goal = σ(β₀ + α·logit(league_avg) + θ − φ)
```
Without β₀ and α, the estimate is biased.
**Fix**: Store `beta0` and `alpha` in `meta.json` during `build_data.py` so they're available to `app.py`.

#### Issue 13.3 — Leaderboard tables don't deduplicate by player name

The `_canonical_weekly()` fix only applies to trajectory charts and H2H. The leaderboard tables (`page_goalie_leaderboard`, `page_shooter_leaderboard`) render directly from summary CSVs which can have two rows for "Sebastian Aho" with different stats. Users will see two rows and be confused.
**Fix**: In `build_data.py`, when building `shooter_summary`, deduplicate names by keeping the row with the most shots.

#### Issue 13.4 — H2H normalisation uses season-end skills only for SD

**File**: `app.py`, H2H page  
The percentile normalisation computes SD from the distribution of players' final-week skill only. This is valid for the headline metric but creates a distortion when applied to the weekly trajectory bands: early weeks are normalised against a distribution that includes late-season evolution, making early-season values appear artificially compressed.
**Impact**: Minor. The normalisation is "close enough" since tau is small.

#### Issue 13.5 — `sv_pct` and `xgsv_pct` in goalie table are misleading

**File**: `build_data.py`, `app.py`  
Raw save percentage and expected save percentage (from xG) are included in the goalie leaderboard. These are not IRT-adjusted and can mislead users into comparing them directly with the IRT metrics. Since the dashboard's purpose is to present IRT-adjusted metrics, these raw stats create confusion.
**Fix**: Remove from `GOALIE_COLS` display; keep in data for reference but don't surface in the leaderboard table.

#### Issue 13.6 — Sort-by dropdown includes `sv_pct` for goalies

If sv_pct is removed from the table, it should also be removed from the `sort_col` selectbox options.

### Updated Roadmap

| # | Task | Status |
|---|---|---|
| 13.1 | Fix confidence bands → use learned tau | ✅ Fixed — `build_data.py` now extracts `log_tau_phi`/`log_tau_theta` from `state.param_dict` |
| 13.2 | Fix H2H goals-per-100 → include β₀, α | ✅ Fixed — β₀, α, mean_xg_logit stored in `meta.json`; H2H uses IRT baseline |
| 13.3 | Deduplicate leaderboard summary tables | Open (minor) |
| 13.4 | Normalisation SD from season-end | Acceptable (minor) |
| 13.5 | Remove sv_pct/xgsv_pct from goalie display | ✅ Fixed — removed from `GOALIE_COLS` |
| 13.6 | Remove sv_pct from goalie sort options | ✅ Fixed — removed from sort selectbox |
| 2.4 | Add Methodology tab to dashboard | ✅ Fixed — 📖 Methodology page with non-technical + technical sections |
| 2.7 | Add per-page documentation bullets | ✅ Fixed — ℹ️ expander on each page |
| — | Leaderboard overall/team rank columns | ✅ Fixed — Overall Rank + Team Rank when team filter active |

### Memory / Skill Files to Create

Create a repo-scoped memory file at `/memories/repo/hockey_irt_model.md` with:

```
# Hockey IRT Model — Key Facts

## Architecture
- PyTorch MAP model in goalie_model/ package
- Entry point: run_model.py (fit) → dashboard/build_data.py (export) → dashboard/app.py (display)
- Config: goalie_model/config.py (ModelConfig dataclass)
- Model: goalie_model/irt_model.py (DynamicIRTModel)
- Fitting: goalie_model/fit.py (fit_full_map, update_map_with_new_week)
- Summaries: goalie_model/summarize.py (summarize_skills, compute_gsax)
- Dashboard data: dashboard/build_data.py (compute_fsax, main)

## Known Issues (from review)
- tau_theta and tau_phi collapse to ~0 during MAP → weekly skills are static
- sigma_season_theta/phi also collapse → no season-to-season variation
- xG model over-predicts by ~30% (mean xG ~0.07 vs actual ~0.05)
- sum-to-zero penalty in irt_model.py uses sum() instead of mean() on player axis
- Adam runs max 2000 epochs without converging; L-BFGS does 20 steps
- Confidence bands in build_data.py use prior tau (0.2), not learned tau
- No strength-state filtering (PP/EV/PK all pooled)

## File Layout
- goalie_model/config.py: ModelConfig with all hyperparameters
- goalie_model/irt_model.py: DynamicIRTModel (PyTorch nn.Module)
  - _log_prior(): priors on all parameters
  - _sum_to_zero_penalty(): soft centering constraint
  - loss(): negative log-posterior
  - predict_logit(): forward pass
  - get_shooter_skills() / get_goalie_skills(): extract centered skills
- goalie_model/fit.py: optimization loops
  - _run_adam(): Adam with cosine LR + early stopping
  - _run_lbfgs(): L-BFGS refinement
- goalie_model/summarize.py: post-processing
  - compute_gsax(): per-goalie GSAx calculation
- dashboard/build_data.py: 
  - compute_fsax(): per-shooter FSAx calculation
  - main(): assembles all CSVs
- dashboard/app.py: Streamlit pages
```

### Specific Implementation Guidance

#### Task 1.1: Fix Tau Collapse

In `config.py`, add a new field:
```python
tau_min: float = 0.01       # Lower bound for random-walk SD
sigma_season_min: float = 0.05  # Lower bound for season-start SD  
```

In `irt_model.py`, in the `_log_prior()` method, after computing `tau_theta` and `tau_phi` from log-space, apply clamping:
```python
tau_theta = torch.clamp(self.log_tau_theta.exp(), min=cfg.tau_min)
tau_phi   = torch.clamp(self.log_tau_phi.exp(),   min=cfg.tau_min)
sig_seas_theta = torch.clamp(self.log_sigma_season_theta.exp(), min=cfg.sigma_season_min)
sig_seas_phi   = torch.clamp(self.log_sigma_season_phi.exp(),   min=cfg.sigma_season_min)
```

Also apply the same clamping in the weekly random-walk prior section (the `for s in range(self.n_seasons)` loop).

**IMPORTANT**: The same clamped values must be used consistently in ALL places where tau/sigma_season are referenced in `_log_prior()`. Search for all uses of `tau_theta`, `tau_phi`, `sig_seas_theta`, `sig_seas_phi` in that method.

#### Task 1.2: Model-Recalibrated Baseline for FSAx_raw / GSAx_raw

In `build_data.py` `compute_fsax()`, change:
```python
fsax_raw = round(goals_actual - goals_xg_only, 3)
```
to:
```python
# Use IRT-recalibrated baseline (β₀ + α·xg, θ=φ=0) instead of raw xG
goals_baseline = float(p_avg_goalie[mask].sum())  # already uses β₀ + α·xg - mean_phi
fsax_raw = round(goals_actual - goals_baseline, 3)
```

Similarly in `summarize.py` `compute_gsax()`, change:
```python
gsax_raw = round(goals_xg_only - goals_actual, 2)
```
to use the IRT recalibrated baseline instead of raw xG.

Keep the raw xG columns available for reference, but label the recalibrated versions prominently.

#### Task 1.3: Fix Sum-to-Zero Penalty

In `irt_model.py`, `_sum_to_zero_penalty()`, change:
```python
theta_means = self.theta[:, s, :T].sum(dim=0)
phi_means = self.phi[:, s, :T].sum(dim=0)
```
to:
```python
theta_means = self.theta[:, s, :T].mean(dim=0)
phi_means = self.phi[:, s, :T].mean(dim=0)
```

#### Task 1.4: Increase Optimization Budget

In `config.py`, change defaults:
```python
max_epochs_full: int = 5000   # was 2000
lbfgs_steps: int = 100        # was 20
patience: int = 150            # was 75
```

In `run_model.py`, change the `--epochs-full` default to 5000.

#### Task 2.1: Default to Even Strength

In `run_model.py`, add `strength_states=["EVEN"]` to the ModelConfig constructor. Keep the config parameter so users can override.

**Note**: This changes the data significantly. McDavid's even-strength line: 256 shots, 17 goals, 25.23 xG. The model will only see 5v5 shots.

#### Task 2.4: Methodology Page

Add a new function `page_methodology()` to `app.py`. Include the non-technical and technical explanations from §10 of this document. Register it in the PAGES dict.

#### Task 2.5: Fix Goalie Trajectory μ_φ Chart

In `app.py`, `page_goalie_trajectory()`, the second chart plots `mu_phi` with `phi_lo`/`phi_hi` bands. This is wrong — `mu_phi` is a static career baseline, not a weekly value. Either:
- Remove the second chart entirely (mu_phi is just a horizontal line)
- Or replace it with a "performance vs baseline" chart showing `phi - mu_phi` (deviation from career average)

#### Task 2.3: Fix Confidence Bands

In `build_data.py`, replace:
```python
tau = getattr(state.config, "prior_tau", 0.2)
```
with:
```python
import math
tau_theta_learned = math.exp(float(state.param_dict["log_tau_theta"]))
tau_phi_learned = math.exp(float(state.param_dict["log_tau_phi"]))
tau_min = getattr(state.config, "tau_min", 0.01)
tau_theta_eff = max(tau_theta_learned, tau_min)
tau_phi_eff = max(tau_phi_learned, tau_min)
```
Then use `tau_phi_eff` for goalie bands and `tau_theta_eff` for shooter bands.

### Testing After Changes

After implementing Phase 1:
1. Run `python run_model.py` — verify tau_theta and tau_phi learn values > tau_min
2. Run `python check_params.py` — verify per-player weekly ranges are now > 0.001
3. Run `python dashboard/build_data.py` — verify FSAx_raw values are centered near 0
4. Check McDavid's ranking — should move toward the middle of the distribution
5. Check weekly trajectories show visible week-to-week variation
6. Run `streamlit run dashboard/app.py` and visually verify trajectory charts

### Files That Need Editing

Phase 1 (critical fixes):
- `goalie_model/config.py` — add tau_min, sigma_season_min, increase epochs
- `goalie_model/irt_model.py` — clamp tau/sigma, fix sum-to-zero mean()
- `goalie_model/summarize.py` — use recalibrated baseline for gsax_raw
- `dashboard/build_data.py` — use recalibrated baseline for fsax_raw, fix confidence bands
- `run_model.py` — increase default epochs, add strength_states default

Phase 2 (dashboard):
- `dashboard/app.py` — add methodology page, fix μ_φ chart, add explanatory text

### Files That Should NOT Be Changed
- `goalie_model/data_prep.py` — data pipeline is correct
- `goalie_model/fit.py` — optimization loop is correct (just needs more iterations)
- `ingest_scripts/*` — ingestion is correct
- `Statsyuk-xGoals-Model/*` — external model, do not modify
