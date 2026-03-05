# Dynamic Goalie–Shooter IRT Model: Implementation Requirements

## 1. Scope and Objectives

- Build a Bayesian MAP-estimated model of shooter finishing skill and goalie shot-stopping skill.
- Use shot-level xG (from Statsyuk-xGoals-Model) as a fixed shot difficulty term.
- Model two seasons simultaneously: previous season (S=1) and current season (S=2).
- Allow weekly-evolving skills within each season via a random-walk prior.
- Target runtime: full two-season fit on a personal CPU in well under ~20 minutes; weekly updates in ~1–5 minutes (with warm-starting and/or windowed updates).

## 2. Data Requirements and Preprocessing

### 2.1. Input Data Structure

Each row corresponds to a single unblocked shot on goal. Minimal columns required:

- `game_id`: unique game identifier.
- `season`: integer/label distinguishing previous vs current season (e.g., 2023, 2024).
- `game_date`: date or datetime.
- `shooter_id`: unique shooter identifier.
- `goalie_id`: unique goalie identifier.
- `is_goal`: 1 if shot became a goal, 0 otherwise.
- `xg`: xG estimate (probability) from Statsyuk-xGoals-Model.

Optional but recommended additional columns (for filtering/diagnostics only):

- `strength_state` (e.g., 5v5, 5v4, 4v5, etc.).
- `score_state`, `period`, `home_away`, etc.

### 2.2. Season and Week Indexing

- Define a mapping from raw `season` to integer season index S ∈ {1, 2}:
  - S=1: previous season.
  - S=2: current season.
- Define a weekly time index within each season:
  - Compute `week_index` per shot by grouping game dates into league weeks (e.g., ISO week or custom week bins starting from first game date of the season).
  - Map each (season, week) pair to a contiguous integer t_s ∈ {1, …, T_s}.
- Store:
  - `season_index[i]` (1 or 2).
  - `week_index[i]` (1…T_s within that season).

### 2.3. Player Indexing

- Construct internal integer indices for shooters and goalies:
  - Map `shooter_id` → `shooter_index` ∈ {0,…,N_shooters-1}.
  - Map `goalie_id` → `goalie_index` ∈ {0,…,N_goalies-1}.
- Persist mappings (for reporting/model usage):
  - `shooter_id_to_index` and inverse.
  - `goalie_id_to_index` and inverse.

### 2.4. xG Transformation

- Convert xG probabilities to logits:
  - `xg_logit = log(xg / (1 - xg))`.
  - Handle numeric stability (clip xg to [1e-5, 1 - 1e-5]).

### 2.5. Data Filtering

- Exclude shots with missing `xg`, `shooter_id`, `goalie_id`, or `is_goal`.
- Optionally: restrict to particular strength states (e.g., 5v5, 5v4, 4v5) controlled by a config.

## 3. Model Specification

### 3.1. Likelihood

For shot i:

- Observed outcome: y_i ∈ {0,1}.
- Bernoulli likelihood:
  - y_i ~ Bernoulli(p_i).

### 3.2. Linear Predictor

For shot i with shooter p = shooter_index[i], goalie q = goalie_index[i], season s = season_index[i], week t = week_index[i]:

- xG logit term: η_i^xG = xg_logit[i].
- Linear predictor:

  logit(p_i) = β0 + α · η_i^xG + θ[p, s, t] − φ[q, s, t]

Where:

- β0: global intercept.
- α: scaling/calibration coefficient for xG logit.
- θ[p, s, t]: latent shooter finishing skill.
- φ[q, s, t]: latent goalie shot-stopping skill (higher φ reduces p_i).

### 3.3. Hierarchical Priors

#### 3.3.1. Player Baselines (Career-Level)

For each shooter p:

- μ_θ[p] ~ Normal(0, σ_μ_θ²)

For each goalie q:

- μ_φ[q] ~ Normal(0, σ_μ_φ²)

#### 3.3.2. Season-Level Initial Skills

For each season s ∈ {1,2} and player:

- Shooter season start (week t=1):
  - θ[p, s, 1] ~ Normal(μ_θ[p], σ_season_θ²)

- Goalie season start (week t=1):
  - φ[q, s, 1] ~ Normal(μ_φ[q], σ_season_φ²)

#### 3.3.3. Within-Season Weekly Dynamics (Random Walk)

For weeks t = 2,…,T_s:

- θ[p, s, t] ~ Normal(θ[p, s, t−1], τ_θ²)
- φ[q, s, t] ~ Normal(φ[q, s, t−1], τ_φ²)

### 3.4. Global Priors

- β0 ~ Normal(0, 10²)
- α ~ Normal(1, 0.5²)
- σ_μ_θ, σ_μ_φ ~ HalfNormal(1.0)
- σ_season_θ, σ_season_φ ~ HalfNormal(0.5)
- τ_θ, τ_φ ~ HalfNormal(0.2)

### 3.5. Identifiability and Centering

To prevent drift and improve identifiability:

- Implicit centering via priors on μ_θ, μ_φ with mean 0.
- Optionally, add soft sum-to-zero constraints per week:
  - Penalize (∑_p θ[p, s, t])² and (∑_q φ[q, s, t])² for each (s,t).
- At reporting time, center skills by subtracting weekly means if needed.

## 4. Parameterization and Shapes

Assuming:

- N_shooters: number of unique shooters.
- N_goalies: number of unique goalies.
- S = 2 seasons.
- T_s: weeks per season s (can differ across seasons).

Internal representations (suggested):

- θ: float tensor of shape [N_shooters, S, max_T], with masking for t > T_s.
- φ: float tensor of shape [N_goalies, S, max_T], with masking for t > T_s.
- μ_θ: float vector [N_shooters].
- μ_φ: float vector [N_goalies].
- β0, α: scalar parameters.
- σ_μ_θ, σ_μ_φ, σ_season_θ, σ_season_φ, τ_θ, τ_φ: scalar positive parameters.

Masking / validity:

- Maintain `valid_week_mask_shooters[p, s, t]` and `valid_week_mask_goalies[q, s, t]` or infer validity from T_s when computing penalties.

## 5. Objective Function for MAP

We maximize log posterior, or equivalently minimize negative log posterior:

- Objective = −log_likelihood − log_prior_terms.

### 5.1. Log-Likelihood

For all shots i:

- Compute η_i^xG, θ[p,s,t], φ[q,s,t].
- logit(p_i) = β0 + α · η_i^xG + θ[p,s,t] − φ[q,s,t].
- p_i = sigmoid(logit(p_i)).
- Add Bernoulli log-likelihood:
  - logL_i = y_i * log(p_i + ε) + (1 − y_i) * log(1 − p_i + ε).

Total log-likelihood = ∑_i logL_i.

### 5.2. Log-Prior Terms (as Penalties)

Implement priors as L2 penalties (equivalent to Gaussian log priors) plus HalfNormal priors on scale parameters:

- Player baselines:
  - ∑_p NormalLogPDF(μ_θ[p] | 0, σ_μ_θ) + ∑_q NormalLogPDF(μ_φ[q] | 0, σ_μ_φ).

- Season start:
  - ∑_{p,s} NormalLogPDF(θ[p,s,1] | μ_θ[p], σ_season_θ).
  - ∑_{q,s} NormalLogPDF(φ[q,s,1] | μ_φ[q], σ_season_φ).

- Random-walk dynamics:
  - ∑_{p,s,t>1} NormalLogPDF(θ[p,s,t] | θ[p,s,t−1], τ_θ).
  - ∑_{q,s,t>1} NormalLogPDF(φ[q,s,t] | φ[q,s,t−1], τ_φ).

- Global priors:
  - NormalLogPDF(β0 | 0, 10) + NormalLogPDF(α | 1, 0.5).
  - HalfNormalLogPDF for σ_μ_θ, σ_μ_φ, σ_season_θ, σ_season_φ, τ_θ, τ_φ.

### 5.3. Optimization Target

- Negative log-posterior:

  loss = −(log_likelihood + log_prior)

- Minimize `loss` using gradient-based optimization.

## 6. Optimization and Runtime Strategy

### 6.1. Framework

- Preferred: PyTorch or JAX for explicit control over tensors, masks, and optimization.
- Alternative: PyMC (using `find_MAP`) if convenient, but must validate runtime.

### 6.2. Initialization

- Step 1: Fit simple logistic regression of `is_goal` on `xg_logit` only to initialize β0 and α.
- Step 2: Initialize all μ_θ, μ_φ, θ, φ to 0.
- Step 3: For subsequent weekly runs, load previous MAP solution as initialization.

### 6.3. Optimizer Configuration

- Initial full fit:
  - Optimizer: Adam or AdamW.
  - Learning rate: ~1e-2 (tunable), with decay.
  - Epochs/iterations: chosen to converge but bounded (e.g., 1000–2000 iterations for full history) with early stopping on validation loss.

- Weekly updates:
  - Warm-start from previous parameters.
  - Either:
    - (A) Re-optimize all parameters with small number of iterations (e.g., 100–200), or
    - (B) Freeze early weeks and only update the last K weeks (configurable, e.g., K=6–8), plus global parameters.

- Optional refinement: After Adam, run a small number of L-BFGS iterations initialized at the Adam solution.

### 6.4. Batching and Memory

- If needed, implement mini-batch likelihood computation over shots to reduce peak memory usage.
- Accumulate gradient over mini-batches, then optimizer step.

## 7. Outputs and Post-Processing

### 7.1. Core Outputs

For each weekly run, save:

- Parameter state (for warm-starting):
  - θ[p,s,t], φ[q,s,t], μ_θ[p], μ_φ[q], β0, α, and all scale parameters.
- Skill summaries:
  - Shooter table: career baseline μ_θ, season-start θ[p,s,1], weekly θ[p,2,t] for current season.
  - Goalie table: analogous μ_φ, season-start φ[q,s,1], weekly φ[q,2,t].

### 7.2. Derived Metrics

- For each shooter p over a period (e.g., season-to-date or last N weeks):
  - Finishing contribution relative to xG.
- For each goalie q:
  - Goals Saved Above xG (GSAx) computed using model-predicted probabilities vs raw xG-only probabilities.

### 7.3. Centering and Reporting

- Optionally center θ and φ by subtracting weekly league means before reporting to make “0” represent league-average that week.
- Provide configuration flag:
  - `center_weekly = True/False`.

## 8. Configuration and Interfaces

### 8.1. Configuration File

Create a simple config structure (YAML/JSON or Python dataclass) with fields:

- Seasons to include (previous, current identifiers).
- Strength state filters.
- Weekly binning method (ISO week vs custom start date and 7-day bins).
- Optimization hyperparameters: learning rate, epochs, batch size, etc.
- Dynamics window for weekly updates (K weeks to reoptimize).
- Centering options.

### 8.2. Public API Functions (Planned)

High-level Python functions to implement later:

1. `prepare_shot_data(raw_df, config) -> model_data`:
   - Takes raw shots with xG and returns indexed arrays/tensors and mappings.

2. `fit_full_map(model_data, config) -> model_state`:
   - Runs full two-season MAP optimization from scratch.

3. `update_map_with_new_week(model_state, new_shots_df, config) -> model_state`:
   - Ingests new shots from current season, updates data, warm-starts optimization, and returns updated state.

4. `summarize_skills(model_state, when="current_week", centered=True) -> shooter_table, goalie_table`:
   - Returns tidy tables of skills and derived metrics for visualization or export.

5. `save_model_state(model_state, path)` / `load_model_state(path) -> model_state`.

## 9. Validation and Diagnostics

### 9.1. Predictive Performance

- Split historical data (e.g., previous season) into training and validation weeks.
- Compare:
  - xG-only model vs xG + static IRT vs xG + dynamic IRT.
- Metrics:
  - Log loss, Brier score, calibration plots.

### 9.2. Reasonableness Checks

- Trajectories of θ[p,2,t] and φ[q,2,t] for selected well-known players should:
  - Move sensibly with observed hot/cold streaks, but not be dominated by single games.
- Check variance/hyperparameters are in reasonable ranges (not collapsing to zero or exploding).

### 9.3. Runtime Monitoring

- Log wall-clock runtime and iteration counts for:
  - Initial full fit.
  - Each weekly update.
- Adjust batch size, epochs, and dynamic window K if runtime is too high.

---

This document defines the data requirements, model structure, parameterization, and optimization strategy for implementing the dynamic goalie–shooter IRT model with weekly updates and MAP estimation over two seasons.