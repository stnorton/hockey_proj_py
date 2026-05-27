# Hex Map Dashboard Page — Requirements

## Overview

A new dashboard page ("Shot Hex Map") that visualises per-hex shooting and
save skill deviations on a half-rink diagram.  For a selected player, each hex
is coloured by how much better or worse that player performed relative to
expected (goals vs xG), adjusted for the quality of opponents they faced using
the IRT model's θ (shooter) and φ (goalie) estimates.

A league-wide aggregate view is also available, showing which zones on the ice
have the highest finishing or save-skill deviations across all players.

---

## Scope

- **New page** added to the existing Streamlit dashboard (`dashboard/app.py`).
- **No model changes** — this is an empirical hex-level stat computed from shot
  data and existing IRT-derived adjustments, not a new IRT parameter dimension.
- **Respects the existing situation toggle** (All / Even / PP) in the sidebar.
- **Full-season aggregate** only (no week-range slider).

---

## Data Pipeline

### Source Data

Shot-level data from the xG CSVs (`ingest_scripts/nhl_pbp_*_with_xg.csv`),
which include:

| Column | Use |
|---|---|
| `xCoord`, `yCoord` | Shot location |
| `homeTeamDefendingSide` | Coordinate normalisation |
| `details.shootingPlayerId` / `details.scoringPlayerId` | Shooter ID |
| `details.goalieInNetId` | Goalie ID |
| `shot_made` | Binary outcome (1 = goal) |
| `xG` | Pre-shot expected goal probability |
| `situation_code` | Strength-state filtering |
| `game_id` | Season assignment |
| `periodDescriptor.periodType` | Shootout exclusion |

### Hex Binning

Use **15 ft axial hexes**, matching the existing `clean_and_aggregate_shots.py`
convention:

```python
HEX_SIZE_FEET = 15.0

def hex_coords(x, y, size=HEX_SIZE_FEET):
    q = (2/3) * x / size
    r = (-1/3) * x / size + (sqrt(3)/3) * y / size
    return q, r

hex_q_round = round(q)
hex_r_round = round(r)
hex_id = f"{hex_q_round}_{hex_r_round}"
```

Coordinate normalisation: flip `(x, y)` when `homeTeamDefendingSide == "right"`
so all shots face the same net.  Keep only offensive-zone shots (`x_norm > 0`).

### IRT Adjustment

For each shot, compute an **IRT-adjusted expected goal probability** that
accounts for the specific shooter's and goalie's skill:

```
logit(p_adj_i) = β₀ + α · logit(xG_i) + θ_j − φ_k
```

where θ_j is the shooter's season-average skill (`mu_theta`) and φ_k is the
goalie's season-average skill (`mu_phi`) from the fitted IRT model.  This
produces a per-shot adjusted baseline that reflects the quality of opponents.

The per-hex deviation stat is then:

- **Shooter hex map**: for each hex, the deviation is
  `Σ(is_goal) − Σ(p_adj)` across all shots by that player from that hex —
  "goals above IRT-adjusted expectation from this zone."  Positive = the
  shooter finishes better than expected from this zone (after accounting for
  goalie quality).

- **Goalie hex map**: for each hex, the deviation is
  `Σ(p_adj) − Σ(is_goal)` (sign-flipped) across all shots the goalie faced
  from that hex — "saves above IRT-adjusted expectation from this zone."
  Positive = the goalie stops more than expected from this zone (after
  accounting for shooter quality).

### Pre-aggregated CSV

A new export step in `dashboard/build_data.py` produces one CSV per situation:

**`dashboard/data/<situation>/hex_shooter.csv`**

| Column | Type | Description |
|---|---|---|
| `shooter_id` | int | Player ID |
| `player_name` | str | Display name |
| `team` | str | Current team |
| `hex_q` | int | Axial hex q coordinate (rounded) |
| `hex_r` | int | Axial hex r coordinate (rounded) |
| `hex_x` | float | Cartesian x centre of hex (feet) |
| `hex_y` | float | Cartesian y centre of hex (feet) |
| `shots` | int | Number of shots from this hex |
| `goals` | int | Actual goals from this hex |
| `xg_sum` | float | Sum of raw xG |
| `irt_xg_sum` | float | Sum of IRT-adjusted p_adj |
| `deviation` | float | goals − irt_xg_sum (goals above adjusted expected) |
| `dev_per_shot` | float | deviation / shots |

**`dashboard/data/<situation>/hex_goalie.csv`**

| Column | Type | Description |
|---|---|---|
| `goalie_id` | int | Player ID |
| `player_name` | str | Display name |
| `team` | str | Current team |
| `hex_q` | int | Axial hex q coordinate (rounded) |
| `hex_r` | int | Axial hex r coordinate (rounded) |
| `hex_x` | float | Cartesian x centre of hex (feet) |
| `hex_y` | float | Cartesian y centre of hex (feet) |
| `shots` | int | Shots faced from this hex |
| `goals` | int | Goals allowed from this hex |
| `xg_sum` | float | Sum of raw xG |
| `irt_xg_sum` | float | Sum of IRT-adjusted p_adj |
| `deviation` | float | irt_xg_sum − goals (saves above adjusted expected) |
| `dev_per_shot` | float | deviation / shots |

**`dashboard/data/<situation>/hex_league.csv`**

League-wide aggregate (no player grouping). One row per hex. Used for the
league-average reference view.

| Column | Type | Description |
|---|---|---|
| `hex_q` | int | |
| `hex_r` | int | |
| `hex_x` | float | |
| `hex_y` | float | |
| `shots` | int | Total shots from this hex across all players |
| `goals` | int | Total goals |
| `xg_sum` | float | Sum of raw xG |
| `goal_rate` | float | goals / shots |
| `xg_rate` | float | xg_sum / shots |

### Minimum Shot Threshold

Hexes with fewer than a configurable minimum number of shots (default: 5 for
individual players, 20 for league aggregate) are excluded from display to
avoid noisy single-shot hexes.

---

## Half-Rink Diagram

### Rink Geometry

The background layer is a half-rink SVG or drawn programmatically with the
standard NHL offensive-zone features:

- **Boards outline**: rounded-corner rectangle, half-rink width = 100 ft
  (x: 0 to 100), full width = 85 ft (y: −42.5 to +42.5).
- **Goal line**: x = 89 ft.
- **Blue line**: x = 25 ft (offensive-zone entry).
- **Goal crease**: semicircle centred at (89, 0), radius = 6 ft.
- **Faceoff circles**: two offensive-zone circles at (69, ±22), radius = 15 ft.
- **Faceoff dots**: centre ice excluded (half-rink); offensive-zone dots at
  (69, ±22) and neutral-zone dots at (20.5, ±22).
- **Centre dot**: at (0, 0) — visible at the cut-off edge.
- **Net**: rectangle at (89, −3) to (91, +3) drawn as a simple icon.
- **Trapezoid**: lines from (89, ±11) to (100, ±8).

Drawn in light grey / low-opacity lines so the hex colours remain visually
dominant.

### Coordinate Mapping

The shot coordinates from the NHL API are in feet:
- x ∈ [−100, 100] (−100 = far end, 0 = centre, 100 = near end).
- y ∈ [−42.5, +42.5] (side boards).

After normalisation all shots have positive x.  The hex map displays x ∈ [0, 100],
y ∈ [−42.5, +42.5].

### Hex Rendering

Each hex is drawn as a regular hexagon polygon centred at the Cartesian
(hex_x, hex_y) position.  Hex vertex radius = `HEX_SIZE_FEET`.

Hex fill colour is determined by the `deviation` or `dev_per_shot` value
(user-togglable; see Controls below).

---

## Colour Scale

**Diverging colour map** centred at zero:

- **Positive deviation** (above average): blue gradient (darker = stronger).
- **Negative deviation** (below average): red gradient (darker = stronger).
- **Zero / neutral**: white.

Colour scale bounds are symmetric: `[−max(|dev|), +max(|dev|)]` so that
white always corresponds to zero.

Hex opacity scales with shot count (more shots = more opaque; fewer shots =
more transparent) to visually de-emphasise low-confidence hexes.

---

## Page Layout & Controls

### Player Selection Mode

A toggle at the top of the page:

| Mode | Description |
|---|---|
| **Individual Player** | Select a single shooter or goalie and see their personal hex map |
| **League Aggregate** | Show league-wide goal rate vs xG rate per hex (no player filter) |

### Individual Player Mode

1. **Role toggle**: `Shooter` / `Goalie` radio button.
2. **Player selector**: searchable dropdown populated from the summary CSV.
   Shows `"Name (Team)"` format.  Default: top player by FSAx (shooter) or
   GSAx (goalie).
3. **Min shots slider**: minimum shots per hex to display (default 5,
   range 1–50).
4. **Metric toggle**: `Total deviation` (sum) vs `Deviation per shot` (rate).

### League Aggregate Mode

1. **View toggle**: `Goal rate` vs `xG rate` vs `Goal rate − xG rate`.
2. **Min shots slider**: minimum shots per hex (default 20).

### Hex Tooltip (on hover)

Displayed in a popup when the user hovers over a hex:

- Hex location (approximate zone name: "slot", "point", "left circle", etc.,
  or just coordinates).
- Shots: N
- Goals: N
- xG: N.NN
- IRT-adj xG: N.NN  *(individual mode only)*
- Deviation: +/−N.NN
- Dev/shot: +/−N.NNN

### Colour Legend

A horizontal colour bar below the chart showing the diverging scale from
negative (red) through zero (white) to positive (blue), with numeric labels.

---

## Integration with Existing Dashboard

### Sidebar

- The page appears in the sidebar page selector as **"🗺️ Shot Hex Map"**.
- The existing **Situation** radio (All / Even / PP) is respected: the hex
  data is loaded from the corresponding `<situation>/` subdirectory.

### Navigation

- The PAGES dict in `app.py` gets a new entry mapping to
  `page_hex_map(situation)`.

### Data Loading

New `@st.cache_data` functions:

```python
@st.cache_data
def load_hex_shooter(situation: str = "all") -> pd.DataFrame:
    return pd.read_csv(_data_dir(situation) / "hex_shooter.csv")

@st.cache_data
def load_hex_goalie(situation: str = "all") -> pd.DataFrame:
    return pd.read_csv(_data_dir(situation) / "hex_goalie.csv")

@st.cache_data
def load_hex_league(situation: str = "all") -> pd.DataFrame:
    return pd.read_csv(_data_dir(situation) / "hex_league.csv")
```

---

## Build Script Changes (`dashboard/build_data.py`)

The `_build_one_situation()` function adds a new step after the existing
shooter/goalie summary exports:

1. Load raw shot-level data from the model state's `shots_df` (already
   available in the state pickle — contains `xCoord`, `yCoord`, `xG`,
   `shooter_id`, `goalie_id`, `is_goal`, `game_id`, `situation_code`,
   `game_date`).  If coordinates are not in the state pickle, reload from
   the xG CSV.
2. Normalise coordinates and filter to offensive zone.
3. Compute hex_q, hex_r, hex_x, hex_y using the 15 ft hex convention.
4. Look up each shooter's `mu_theta` and each goalie's `mu_phi` from the
   model state.  Compute IRT-adjusted p_adj per shot.
5. Group by `(shooter_id, hex_q, hex_r)` → aggregate shots, goals,
   xg_sum, irt_xg_sum, deviation → `hex_shooter.csv`.
6. Group by `(goalie_id, hex_q, hex_r)` → same aggregation (sign-flipped)
   → `hex_goalie.csv`.
7. Group by `(hex_q, hex_r)` only → `hex_league.csv`.
8. Attach player names / teams from the existing name cache.

---

## Charting Library

The hex grid + rink overlay will be rendered using **Altair** (already a
project dependency) with layered marks:

1. **Base layer**: rink lines drawn with `mark_rule` / `mark_arc` /
   `mark_line` in light grey.
2. **Hex layer**: `mark_point(shape="M0,-1 L0.87,-0.5 L0.87,0.5 L0,1 L-0.87,0.5 L-0.87,-0.5Z")`
   or a custom SVG hex path scaled to 15 ft, with `color` mapped to deviation
   and `opacity` mapped to shot count.
   Alternatively, compute hex polygon vertices in Python and use
   `mark_geoshape` or `mark_area` with explicit vertex coordinates for each
   hex.
3. **Tooltip layer**: Altair's built-in tooltip encoding.
4. **Colour legend**: Altair `alt.Scale(scheme="redblue", domainMid=0)` or
   equivalent diverging scale.

If Altair's polygon/hex support proves inadequate, fall back to
**Matplotlib + Streamlit `st.pyplot()`**:
- `matplotlib.collections.RegularPolyCollection` with `numsides=6` renders
  hex grids natively.
- `matplotlib.patches` for the rink outline.
- `st.pyplot(fig)` to embed in Streamlit.

---

## Estimated Data Sizes

| File | Rows (approx) | Size |
|---|---|---|
| `hex_shooter.csv` | ~30,000 (1,500 shooters × ~20 hexes each) | ~2 MB |
| `hex_goalie.csv` | ~5,000 (150 goalies × ~35 hexes each) | ~300 KB |
| `hex_league.csv` | ~50–80 hexes | ~5 KB |

These are committed to the repo alongside the existing dashboard CSVs.

---

## Acceptance Criteria

1. Page loads in < 3 seconds on Streamlit Community Cloud.
2. Half-rink diagram is recognisable (boards, crease, circles, blue line).
3. Selecting a shooter shows hex map coloured by goals vs IRT-adjusted xG
   per zone; positive (blue) = finishes above expected, negative (red) =
   below expected.
4. Selecting a goalie shows hex map with same colour logic but for saves
   above expected (sign-flipped so blue = good for goalie too).
5. League aggregate view shows raw goal rate vs xG rate per hex.
6. Hovering a hex shows shot count, goals, xG, deviation.
7. Situation toggle (All / Even / PP) correctly switches between data sets.
8. Hexes with too few shots are hidden (min-shots slider functional).
9. Colour scale is diverging, symmetric around zero, with a visible legend.
10. No model changes required — all computation is empirical + IRT lookups.
