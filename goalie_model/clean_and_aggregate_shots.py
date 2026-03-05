
import pandas as pd
import numpy as np
from math import sqrt, floor, log

##
#data import
source_file = r'ingest_scripts\nhl_pbp_2024_2025_with_xg.csv'
##

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
HEX_SIZE_FEET = 15.0      # ~15ft hexes (reduced from 5ft for better aggregation)
OFFENSIVE_ZONE_X = 0.0    # keep only shots toward one net
TIME_FREQ = "M"           # monthly buckets for dynamic modeling

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
df = pd.read_csv(source_file)

# Basic type cleaning
df = df.dropna(subset=["xCoord", "yCoord", "xG", "details.goalieInNetId"])

# For goals, the shooter is in details.scoringPlayerId; for non-goals, it's in details.shootingPlayerId
# Combine them into a single shooter_id column
df['shooter_id_raw'] = df['details.shootingPlayerId'].fillna(df['details.scoringPlayerId'])
df = df.dropna(subset=['shooter_id_raw'])


# %%
# ------------------------------------------------------------
# 2. Normalize coordinates (so all shots face one net)
# ------------------------------------------------------------
def normalize_coords(row):
    x, y = row["xCoord"], row["yCoord"]
    side = str(row["homeTeamDefendingSide"]).lower()
    if side == "right":
        return -x, -y
    return x, y

df[["x_norm", "y_norm"]] = df.apply(lambda r: pd.Series(normalize_coords(r)), axis=1)

# Keep only offensive-zone shots
df = df[df["x_norm"] > OFFENSIVE_ZONE_X]

# ------------------------------------------------------------
# 3. Assign player and context fields
# ------------------------------------------------------------
df["shooter_id"] = df["shooter_id_raw"].astype(str)
df["goalie_id"] = df["details.goalieInNetId"].astype(str)
df["game_id"] = df["game_id"].astype(str)

def decode_strength(code):
    """
    Convert numeric NHL situation_code (e.g., 1551, 1541, 1451)
    into a high-level strength category: EVEN, PP, PK, or OTHER.

    Typical interpretation:
        1551 → 5v5  (EVEN)
        1541 → 5v4  (Power Play for shooter)
        1451 → 4v5  (Penalty Kill for shooter)
        3333 → 3v3  (EVEN, OT)
    """
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "EVEN"

    # Extract away and home skater counts
    # Usually encoded as A H H G or A H G G — depends on feed version;
    # we just take the middle two digits.
    away = (code // 100) % 10
    home = (code // 10) % 10

    # Default to even strength
    if away == home:
        return "EVEN"
    elif away > home:
        return "PP"    # shooter team (away) likely has more skaters
    elif away < home:
        return "PK"
    else:
        return "OTHER"

# Apply it to your dataframe
df["strength_state"] = df["situation_code"].apply(decode_strength)


# %%

# ------------------------------------------------------------
# 4. Hex-binning for shot location
# ------------------------------------------------------------
# Helper: axial hex coordinates using a simple offset projection
def hex_coords(x, y, size=HEX_SIZE_FEET):
    q = (2/3) * x / size
    r = (-1/3) * x / size + (sqrt(3)/3) * y / size
    return q, r

df[["hex_q", "hex_r"]] = df.apply(lambda r: pd.Series(hex_coords(r["x_norm"], r["y_norm"], HEX_SIZE_FEET)), axis=1)

# Round to nearest hex center
df["hex_q_round"] = df["hex_q"].round().astype(int)
df["hex_r_round"] = df["hex_r"].round().astype(int)

# Stable hashable hex ID
df["hex_id"] = df.apply(lambda r: f"{r.hex_q_round}_{r.hex_r_round}", axis=1)

# ------------------------------------------------------------
# 5. Compute logit(xG) offset
# ------------------------------------------------------------
def safe_logit(p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

df["offset_logit_xg"] = df["xG"].apply(safe_logit)

# ------------------------------------------------------------
# 6. Temporal binning (monthly)
# ------------------------------------------------------------
# Use game_date column if available, otherwise fall back to game_id parsing
if "game_date" in df.columns:
    df["time_id"] = pd.to_datetime(df["game_date"], errors="coerce").dt.to_period(TIME_FREQ).astype(str)
elif "game_id" in df.columns:
    # Try to extract date from game_id if it follows format like 2024010001
    # First 4 digits are season start year, next 2 are game type, last 4 are game number
    df["time_id"] = pd.to_datetime(df["game_id"].astype(str).str[:4] + "-01-01", errors="coerce").dt.to_period(TIME_FREQ).astype(str)
else:
    df["time_id"] = "ALL"

# Fallback if no valid time mapping: treat all as one period
if df["time_id"].isnull().all():
    df["time_id"] = "ALL"

# ------------------------------------------------------------
# 7. Aggregate to binomial counts
# ------------------------------------------------------------
agg_cols = [
    "shooter_id", "goalie_id", "hex_id", "time_id", "strength_state"
]
df_agg = (
    df.groupby(agg_cols)
      .agg(
          goals=("shot_made", "sum"),
          shots=("shot_made", "count"),
          offset_logit_xg=("offset_logit_xg", "mean"),
          x_mean=("x_norm", "mean"),
          y_mean=("y_norm", "mean"),
      )
      .reset_index()
)

# Optional: add rink_id if relevant
if "rink_id" in df.columns:
    df_agg["rink_id"] = df["rink_id"].mode()[0]

# ------------------------------------------------------------
# 8. Save outputs
# ------------------------------------------------------------
df_agg.to_csv("aggregated_shots_clean.csv", index=False)
print("✅ Cleaned and aggregated shot data saved as aggregated_shots_clean.csv")
