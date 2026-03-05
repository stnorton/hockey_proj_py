#!/usr/bin/env python
# ---------------------------------------------------------------------------
#  Predict xG for 2024-2025 NHL Play-by-Play Data
#  Based on Statsyuk-xGoals-Model/pipeline.ipynb
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pickle
import os

# ---------------------------------------------------
#                 UTILITY FUNCTIONS
# ---------------------------------------------------
def time_to_seconds(t_str):
    """Convert MM:SS string to seconds."""
    try:
        mm, ss = t_str.split(':')
        return int(mm) * 60 + int(ss)
    except Exception:
        return np.nan


def add_prior_event_features(df):
    """Add features based on prior event timing and location."""
    df = df.copy()
    df["current_time_s"] = df["time_in_period"].apply(time_to_seconds)
    df["prev_time_s"] = df["prev_event_time"].apply(time_to_seconds)

    same_period = df["period_number"] == df["prev_event_period"]
    df["time_since_last_event"] = np.where(
        same_period,
        df["current_time_s"] - df["prev_time_s"],
        np.nan,
    )

    valid_coords = (
        same_period
        & df[["xCoord", "yCoord", "prev_event_x", "prev_event_y"]].notnull().all(axis=1)
    )
    df["distance_from_last_event"] = np.where(
        valid_coords,
        np.sqrt(
            (df["xCoord"] - df["prev_event_x"]) ** 2
            + (df["yCoord"] - df["prev_event_y"]) ** 2
        ),
        np.nan,
    )

    df["delta_x"] = np.where(valid_coords, df["xCoord"] - df["prev_event_x"], np.nan)
    df["delta_y"] = np.where(valid_coords, df["yCoord"] - df["prev_event_y"], np.nan)

    df["movement_angle"] = np.degrees(np.arctan2(df["delta_y"], df["delta_x"]))
    df["movement_speed"] = df["distance_from_last_event"] / df["time_since_last_event"]
    return df


def clean_and_calculate_coords(df):
    """
    Clean coordinates and calculate shot distance/angle.
    Normalize all shots to positive x-coordinate (attacking direction).
    """
    df = df.copy()
    
    # Flip negative x-coordinates to positive
    mask = df["xCoord"] < 0
    df.loc[mask, "xCoord"] = -df.loc[mask, "xCoord"]
    df.loc[mask, "yCoord"] = -df.loc[mask, "yCoord"]

    # Filter to valid rink coordinates
    df = df[(df["xCoord"].between(-99, 99)) & (df["yCoord"].between(-42, 42))]
    df = df.dropna(subset=["xCoord", "yCoord"])

    # Calculate shot distance and angle from net center (89, 0)
    x_abs = df["xCoord"].abs()
    df["shot_distance_calc"] = np.sqrt((89 - x_abs) ** 2 + df["yCoord"] ** 2)
    df["shot_angle_signed"] = np.degrees(
        np.arctan2(df["yCoord"], (89 - df["xCoord"]))
    )
    df["shot_angle_calc"] = df["shot_angle_signed"]
    return df


def compute_binned_score_diff(row):
    """Bin score differential into categories."""
    diff = row.get("homeScore", 0) - row.get("awayScore", 0)
    if diff <= -2:
        return "down2+"
    elif diff == -1:
        return "down1"
    elif diff == 0:
        return "tie"
    elif diff == 1:
        return "up1"
    else:
        return "up2+"


def point_in_polygon(x, y, polygon):
    """Check if point (x, y) is inside a polygon."""
    num_points = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(num_points + 1):
        p2x, p2y = polygon[i % num_points]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    xinters = (
                        (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1y != p2y
                        else p1x
                    )
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# ---------------------------------------------------
#      BUILD FEATURE MATRIX (Same as training)
# ---------------------------------------------------
def build_feature_matrix(df):
    """
    Build the feature matrix with all engineered features.
    Returns only X (no y since we're predicting).
    """
    df = df.copy()

    numeric_cols = [
        "shot_distance_calc",
        "shot_angle_calc",
        "is_forward",
        "time_since_last_event",
        "distance_from_last_event",
        "delta_x",
        "delta_y",
        "movement_angle",
        "movement_speed",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0

    X_num = df[numeric_cols].fillna(0)

    # Non-linear transforms
    X_num["distance_sq"] = X_num["shot_distance_calc"] ** 2
    X_num["log_distance"] = np.log1p(X_num["shot_distance_calc"])
    X_num["angle_sq"] = X_num["shot_angle_calc"] ** 2
    X_num["dist_x_angle"] = X_num["shot_distance_calc"] * X_num["shot_angle_calc"]

    # Cross-terms & higher-order
    X_num["movement_speed_sq"] = X_num["movement_speed"] ** 2
    X_num["time_since_last_event_sq"] = X_num["time_since_last_event"] ** 2
    X_num["dist_x_speed"] = X_num["shot_distance_calc"] * X_num["movement_speed"]

    # Distance bin
    bins = [0, 10, 20, 30, 50, 200]
    labels = [1, 2, 3, 4, 5]
    X_num["dist_bin"] = (
        pd.cut(
            X_num["shot_distance_calc"],
            bins=bins,
            labels=labels,
        )
        .astype(float)
        .fillna(0)
    )

    # "Slot" indicator
    X_num["in_slot"] = np.where(
        (X_num["shot_distance_calc"] < 25) & (X_num["shot_angle_calc"].abs() < 30),
        1,
        0,
    )

    # behind_net indicator
    X_num["behind_net"] = np.where(
        (df["xCoord"] > 89) | (df["xCoord"] < -89),
        1,
        0,
    )

    # Radial distance & "home plate"
    X_num["radial_distance"] = df["yCoord"].abs()
    home_plate_polygon = [(89, -3.5), (89, 3.5), (69, 22), (52, 0), (69, -22)]
    X_num["home_plate"] = [
        int(point_in_polygon(x, y, home_plate_polygon))
        for x, y in zip(df["xCoord"], df["yCoord"])
    ]

    # Period & time fraction
    X_num["period"] = df.get("period_number", 0).fillna(0).astype(int)

    if "time_in_period" in df.columns:
        df["time_s"] = df["time_in_period"].apply(time_to_seconds)
        X_num["time_fraction"] = (df["time_s"] / 1200).clip(0, 1)
    else:
        X_num["time_fraction"] = 0.0

    # Score diff dummies
    df["homeScore"] = df.get("homeScore", 0)
    df["awayScore"] = df.get("awayScore", 0)
    df["score_diff_cat"] = df.apply(compute_binned_score_diff, axis=1)
    score_diff_dummies = pd.get_dummies(df["score_diff_cat"], prefix="scoreDiff")

    # Shot-type one-hots (only these 4 types)
    valid_shot_types = ["wrist", "snap", "slap", "backhand"]
    filtered_shotType = (
        df["shotType"].where(df["shotType"].isin(valid_shot_types), np.nan)
        if "shotType" in df.columns
        else pd.Series(dtype=str, index=df.index)
    )
    shot_type_dummies = pd.get_dummies(filtered_shotType, prefix="shotType")

    # Combine everything
    X = pd.concat([X_num, score_diff_dummies, shot_type_dummies], axis=1)
    return X


# ---------------------------------------------------
#      MAIN PREDICTION PIPELINE
# ---------------------------------------------------
def predict_xg_for_csv(input_csv, output_csv, model_path):
    """
    Load the CSV, clean/engineer features, predict xG, and save to new CSV.
    
    Parameters:
    -----------
    input_csv : str
        Path to the input CSV file (nhl_pbp_allfields_2024_2025.csv)
    output_csv : str
        Path to save the output CSV with xG predictions
    model_path : str
        Path to the trained model pickle file
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Store important columns to preserve in output
    preserve_columns = []
    for col in ['game_id', 'game_date', 'eventId']:
        if col in df.columns:
            preserve_columns.append(col)
    
    # Map CSV column names to expected names
    # Based on the CSV structure, map the columns appropriately
    column_mapping = {
        'periodDescriptor.number': 'period_number',
        'timeInPeriod': 'time_in_period',
        'situationCode': 'situation_code',
        'details.xCoord': 'xCoord',
        'details.yCoord': 'yCoord',
        'details.shotType': 'shotType',
        'details.awayScore': 'awayScore',
        'details.homeScore': 'homeScore',
        'typeCode': 'event_type',
    }
    
    df = df.rename(columns=column_mapping)
    
    # Filter to shots only (goal=505, shot-on-goal=506, missed-shot=507, blocked-shot=508)
    shot_events = [505, 506, 507, 508]  # Based on NHL API codes
    if 'event_type' in df.columns:
        df = df[df['event_type'].isin(shot_events)].copy()
        print(f"After filtering to shot events: {len(df)} rows")
    
    # Create shot_made column (1 if goal, 0 otherwise)
    # 505 = goal, all others are non-goals
    df['shot_made'] = (df['event_type'] == 505).astype(int)
    
    # Drop rows with missing critical fields
    df = df.dropna(subset=['time_in_period', 'xCoord', 'yCoord'])
    print(f"After dropping missing critical columns: {len(df)} rows")
    
    # Add is_forward column (placeholder - would need player position data)
    df['is_forward'] = 1  # Default assumption
    
    # Create dummy columns for prior event features (if not available in source data)
    for col in ['prev_event_time', 'prev_event_period', 'prev_event_x', 'prev_event_y']:
        if col not in df.columns:
            df[col] = np.nan
    
    print("\nCleaning coordinates and calculating features...")
    df = clean_and_calculate_coords(df)
    df = add_prior_event_features(df)
    
    print(f"After cleaning: {len(df)} rows")
    
    if len(df) == 0:
        print("ERROR: No valid rows remaining after cleaning!")
        return
    
    print("\nBuilding feature matrix...")
    X = build_feature_matrix(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Load the trained model
    print(f"\nLoading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Predicting xG values...")
    xg_predictions = model.predict_proba(X)[:, 1]
    df['xG'] = xg_predictions
    
    # Reorder columns to put important identifiers first, then data, then xG at the end
    output_columns = []
    
    # Add preserved columns first (game_id, game_date, eventId)
    for col in preserve_columns:
        if col in df.columns:
            output_columns.append(col)
    
    # Add all other columns except xG
    for col in df.columns:
        if col not in output_columns and col != 'xG':
            output_columns.append(col)
    
    # Add xG at the end
    output_columns.append('xG')
    
    # Reorder the dataframe
    df = df[output_columns]
    
    # Save to output CSV
    print(f"\nSaving results to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    print(f"\nCompleted! Saved {len(df)} rows with xG predictions.")
    print(f"\nSummary statistics:")
    print(f"  Mean xG: {df['xG'].mean():.4f}")
    print(f"  Median xG: {df['xG'].median():.4f}")
    print(f"  Min xG: {df['xG'].min():.4f}")
    print(f"  Max xG: {df['xG'].max():.4f}")
    print(f"  Goals: {df['shot_made'].sum()}")
    print(f"  Total shots: {len(df)}")
    print(f"\nSample predictions:")
    print(df[['shot_distance_calc', 'shot_angle_calc', 'shotType', 'shot_made', 'xG']].head(10))


if __name__ == "__main__":
    import argparse

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(
        script_dir, "..", "Statsyuk-xGoals-Model", "xgb_combined_gpu_random.pkl"
    )

    parser = argparse.ArgumentParser(
        description="Predict xG for an NHL play-by-play CSV."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=os.path.join(script_dir, "nhl_pbp_allfields_2024_2025.csv"),
        help=(
            "Path to the allfields CSV produced by ingest_2425.py. "
            "Defaults to nhl_pbp_allfields_2024_2025.csv."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output CSV path. Defaults to the input filename with "
            "'_allfields_' replaced by '_' and '_with_xg' appended before .csv."
        ),
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="Path to the trained xG model pickle. Defaults to Statsyuk model.",
    )
    args = parser.parse_args()

    input_csv = args.input_csv

    # Auto-derive output filename from input
    if args.out:
        output_csv = args.out
    else:
        base = os.path.basename(input_csv)                     # nhl_pbp_allfields_2025_2026.csv
        out_base = base.replace("_allfields_", "_") \
                       .replace(".csv", "_with_xg.csv")        # nhl_pbp_2025_2026_with_xg.csv
        output_csv = os.path.join(os.path.dirname(input_csv), out_base)

    print(f"Input  : {input_csv}")
    print(f"Output : {output_csv}")
    print(f"Model  : {args.model}")

    predict_xg_for_csv(input_csv, output_csv, args.model)
