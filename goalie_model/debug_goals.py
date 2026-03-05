import pandas as pd
import numpy as np

# Load the source data
df = pd.read_csv(r'ingest_scripts\nhl_pbp_2024_2025_with_xg.csv')

print("Original data:")
print(f"Total rows: {len(df)}")
print(f"shot_made column exists: {'shot_made' in df.columns}")
print(f"Goals in source: {df['shot_made'].sum()}")
print(f"Non-goals: {(df['shot_made'] == 0).sum()}")
print(f"\nshot_made dtype: {df['shot_made'].dtype}")
print(f"\nFirst few shot_made values: {df['shot_made'].head(20).tolist()}")

# Now check after dropna
df = df.dropna(subset=["xCoord", "yCoord", "xG", "details.shootingPlayerId", "details.goalieInNetId"])
print(f"\nAfter dropna:")
print(f"Total rows: {len(df)}")
print(f"Goals remaining: {df['shot_made'].sum()}")

# Check a small aggregation
print("\n\nTesting aggregation:")
test_agg = df.groupby('game_id').agg(
    goals=("shot_made", "sum"),
    shots=("shot_made", "count")
).reset_index()
print(test_agg[test_agg['goals'] > 0].head(10))
print(f"\nGames with goals: {(test_agg['goals'] > 0).sum()}")
print(f"Total goals in aggregation: {test_agg['goals'].sum()}")
