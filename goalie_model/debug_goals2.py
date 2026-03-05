import pandas as pd

df = pd.read_csv(r'ingest_scripts\nhl_pbp_2024_2025_with_xg.csv')

print("Checking why goals are dropped:\n")

goals = df[df['shot_made'] == 1]
print(f"Total goals: {len(goals)}")
print(f"Goals with shooting player: {goals['details.shootingPlayerId'].notna().sum()}")
print(f"Goals with goalie: {goals['details.goalieInNetId'].notna().sum()}")
print(f"Goals with both: {(goals['details.shootingPlayerId'].notna() & goals['details.goalieInNetId'].notna()).sum()}")

print("\n\nNon-goals:")
non_goals = df[df['shot_made'] == 0]
print(f"Total non-goals: {len(non_goals)}")
print(f"Non-goals with shooting player: {non_goals['details.shootingPlayerId'].notna().sum()}")
print(f"Non-goals with goalie: {non_goals['details.goalieInNetId'].notna().sum()}")

print("\n\nSample goal rows:")
print(goals[['event_type', 'details.shootingPlayerId', 'details.goalieInNetId', 'details.scoringPlayerId']].head(10))
