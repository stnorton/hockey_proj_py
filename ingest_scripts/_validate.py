import pandas as pd

df = pd.read_csv('nhl_pbp_allfields_2025_2026.csv')
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Games: {df['game_id'].nunique():,}")
print(f"Date range: {df['game_date'].min()}  ->  {df['game_date'].max()}")

# The raw API uses typeCode: 505=goal, 506=shot, 507=missed, 508=blocked
shots = df[df['typeCode'].isin([505, 506, 507, 508])]
goals = df[df['typeCode'] == 505]
print(f"Shot events (505-508): {len(shots):,}")
print(f"Goals (505): {len(goals):,}")
print(f"Goal rate: {len(goals)/len(shots)*100:.2f}%")

# Check columns that predict_xg.py needs (raw names before its column_mapping)
needed = [
    'details.xCoord', 'details.yCoord', 'details.shotType',
    'situationCode', 'details.shootingPlayerId', 'details.goalieInNetId',
    'typeCode', 'timeInPeriod', 'periodDescriptor.number',
    'game_id', 'game_date',
]
missing = [c for c in needed if c not in df.columns]
print(f"Missing expected cols: {missing if missing else 'none'}")

print(f"Shots missing xCoord: {shots['details.xCoord'].isna().sum():,}")
print(f"Shots missing yCoord: {shots['details.yCoord'].isna().sum():,}")
print(f"Shots missing goalie: {shots['details.goalieInNetId'].isna().sum():,}")
