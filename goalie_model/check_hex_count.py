import pandas as pd
import numpy as np
from math import sqrt

# Load data
df = pd.read_csv(r'ingest_scripts\nhl_pbp_2024_2025_with_xg.csv')
df = df.dropna(subset=['xCoord', 'yCoord'])

# Normalize coordinates
def normalize_coords(row):
    x, y = row['xCoord'], row['yCoord']
    side = str(row['homeTeamDefendingSide']).lower()
    return (-x, -y) if side == 'right' else (x, y)

df[['x_norm', 'y_norm']] = df.apply(lambda r: pd.Series(normalize_coords(r)), axis=1)
df = df[df['x_norm'] > 0.0]

# Test different hex sizes
for HEX_SIZE_FEET in [5.0, 10.0, 15.0, 20.0, 25.0]:
    def hex_coords(x, y, size=HEX_SIZE_FEET):
        q = (2/3) * x / size
        r = (-1/3) * x / size + (sqrt(3)/3) * y / size
        return q, r
    
    df[['hex_q', 'hex_r']] = df.apply(
        lambda r: pd.Series(hex_coords(r['x_norm'], r['y_norm'], HEX_SIZE_FEET)), axis=1
    )
    df['hex_q_round'] = df['hex_q'].round().astype(int)
    df['hex_r_round'] = df['hex_r'].round().astype(int)
    df['hex_id'] = df.apply(lambda r: f"{r.hex_q_round}_{r.hex_r_round}", axis=1)
    
    unique_hexes = df['hex_id'].nunique()
    avg_per_hex = len(df) / unique_hexes
    
    print(f'\n{HEX_SIZE_FEET:.0f}ft hexes:')
    print(f'  Unique hexes: {unique_hexes}')
    print(f'  Average shots per hex: {avg_per_hex:.1f}')
    print(f'  Total shots: {len(df)}')
