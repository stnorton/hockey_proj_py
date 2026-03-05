"""
add_player_names.py
-------------------
Enrich model_output CSVs with player names fetched from the NHL API.
Names are cached in model_output/player_names_cache.json so repeated
runs are instant for already-resolved IDs.

Goalie names (~144 IDs, ~25 s) are always added.
Shooter names (~1500 IDs, ~4 min) are opt-in via --shooters flag.

Usage
-----
  python add_player_names.py              # goalies only (fast)
  python add_player_names.py --shooters   # goalies + shooters
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goalie_model.summarize import fetch_player_names

OUT_DIR    = "model_output"
CACHE_PATH = os.path.join(OUT_DIR, "player_names_cache.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shooters", action="store_true",
                   help="Also fetch shooter names (adds ~4 min on first run).")
    args = p.parse_args()

    gsax     = pd.read_csv(os.path.join(OUT_DIR, "goalie_gsax.csv"))
    goalies  = pd.read_csv(os.path.join(OUT_DIR, "goalie_skills.csv"))
    shooters = pd.read_csv(os.path.join(OUT_DIR, "shooter_skills.csv"))

    # --- Goalie names ---------------------------------------------------------
    n = gsax["goalie_id"].nunique()
    print(f"Fetching goalie names ({n} unique, cached after first run)...", flush=True)
    t0 = time.time()
    goalie_names = fetch_player_names(
        gsax["goalie_id"].dropna().unique(), cache_path=CACHE_PATH
    )
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    def _name(id_series, names):
        return id_series.map(lambda x: names.get(int(float(x)), str(x)) if x == x else None)

    for df in (gsax, goalies):
        if "player_name" not in df.columns:
            df.insert(0, "player_name", _name(df["goalie_id"], goalie_names))
        else:
            df["player_name"] = _name(df["goalie_id"], goalie_names)

    # --- Shooter names (optional) ---------------------------------------------
    if args.shooters:
        n = shooters["shooter_id"].nunique()
        print(f"Fetching shooter names ({n} unique, cached after first run)...", flush=True)
        t1 = time.time()
        shooter_names = fetch_player_names(
            shooters["shooter_id"].dropna().unique(), cache_path=CACHE_PATH
        )
        print(f"  done in {time.time()-t1:.1f}s", flush=True)
        if "player_name" not in shooters.columns:
            shooters.insert(0, "player_name", _name(shooters["shooter_id"], shooter_names))
        else:
            shooters["player_name"] = _name(shooters["shooter_id"], shooter_names)

    # --- Save -----------------------------------------------------------------
    gsax.to_csv(os.path.join(OUT_DIR, "goalie_gsax.csv"),       index=False)
    goalies.to_csv(os.path.join(OUT_DIR, "goalie_skills.csv"),  index=False)
    shooters.to_csv(os.path.join(OUT_DIR, "shooter_skills.csv"), index=False)
    print("CSVs saved.")

    print()
    cols = ["player_name", "shots_faced", "goals_actual", "goals_xg_only"]
    if "gsax_raw" in gsax.columns:
        cols.append("gsax_raw")
    cols.append("gsax")
    print(gsax.sort_values("gsax", ascending=False)[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
