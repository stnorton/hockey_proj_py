# %%
# --------------------------------------------------------
# 1. Initialize API client
# --------------------------------------------------------
from nhlpy import NHLClient
import argparse
import time
import datetime
import pandas as pd
from itertools import chain

client = NHLClient()


# --------------------------------------------------------
# Helpers: season string and current-season detection
# --------------------------------------------------------

def current_season_str() -> str:
    """
    Return the NHL season string (e.g. '20252026') that is active *today*.
    NHL seasons start in October, so:
      - Jan–Aug  → season started previous calendar year   (e.g. Mar 2026 → '20252026')
      - Sep–Dec  → season started this calendar year       (e.g. Oct 2025 → '20252026')
    """
    today = datetime.date.today()
    if today.month >= 9:
        start_year = today.year
    else:
        start_year = today.year - 1
    return f"{start_year}{start_year + 1}"


def season_str_to_label(season: str) -> str:
    """Convert '20242025' → '2024_2025' for use in filenames."""
    return f"{season[:4]}_{season[4:]}"


def get_last_game_id_in_csv(csv_path: str) -> str | None:
    """
    Return the numerically largest game_id already present in an existing
    allfields CSV, or None if the file does not exist / has no game_id column.

    Game IDs are numeric strings like '2025020503'; the largest value is the
    most recently played game that was ingested.
    """
    import os
    if not os.path.exists(csv_path):
        return None
    try:
        # Only read the game_id column — fast even for large files
        ids = pd.read_csv(csv_path, usecols=["game_id"])["game_id"]
        if ids.empty:
            return None
        last = str(int(ids.astype(str).str.extract(r"(\d+)", expand=False).astype(float).max()))
        print(f"📂 Existing file found. Last ingested game_id: {last}")
        return last
    except Exception as e:
        print(f"⚠️  Could not read existing file ({e}); will do a full fetch.")
        return None

# --------------------------------------------------------
# 2. Get all team abbreviations
# --------------------------------------------------------
def get_all_team_abbreviations(client):
    """Return list of NHL team abbreviations."""
    teams_list = client.teams.teams()
    return sorted([t["abbr"] for t in teams_list if "abbr" in t])

# --------------------------------------------------------
# 3. Get game IDs *and dates* for the season
# --------------------------------------------------------
def get_all_games_with_dates(
    client,
    season: str = "20242025",
    cutoff_date: datetime.date | None = None,
) -> pd.DataFrame:
    """
    Collect unique game IDs *and* their game dates for a given season
    by iterating over team schedules.

    Parameters
    ----------
    season : str
        Eight-digit season string, e.g. '20242025'.
    cutoff_date : datetime.date or None
        If provided, only return games whose game_date is on or before this date.
        Pass ``datetime.date.today()`` to get only games played so far.

    Returns a DataFrame with ['game_id', 'game_date'].
    """
    team_abbrs = get_all_team_abbreviations(client)
    records = []

    # Determine which schedule method exists
    schedule_methods = [
        getattr(client.schedule, "team_season_schedule", None),
        getattr(client.schedule, "team_schedule", None),
        getattr(client.schedule, "schedule_by_team", None),
    ]
    schedule_method = next((m for m in schedule_methods if callable(m)), None)
    if schedule_method is None:
        raise AttributeError("No valid team schedule method found in client.schedule")

    for abbr in team_abbrs:
        print(f"Fetching schedule for {abbr} ...")
        try:
            try:
                schedule = schedule_method(team_abbr=abbr, season=season)
            except TypeError:
                schedule = schedule_method(abbr, season)

            games = schedule.get("games") if isinstance(schedule, dict) else schedule
            if not isinstance(games, list):
                print(f"⚠️ Unexpected structure for {abbr}, skipping")
                continue

            for g in games:
                gid = g.get("gamePk") or g.get("game_id") or g.get("id")
                gdate = g.get("gameDate") or g.get("game_date") or g.get("date")
                if gid and gdate:
                    records.append({"game_id": str(gid), "game_date": pd.to_datetime(gdate)})
        except Exception as e:
            print(f"❌ Failed for {abbr}: {e}")
        time.sleep(0.2)

    df_games = pd.DataFrame(records).drop_duplicates("game_id")

    # ── Regular-season filter ─────────────────────────────────────────────────
    # NHL game IDs are formatted as YYYYTTGGGG where TT encodes the game type:
    #   01 = preseason, 02 = regular season, 03 = playoffs.
    # We only want regular-season games (TT == "02") so that playoff games
    # fetched from the team schedule are never ingested into the stats model.
    before_type = len(df_games)
    df_games = df_games[
        df_games["game_id"].astype(str).str.strip().str[4:6] == "02"
    ].copy()
    filtered_type = before_type - len(df_games)
    if filtered_type:
        print(f"⚙️  Filtered out {filtered_type} non-regular-season game(s) (preseason/playoffs).")

    if cutoff_date is not None:
        before = len(df_games)
        df_games = df_games[
            pd.to_datetime(df_games["game_date"]).dt.date <= cutoff_date
        ].copy()
        print(
            f"✅ Retrieved {len(df_games)} regular-season games on or before {cutoff_date} "
            f"(filtered from {before}) for {season}"
        )
    else:
        print(f"✅ Retrieved {len(df_games)} unique regular-season games with dates for {season}")

    return df_games

# --------------------------------------------------------
# 4. Flatten play-by-play JSON for a single game
# --------------------------------------------------------
def flatten_json(obj, parent_key="", sep="."):
    """Recursively flatten nested JSON/dict structures."""
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_json(v, new_key, sep=sep).items())
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flatten_json(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, obj))
    return dict(items)

def fetch_and_flatten_all_fields(game_id: str) -> pd.DataFrame:
    """Fetch and flatten play-by-play for one game."""
    client = NHLClient()
    pbp = client.game_center.play_by_play(game_id=game_id)
    if isinstance(pbp, list) and pbp:
        pbp = pbp[0]
    plays = pbp.get("plays", [])
    if not isinstance(plays, list):
        raise ValueError(f"Unexpected structure for {game_id}: no plays list found")
    flat = [flatten_json(play) for play in plays]
    df = pd.DataFrame(flat)
    df["game_id"] = game_id
    return df

# --------------------------------------------------------
# 5. Fetch and aggregate flattened play-by-play for full season
# --------------------------------------------------------
def fetch_season_all_fields(
    client,
    season: str = "20242025",
    out_csv: str | None = None,
    cutoff_date: datetime.date | None = None,
    resume_after: str | None = None,
):
    """
    Download all games for ``season``, attach dates, and save to CSV.

    Parameters
    ----------
    season : str
        Eight-digit season string, e.g. '20242025'.
    out_csv : str or None
        Output CSV path.  Defaults to ``nhl_pbp_allfields_{YYYY}_{YYYY+1}.csv``
        in the same directory as this script.
    cutoff_date : datetime.date or None
        If set, only games on or before this date are fetched.
        Pass ``datetime.date.today()`` for a season-to-date pull.
    resume_after : str or None
        If set, skip all game IDs whose numeric value is <= this value.
        New rows are *appended* to the existing CSV rather than overwriting it.
        Obtained automatically via get_last_game_id_in_csv().
    """
    if out_csv is None:
        label = season_str_to_label(season)
        out_csv = f"nhl_pbp_allfields_{label}.csv"
    print(f"📄 Output file: {out_csv}")
    df_games = get_all_games_with_dates(client, season, cutoff_date=cutoff_date)

    # ── Filter to only new games when resuming ────────────────────────────────
    if resume_after is not None:
        resume_int = int(resume_after)
        before = len(df_games)
        df_games = df_games[
            df_games["game_id"].astype(str).str.extract(r"(\d+)", expand=False)
            .astype(float).astype(int) > resume_int
        ].copy()
        skipped = before - len(df_games)
        print(f"⏭️  Skipping {skipped} already-ingested games; {len(df_games)} new games to fetch.")

    game_ids = df_games["game_id"].tolist()

    if not game_ids:
        print("✅ Already up to date — no new games to ingest.")
        return

    print(f"🔹 Downloading play-by-play for {len(game_ids)} games in {season}")

    all_dfs = []
    for i, gid in enumerate(game_ids, 1):
        try:
            df = fetch_and_flatten_all_fields(gid)
            gdate = df_games.loc[df_games["game_id"] == gid, "game_date"].iloc[0]
            df["game_date"] = gdate
            all_dfs.append(df)
            print(f"[{i}/{len(game_ids)}] ✅ Game {gid}: {len(df)} plays")
        except Exception as e:
            print(f"[{i}/{len(game_ids)}] ❌ Game {gid}: {e}")
        time.sleep(0.25)

    if not all_dfs:
        print("⚠️ No play data collected.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined = combined.sort_values(["game_date", "game_id"])

    # Append to existing file if resuming, otherwise write fresh
    import os
    if resume_after is not None and os.path.exists(out_csv):
        # Reorder columns to match the existing file to prevent misalignment
        existing_cols = pd.read_csv(out_csv, nrows=0).columns.tolist()
        combined = combined.reindex(columns=existing_cols)
        combined.to_csv(out_csv, mode="a", header=False, index=False)
        print(f"✅ Appended {len(combined):,} new plays to {out_csv}")
    else:
        combined.to_csv(out_csv, index=False)
        print(f"✅ Saved {len(combined):,} total plays with game dates to {out_csv}")

# --------------------------------------------------------
# 6. Run end-to-end
# --------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest NHL play-by-play data for a given season."
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help=(
            "Eight-digit season string, e.g. '20242025' for the 2024-25 season. "
            "If omitted and --current is set, the active season is auto-detected."
        ),
    )
    parser.add_argument(
        "--current",
        action="store_true",
        help=(
            "Fetch only games up to and including today's date. "
            "If --season is not supplied, the currently-active NHL season is used."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: auto-derived from season string).",
    )
    args = parser.parse_args()

    # Resolve season string
    if args.season:
        season = args.season
    elif args.current:
        season = current_season_str()
        print(f"Auto-detected current season: {season}")
    else:
        season = "20242025"  # default fallback
        print(f"No --season or --current supplied; defaulting to {season}")

    # Resolve output path (needed to check for existing file)
    if args.out:
        out_csv = args.out
    else:
        label = season_str_to_label(season)
        out_csv = f"nhl_pbp_allfields_{label}.csv"

    # Resolve cutoff date
    cutoff = datetime.date.today() if args.current else None
    if cutoff:
        print(f"Fetching games up to (and including) {cutoff}")

    # For --current runs, auto-resume from the last ingested game if file exists
    resume_after = None
    if args.current:
        resume_after = get_last_game_id_in_csv(out_csv)

    fetch_season_all_fields(
        client,
        season=season,
        out_csv=out_csv,
        cutoff_date=cutoff,
        resume_after=resume_after,
    )
