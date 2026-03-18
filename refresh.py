"""
refresh.py
----------
End-to-end pipeline: ingest → xG prediction → model fit → dashboard build →
validation → git push.

Usage
-----
  python refresh.py                  # Full run; push to GitHub if tests pass
  python refresh.py --dry-run        # Full run; skip the git push
  python refresh.py --skip-ingest    # Skip data ingestion (use existing CSVs)
  python refresh.py --skip-model     # Skip model refit (use existing state PKLs)
  python refresh.py --skip-build     # Skip dashboard data build (test + push only)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / "ingest_scripts" / ".venv" / "Scripts" / "python.exe")

INGEST_SCRIPT   = str(ROOT / "ingest_scripts" / "ingest_2425.py")
PREDICT_XG      = str(ROOT / "ingest_scripts" / "predict_xg.py")
RUN_MODEL       = str(ROOT / "run_model.py")
BUILD_DATA      = str(ROOT / "dashboard" / "build_data.py")

ALLFIELDS_CSV   = str(ROOT / "ingest_scripts" / "nhl_pbp_allfields_2025_2026.csv")
WITH_XG_CSV     = str(ROOT / "ingest_scripts" / "nhl_pbp_2025_2026_with_xg.csv")

DASHBOARD_DATA  = ROOT / "dashboard" / "data"
SITUATIONS      = ["all", "even", "pp"]

# Files the dashboard requires in every situation subdirectory
REQUIRED_FILES = [
    "goalie_summary.csv",
    "goalie_weekly.csv",
    "hex_goalie.csv",
    "hex_league.csv",
    "hex_shooter.csv",
    "meta.json",
    "shooter_summary.csv",
    "shooter_weekly.csv",
]

# Minimum required columns per CSV (spot-check; not exhaustive)
REQUIRED_COLS: dict[str, list[str]] = {
    "goalie_summary.csv":  ["player_name", "gsax"],
    "shooter_summary.csv": ["shooter_id", "fsax"],
    "goalie_weekly.csv":   ["goalie_id"],
    "shooter_weekly.csv":  ["shooter_id"],
    "hex_goalie.csv":      ["goalie_id"],
    "hex_shooter.csv":     ["shooter_id"],
    "hex_league.csv":      [],
}

# Git paths to stage (gitignored large files are excluded automatically)
GIT_STAGE_PATHS = [
    "dashboard/data",
    "model_output/player_names_cache.json",
    "model_output/player_info_cache.json",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list[str], label: str) -> None:
    """Run a subprocess command; exit with its return code on failure."""
    log.info("▶  %s", label)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        log.error("✗  %s failed (exit %d)", label, result.returncode)
        sys.exit(result.returncode)
    log.info("✓  %s done", label)


# ── Validation ────────────────────────────────────────────────────────────────

def test_dashboard_data() -> bool:
    """
    Validate all required dashboard output files.

    Checks:
      - Every expected file exists and is non-empty.
      - Every CSV is readable and has at least one row.
      - Every CSV contains its required columns.
      - Every meta.json parses as valid JSON.

    Returns True if all checks pass, False otherwise.
    """
    all_ok = True

    for sit in SITUATIONS:
        sit_dir = DASHBOARD_DATA / sit
        for fname in REQUIRED_FILES:
            fpath = sit_dir / fname

            if not fpath.exists():
                log.error("MISSING  %s", fpath.relative_to(ROOT))
                all_ok = False
                continue

            if fpath.stat().st_size == 0:
                log.error("EMPTY    %s", fpath.relative_to(ROOT))
                all_ok = False
                continue

            if fname.endswith(".csv"):
                try:
                    df = pd.read_csv(fpath)
                except Exception as exc:
                    log.error("UNREADABLE  %s: %s", fpath.relative_to(ROOT), exc)
                    all_ok = False
                    continue

                if len(df) == 0:
                    log.error("NO ROWS  %s", fpath.relative_to(ROOT))
                    all_ok = False
                    continue

                for col in REQUIRED_COLS.get(fname, []):
                    if col not in df.columns:
                        log.error(
                            "MISSING COLUMN '%s' in %s",
                            col,
                            fpath.relative_to(ROOT),
                        )
                        all_ok = False

            elif fname.endswith(".json"):
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    if not data:
                        log.error("EMPTY JSON  %s", fpath.relative_to(ROOT))
                        all_ok = False
                except Exception as exc:
                    log.error(
                        "INVALID JSON  %s: %s", fpath.relative_to(ROOT), exc
                    )
                    all_ok = False

    if all_ok:
        log.info("✓  All dashboard data checks passed")
    else:
        log.error("✗  Validation FAILED — not pushing to GitHub")

    return all_ok


# ── Git ───────────────────────────────────────────────────────────────────────

def git_push(dry_run: bool) -> None:
    """Stage changed dashboard data files and push to the current branch."""
    if dry_run:
        log.info("--dry-run: skipping git add / commit / push")
        return

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Stage only the data output paths (gitignore handles large raw files)
    subprocess.run(["git", "add", "--", *GIT_STAGE_PATHS], cwd=str(ROOT), check=True)

    # Nothing to commit → already up to date
    staged = subprocess.run(
        ["git", "diff", "--staged", "--quiet"],
        cwd=str(ROOT),
    )
    if staged.returncode == 0:
        log.info("No changes staged — dashboard already up to date.")
        return

    subprocess.run(
        ["git", "commit", "-m", f"Auto-refresh: {timestamp}"],
        cwd=str(ROOT),
        check=True,
    )
    subprocess.run(["git", "push"], cwd=str(ROOT), check=True)
    log.info("✓  Pushed to GitHub")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh NHL IRT dashboard data.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline but skip the final git push.",
    )
    p.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip data ingestion; use existing allfields / xG CSVs.",
    )
    p.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model refit; use existing state PKL files.",
    )
    p.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip dashboard data build; run tests and (optionally) push only.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("NHL Dashboard Refresh  (dry_run=%s)", args.dry_run)
    log.info("=" * 60)

    # ── 1. Ingest ──────────────────────────────────────────────────────────
    if not args.skip_ingest:
        run(
            [
                PYTHON, INGEST_SCRIPT,
                "--current",
                "--out", ALLFIELDS_CSV,
            ],
            "Ingest play-by-play (current season, incremental)",
        )
        run(
            [PYTHON, PREDICT_XG, ALLFIELDS_CSV],
            "Predict xG for current season",
        )
    else:
        log.info("–  Skipping ingest (--skip-ingest)")

    # ── 2. Model fit ────────────────────────────────────────────────────────
    if not args.skip_model:
        run(
            [PYTHON, RUN_MODEL, "--run-all"],
            "Fit IRT model (all / even / pp)",
        )
    else:
        log.info("–  Skipping model fit (--skip-model)")

    # ── 3. Build dashboard data ─────────────────────────────────────────────
    if not args.skip_build:
        run(
            [PYTHON, BUILD_DATA, "--run-all"],
            "Build dashboard data CSVs",
        )
    else:
        log.info("–  Skipping dashboard build (--skip-build)")

    # ── 4. Validate ─────────────────────────────────────────────────────────
    log.info("Running dashboard data validation …")
    if not test_dashboard_data():
        sys.exit(1)

    # ── 5. Push to GitHub ───────────────────────────────────────────────────
    git_push(dry_run=args.dry_run)

    log.info("=" * 60)
    log.info("Refresh complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
