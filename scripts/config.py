#!/usr/bin/env python3
"""
Configuration module — resolves paths for the WC 2026 prediction engine.

All scripts import from this module so paths are defined in one place.
The data directory can be customized via the WC2026_DATA_DIR environment
variable, otherwise defaults to ./data/26worldcup.github.io (relative to
this file).
"""

from __future__ import annotations

import os
from pathlib import Path

# Base directory of this repo (where config.py lives)
BASE_DIR = Path(__file__).resolve().parent

# Data directory: clone of https://github.com/26worldcup/26worldcup.github.io
# Override with WC2026_DATA_DIR env var, else default to ./data/26worldcup.github.io
DATA_DIR = Path(os.environ.get(
    "WC2026_DATA_DIR",
    BASE_DIR.parent / "data" / "26worldcup.github.io",
))

# Key data files
MATCHES_JSON = DATA_DIR / "public" / "data" / "matches.json"
SIM_MODEL_JSON = DATA_DIR / "public" / "data" / "sim-model.json"
TEAMS_JSON = DATA_DIR / "public" / "data" / "teams.json"
VENUES_JSON = DATA_DIR / "public" / "data" / "venues.json"
SQUADS_JSON = DATA_DIR / "public" / "data" / "squads.json"
INTL_RESULTS_CSV = DATA_DIR / "scripts" / "cache" / "intl-results.csv"

# Output directory
DOWNLOAD_DIR = BASE_DIR.parent / "download"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Cache directory for ESPN lineups
CACHE_DIR = BASE_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def check_data_available() -> bool:
    """Returns True if the data directory has the required files."""
    return MATCHES_JSON.exists() and TEAMS_JSON.exists()


def print_paths() -> None:
    """Print all resolved paths (for debugging)."""
    print(f"BASE_DIR:          {BASE_DIR}")
    print(f"DATA_DIR:          {DATA_DIR}")
    print(f"  exists:          {DATA_DIR.exists()}")
    print(f"  matches.json:    {MATCHES_JSON.exists()}")
    print(f"  teams.json:      {TEAMS_JSON.exists()}")
    print(f"  squads.json:     {SQUADS_JSON.exists()}")
    print(f"  intl-results:    {INTL_RESULTS_CSV.exists()}")
    print(f"DOWNLOAD_DIR:      {DOWNLOAD_DIR}")
    print(f"CACHE_DIR:         {CACHE_DIR}")


if __name__ == "__main__":
    print_paths()
    if not check_data_available():
        print("\n⚠ Data directory not found. Run:")
        print("  bash setup.sh")
        print("Or set WC2026_DATA_DIR to point to your clone of")
        print("  https://github.com/26worldcup/26worldcup.github.io")
