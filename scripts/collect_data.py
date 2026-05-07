"""
Data Collection
======================
Pulls weekly player stats and schedules via nflreadpy (nflverse).
Saves raw data to data/raw/ for processing in the next step.

Run: python scripts/collect_data.py
"""

import nflreadpy as nfl
import pandas as pd
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# 2022-2024 → training data | 2025 → holdout test
SEASONS = [2022, 2023, 2024, 2025]

TARGET_POSITIONS = ["QB", "RB", "WR", "TE"]


def collect_weekly_stats():
    print(f"📥 Fetching weekly stats for seasons: {SEASONS}")
    weekly = nfl.load_player_stats(seasons=SEASONS).to_pandas()

    # Regular season only
    weekly = weekly[weekly["season_type"] == "REG"].copy()
    weekly = weekly[weekly["position"].isin(TARGET_POSITIONS)].copy()

    print(f"   → {len(weekly):,} rows | {weekly['player_display_name'].nunique():,} unique players")
    path = os.path.join(RAW_DIR, "weekly_stats.csv")
    weekly.to_csv(path, index=False)
    print(f"   ✅ Saved to {path}")
    return weekly


def collect_schedules():
    print("📥 Fetching schedules...")
    schedules = nfl.load_schedules(seasons=SEASONS).to_pandas()

    path = os.path.join(RAW_DIR, "schedules.csv")
    schedules.to_csv(path, index=False)
    print(f"   ✅ Saved to {path}")
    return schedules


def collect_rosters():
    print("📥 Fetching rosters...")
    try:
        rosters = nfl.load_rosters(seasons=SEASONS).to_pandas()
        rosters = rosters[rosters["position"].isin(TARGET_POSITIONS)].copy()
        path = os.path.join(RAW_DIR, "rosters.csv")
        rosters.to_csv(path, index=False)
        print(f"   ✅ Saved to {path}")
    except Exception as e:
        print(f"   ⚠️  Rosters skipped: {e}")


def inspect_weekly_columns(weekly: pd.DataFrame):
    print("\n📋 Available stat columns:")
    stat_cols = [c for c in weekly.columns if any(
        kw in c for kw in ["pass", "rush", "rec", "td", "yard", "target", "carry", "fumble", "interception"]
    )]
    for col in stat_cols:
        print(f"   {col:<45} ({weekly[col].notna().sum():,} non-null)")


def main():
    print("=" * 55)
    print("🏈 Fantasy Football Predictor — Data Collection")
    print("=" * 55)

    weekly    = collect_weekly_stats()
    schedules = collect_schedules()
    collect_rosters()

    inspect_weekly_columns(weekly)

    print("\n✅ Data collection complete! Files written to data/raw/")
    print("   Next: run scripts/process_data.py")


if __name__ == "__main__":
    main()
