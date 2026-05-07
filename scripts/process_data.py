"""
Data Processing & Feature Engineering
======================================
- Computes PPR fantasy points
- Engineers rolling averages (3-game and 5-game windows)
- Adds season-to-date averages
- Adds opponent defense strength (rolling PPR allowed per position)
- Adds home/away flag
- Train: 2022-2024  |  Test: 2025

Run: python scripts/process_data.py
"""

import pandas as pd
import numpy as np
import os

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

ROLLING_WINDOWS = [3, 5]

# nflreadpy uses "passing_interceptions" (not "interceptions")
# and "team" (not "recent_team")
RAW_STAT_COLS = [
    "passing_yards", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds", "rushing_fumbles_lost",
    "carries",
    "receptions", "receiving_yards", "receiving_tds", "receiving_fumbles_lost",
    "targets", "target_share", "wopr",
]


# ---------------------------------------------------------------------------
# PPR Scoring
# ---------------------------------------------------------------------------

def compute_ppr_points(df: pd.DataFrame) -> pd.Series:
    pts = pd.Series(0.0, index=df.index)
    pts += df.get("passing_yards",          0).fillna(0) * 0.04
    pts += df.get("passing_tds",            0).fillna(0) * 4
    pts += df.get("passing_interceptions",  0).fillna(0) * -2
    pts += df.get("rushing_yards",          0).fillna(0) * 0.1
    pts += df.get("rushing_tds",            0).fillna(0) * 6
    pts += df.get("receptions",             0).fillna(0) * 1.0
    pts += df.get("receiving_yards",        0).fillna(0) * 0.1
    pts += df.get("receiving_tds",          0).fillna(0) * 6
    pts += df.get("rushing_fumbles_lost",   0).fillna(0) * -2
    pts += df.get("receiving_fumbles_lost", 0).fillna(0) * -2
    return pts.round(2)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).copy()
    roll_cols = [c for c in RAW_STAT_COLS + ["fantasy_points_ppr"] if c in df.columns]

    for col in roll_cols:
        grp = df.groupby("player_id")[col]

        for w in ROLLING_WINDOWS:
            df[f"roll{w}_{col}"] = grp.transform(
                lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean()
            )

        df[f"season_avg_{col}"] = (
            df.groupby(["player_id", "season"])[col]
            .transform(lambda x: x.shift(1).expanding(1).mean())
        )

    return df


def add_opponent_defense(df: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Rolling 5-game avg of PPR points allowed by each defense to each position."""
    if "opponent_team" not in df.columns:
        df["opp_def_roll5"] = df["fantasy_points_ppr"].mean()
        return df

    def_stats = (
        df.groupby(["opponent_team", "position", "season", "week"])["fantasy_points_ppr"]
        .mean()
        .reset_index()
        .rename(columns={"fantasy_points_ppr": "pos_pts_allowed"})
    )

    def_stats = def_stats.sort_values(["opponent_team", "position", "season", "week"])
    def_stats["opp_def_roll5"] = (
        def_stats.groupby(["opponent_team", "position"])["pos_pts_allowed"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df = df.merge(
        def_stats[["opponent_team", "position", "season", "week", "opp_def_roll5"]],
        on=["opponent_team", "position", "season", "week"],
        how="left",
    )

    df["opp_def_roll5"] = df["opp_def_roll5"].fillna(df["fantasy_points_ppr"].mean())
    return df


def add_home_away(df: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    # nflreadpy weekly stats use "team" (not "recent_team")
    team_col = "team" if "team" in df.columns else "recent_team"

    home_teams = schedules[["season", "week", "home_team"]].copy()
    home_teams["is_home"] = 1
    away_teams = schedules[["season", "week", "away_team"]].copy().rename(
        columns={"away_team": "home_team"})
    away_teams["is_home"] = 0

    team_home = pd.concat([home_teams, away_teams]).rename(
        columns={"home_team": team_col})
    df = df.merge(team_home, on=["season", "week", team_col], how="left")
    df["is_home"] = df["is_home"].fillna(0).astype(int)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    roll_cols       = [c for c in df.columns if any(c.startswith(f"roll{w}_") for w in ROLLING_WINDOWS)]
    season_avg_cols = [c for c in df.columns if c.startswith("season_avg_")]
    other           = ["is_home", "week", "opp_def_roll5"]
    return [c for c in roll_cols + season_avg_cols + other if c in df.columns]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("🏈 Fantasy Football Predictor — Data Processing")
    print("=" * 55)

    print("\n📂 Loading raw data...")
    weekly    = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"))
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    print(f"   Weekly rows: {len(weekly):,}")

    print("\n⚙️  Computing PPR fantasy points...")
    weekly["fantasy_points_ppr"] = compute_ppr_points(weekly)
    print(f"   Mean pts: {weekly['fantasy_points_ppr'].mean():.2f} | "
          f"Max pts: {weekly['fantasy_points_ppr'].max():.2f}")

    weekly = weekly[weekly["fantasy_points_ppr"] > 0].copy()
    print(f"   Rows after dropping DNPs: {len(weekly):,}")

    print("\n⚙️  Adding rolling averages (3-game, 5-game) + season averages...")
    weekly = add_rolling_features(weekly)

    print("⚙️  Adding home/away flag...")
    weekly = add_home_away(weekly, schedules)

    print("⚙️  Adding opponent defense strength...")
    weekly = add_opponent_defense(weekly, schedules)

    feature_cols = get_feature_cols(weekly)
    target_col   = "fantasy_points_ppr"

    before = len(weekly)
    weekly = weekly.dropna(subset=feature_cols + [target_col])
    print(f"\n   Dropped {before - len(weekly):,} rows with missing features")
    print(f"   Final dataset: {len(weekly):,} rows, {len(feature_cols)} features")

    weekly.to_csv(os.path.join(PROC_DIR, "processed.csv"), index=False)
    print(f"\n   ✅ Saved processed.csv")

    with open(os.path.join(PROC_DIR, "feature_cols.txt"), "w") as f:
        f.write("\n".join(feature_cols))
    print(f"   ✅ Saved feature_cols.txt ({len(feature_cols)} features)")

    print("\n⚙️  Splitting train/test (2022-2024 train | 2025 test)...")
    train = weekly[weekly["season"] < 2025]
    test  = weekly[weekly["season"] == 2025]
    print(f"   Train: {len(train):,} rows | Test: {len(test):,} rows")

    train.to_csv(os.path.join(PROC_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(PROC_DIR, "test.csv"), index=False)
    print(f"   ✅ Saved train.csv and test.csv")

    print("\n✅ Processing complete! Next: run scripts/train_models.py")


if __name__ == "__main__":
    main()
