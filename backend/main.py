"""
Backend — FastAPI Prediction API
==================================
Prediction strategy:
  - Feature baseline : 2024 end-of-season player stats  (pre-draft 2025 information)
  - Actual comparison: 2025 weekly results               (true holdout — models trained on 2022-2024)
  - Season schedule  : 2025 regular-season schedule

Data source: nflreadpy (nflverse) — replaces deprecated nfl_data_py

Endpoints:
  GET  /health               → liveness check + per-position model status
  GET  /players              → search players by name
  POST /predict              → predict single-game PPR; compare vs 2025 actual when week given
  GET  /predict/season       → project all 17 games of 2025; compare vs 2025 actuals
  GET  /charts/{file}        → serve model performance charts
"""

import nflreadpy as nfl
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Fantasy Football Predictor API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
POSITIONS  = ["QB", "RB", "WR", "TE"]

# nflreadpy uses "passing_interceptions" and "team" (not "recent_team")
_STAT_COLS = [
    "passing_yards", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds", "rushing_fumbles_lost",
    "carries",
    "receptions", "receiving_yards", "receiving_tds", "receiving_fumbles_lost",
    "targets", "target_share", "wopr",
    "fantasy_points_ppr",
]

# ---------------------------------------------------------------------------
# Load position-specific models on startup
# ---------------------------------------------------------------------------

model_bundles: dict = {}

@app.on_event("startup")
def load_models():
    global model_bundles
    for pos in POSITIONS:
        path_new = os.path.join(MODELS_DIR, f"models_{pos}.joblib")
        path_old = os.path.join(MODELS_DIR, f"best_model_{pos}.joblib")
        if os.path.exists(path_new):
            bundle = joblib.load(path_new)
            model_bundles[pos] = bundle
            print(f"✅ {pos}: best={bundle['best']}, {len(bundle['models'])} models loaded")
        elif os.path.exists(path_old):
            old = joblib.load(path_old)
            model_bundles[pos] = {
                "models":       {old["model_name"]: old["model"]},
                "best":         old["model_name"],
                "feature_cols": old["feature_cols"],
                "metrics":      {old["model_name"]: old["metrics"]},
            }
            print(f"⚠️  {pos}: legacy bundle ({old['model_name']} only)")
        else:
            print(f"⚠️  No model for {pos} — run train_models.py first.")


def _resolve_model(bundle: dict, model_name: str | None) -> tuple:
    """Returns (model, name_used, mae) — uses override or position-best."""
    if model_name and model_name in bundle["models"]:
        return bundle["models"][model_name], model_name, bundle["metrics"][model_name]["MAE"]
    best = bundle["best"]
    return bundle["models"][best], best, bundle["metrics"][best]["MAE"]


# ---------------------------------------------------------------------------
# Data caches
# ---------------------------------------------------------------------------

_cache_2024:       pd.DataFrame = None   # feature baseline for all predictions
_cache_2025:       pd.DataFrame = None   # actuals (holdout) + player search
_schedule_2025:    pd.DataFrame = None   # 2025 regular-season schedule
_opp_defense_2024: dict         = {}     # (team, position) → 2024 avg PPR allowed


def _load_weekly(seasons: list[int]) -> pd.DataFrame:
    """Load regular-season weekly player stats via nflreadpy."""
    df = nfl.load_player_stats(seasons=seasons).to_pandas()
    df = df[df["season_type"] == "REG"].copy()
    df = df[df["position"].isin(POSITIONS)].copy()
    if "fantasy_points_ppr" not in df.columns or df["fantasy_points_ppr"].isna().all():
        df["fantasy_points_ppr"] = _compute_ppr(df)
    return df


def _compute_ppr(df: pd.DataFrame) -> pd.Series:
    return (
        df.get("passing_yards",          pd.Series(0, index=df.index)).fillna(0) * 0.04
        + df.get("passing_tds",          pd.Series(0, index=df.index)).fillna(0) * 4
        + df.get("passing_interceptions",pd.Series(0, index=df.index)).fillna(0) * -2
        + df.get("rushing_yards",        pd.Series(0, index=df.index)).fillna(0) * 0.1
        + df.get("rushing_tds",          pd.Series(0, index=df.index)).fillna(0) * 6
        + df.get("receptions",           pd.Series(0, index=df.index)).fillna(0) * 1.0
        + df.get("receiving_yards",      pd.Series(0, index=df.index)).fillna(0) * 0.1
        + df.get("receiving_tds",        pd.Series(0, index=df.index)).fillna(0) * 6
    )


def get_player_stats_2024() -> pd.DataFrame:
    """2024 player stats — feature baseline (simulates pre-draft 2025 knowledge)."""
    global _cache_2024
    if _cache_2024 is None:
        print("📥 Fetching 2024 player stats (feature baseline)...")
        _cache_2024 = _load_weekly([2024])
    return _cache_2024


def get_player_stats_2025() -> pd.DataFrame:
    """2025 player stats — ground truth for actual PPR comparisons."""
    global _cache_2025
    if _cache_2025 is None:
        print("📥 Fetching 2025 player stats (actuals)...")
        _cache_2025 = _load_weekly([2025])
    return _cache_2025


def get_player_stats_cache() -> pd.DataFrame:
    """Used by /players search — returns 2025 data for current roster."""
    return get_player_stats_2025()


def get_schedule_2025() -> pd.DataFrame:
    global _schedule_2025
    if _schedule_2025 is None:
        print("📥 Fetching 2025 NFL schedule...")
        sched = nfl.load_schedules(seasons=[2025]).to_pandas()
        _schedule_2025 = sched[sched["game_type"] == "REG"].copy()
    return _schedule_2025


def get_opp_defense_2024() -> dict:
    """(opponent_team, position) → average PPR allowed in 2024."""
    global _opp_defense_2024
    if not _opp_defense_2024:
        weekly = get_player_stats_2024()
        if "opponent_team" in weekly.columns:
            _opp_defense_2024 = (
                weekly.groupby(["opponent_team", "position"])["fantasy_points_ppr"]
                .mean()
                .to_dict()
            )
    return _opp_defense_2024


# ---------------------------------------------------------------------------
# Feature vector helpers
# ---------------------------------------------------------------------------

def compute_rolling_stats(player_rows: pd.DataFrame) -> tuple[dict, str]:
    """Rolling/season-avg features from a player's game history."""
    df  = player_rows.sort_values("week").copy()
    row = {}
    for col in _STAT_COLS:
        vals = df[col].fillna(0).values if col in df.columns else np.zeros(1)
        row[f"roll3_{col}"]      = float(np.mean(vals[-3:])) if len(vals) > 0 else 0.0
        row[f"roll5_{col}"]      = float(np.mean(vals[-5:])) if len(vals) > 0 else 0.0
        row[f"season_avg_{col}"] = float(np.mean(vals))      if len(vals) > 0 else 0.0
    position = str(df["position"].iloc[-1]) if "position" in df.columns else "RB"
    return row, position


def assemble_vector(rolling_row: dict, feature_cols: list,
                    opp_def: float, is_home: int, week: int, position: str) -> np.ndarray:
    row = dict(rolling_row)
    row["opp_def_roll5"] = opp_def
    row["is_home"]       = is_home
    row["week"]          = week
    for pos in POSITIONS:
        row[f"pos_{pos}"] = 1 if position == pos else 0
    return np.array([row.get(col, 0.0) for col in feature_cols]).reshape(1, -1)


def _lookup_player_2024(name: str) -> tuple[pd.DataFrame, str]:
    """Returns (player_rows, display_name) from 2024 data, or raises 404."""
    w24  = get_player_stats_2024()
    mask = w24["player_display_name"].str.lower().str.contains(name.lower(), na=False)
    rows = w24[mask]
    if rows.empty:
        raise HTTPException(404,
            f"'{name}' not found in 2024 stats. Only players with 2024 data can be projected.")
    display = rows["player_display_name"].mode()[0]
    return rows[rows["player_display_name"] == display], display


def _lookup_actual_2025(display_name: str, week: int) -> float | None:
    w25      = get_player_stats_2025()
    week_row = w25[(w25["player_display_name"] == display_name) & (w25["week"] == week)]
    if week_row.empty:
        return None
    val = week_row["fantasy_points_ppr"].iloc[0]
    return round(float(val), 2) if pd.notna(val) else None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    player_name: str
    week:        int | None = None
    model_name:  str | None = None


class PredictResponse(BaseModel):
    player_name:          str
    position:             str
    predicted_ppr_points: float
    actual_ppr_points:    float | None = None
    model_used:           str
    model_mae:            float
    note:                 str
    opponent:             str  | None = None
    is_home:              bool | None = None
    dnp:                  bool        = False
    is_bye:               bool        = False


class SeasonWeek(BaseModel):
    week:          int
    opponent:      str
    is_home:       bool
    predicted_ppr: float | None = None   # None for bye weeks
    actual_ppr:    float | None = None
    is_bye:        bool         = False
    dnp:           bool         = False  # player was on roster but did not play


class SeasonPredictResponse(BaseModel):
    player_name:            str
    position:               str
    team:                   str
    model_used:             str
    model_mae:              float
    weeks:                  list[SeasonWeek]
    season_total_predicted: float
    season_total_actual:    float | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

_CHART_ALLOWLIST = {"model_comparison.png", "position_accuracy.png", "feature_importance.png", "regression_lines.png"}

@app.get("/charts/{filename}")
def get_chart(filename: str):
    if filename not in _CHART_ALLOWLIST:
        raise HTTPException(404)
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Chart not found — run train_models.py first.")
    return FileResponse(path, media_type="image/png")


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": {pos: pos in model_bundles for pos in POSITIONS},
    }


@app.get("/players")
def search_players(q: str):
    weekly  = get_player_stats_cache()   # 2025 — current roster
    matches = (
        weekly[weekly["player_display_name"].str.lower().str.contains(q.lower(), na=False)]
        ["player_display_name"].drop_duplicates().head(10).tolist()
    )
    return {"results": matches}


@app.get("/models")
def get_models():
    return {
        pos: {
            "best":    bundle["best"],
            "options": list(bundle["models"].keys()),
            "metrics": {
                name: {"MAE": round(m["MAE"], 3), "R2": round(m["R2"], 3)}
                for name, m in bundle["metrics"].items()
            },
        }
        for pos, bundle in model_bundles.items()
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not model_bundles:
        raise HTTPException(503, "No models loaded. Run train_models.py first.")

    player_rows, display_name = _lookup_player_2024(req.player_name)

    feature_cols       = next(iter(model_bundles.values()))["feature_cols"]
    rolling_stats, pos = compute_rolling_stats(player_rows)

    if pos not in model_bundles:
        raise HTTPException(503, f"No model loaded for position {pos}.")

    bundle = model_bundles[pos]
    model, model_used, model_mae = _resolve_model(bundle, req.model_name)
    week   = req.week if req.week is not None else 1

    # Determine player's 2025 team (handles off-season moves)
    w25         = get_player_stats_2025()
    player_2025 = w25[w25["player_display_name"] == display_name]
    team_col    = "team" if "team" in player_rows.columns else "recent_team"

    if not player_2025.empty and "team" in player_2025.columns:
        team = player_2025["team"].mode()[0]
    elif team_col in player_rows.columns:
        team = player_rows[team_col].mode()[0]
    else:
        team = None

    # Defaults — overridden when schedule lookup succeeds
    is_home  = 1
    opponent = None
    opp_def  = rolling_stats.get("season_avg_fantasy_points_ppr", 10.0)

    if team and req.week is not None:
        schedule   = get_schedule_2025()
        team_sched = schedule[(schedule["home_team"] == team) | (schedule["away_team"] == team)]
        week_game  = team_sched[team_sched["week"] == week]
        if week_game.empty:
            return PredictResponse(
                player_name=display_name,
                position=pos,
                predicted_ppr_points=0.0,
                model_used=model_used,
                model_mae=round(model_mae, 2),
                note=f"Week {week} is {team}'s bye week.",
                is_bye=True,
            )
        game     = week_game.iloc[0]
        is_home  = int(game["home_team"] == team)
        opponent = str(game["away_team"] if is_home else game["home_team"])
        opp_def  = get_opp_defense_2024().get((opponent, pos), opp_def)

    X          = assemble_vector(rolling_stats, feature_cols, opp_def, is_home, week, pos)
    prediction = max(0.0, round(float(model.predict(X)[0]), 2))
    actual_ppr = _lookup_actual_2025(display_name, week) if req.week is not None else None
    dnp        = (opponent is not None and actual_ppr is None)

    return PredictResponse(
        player_name=display_name,
        position=pos,
        predicted_ppr_points=prediction,
        actual_ppr_points=actual_ppr,
        model_used=model_used,
        model_mae=round(model_mae, 2),
        note=f"±{model_mae:.1f} pts typical error",
        opponent=opponent,
        is_home=bool(is_home) if opponent else None,
        dnp=dnp,
    )


@app.get("/predict/season", response_model=SeasonPredictResponse)
def predict_season(player_name: str, model_name: str | None = None):
    if not model_bundles:
        raise HTTPException(503, "No models loaded. Run train_models.py first.")

    player_rows, display_name = _lookup_player_2024(player_name)

    feature_cols       = next(iter(model_bundles.values()))["feature_cols"]
    rolling_stats, pos = compute_rolling_stats(player_rows)

    if pos not in model_bundles:
        raise HTTPException(503, f"No model loaded for position {pos}.")

    bundle      = model_bundles[pos]
    model, model_used, model_mae = _resolve_model(bundle, model_name)
    opp_defense = get_opp_defense_2024()
    schedule    = get_schedule_2025()
    w25         = get_player_stats_2025()

    # Player's 2025 team — handles off-season moves
    mask_25     = w25["player_display_name"] == display_name
    player_2025 = w25[mask_25]
    team_col    = "team" if "team" in player_rows.columns else "recent_team"

    if not player_2025.empty and "team" in player_2025.columns:
        team = player_2025["team"].mode()[0]
    elif team_col in player_rows.columns:
        team = player_rows[team_col].mode()[0]
    else:
        raise HTTPException(400, "Could not determine player's 2025 team.")

    team_sched = schedule[
        (schedule["home_team"] == team) | (schedule["away_team"] == team)
    ].sort_values("week")

    if team_sched.empty:
        raise HTTPException(404, f"No 2025 schedule found for team '{team}'.")

    global_avg    = rolling_stats.get("season_avg_fantasy_points_ppr", 10.0)
    game_by_week  = {int(g["week"]): g for _, g in team_sched.iterrows()}
    weeks_result  = []

    for week_num in range(1, 19):
        if week_num not in game_by_week:
            weeks_result.append(SeasonWeek(
                week=week_num, opponent="BYE", is_home=False, is_bye=True))
            continue

        game     = game_by_week[week_num]
        is_home  = bool(game["home_team"] == team)
        opponent = str(game["away_team"] if is_home else game["home_team"])

        opp_def = opp_defense.get((opponent, pos), global_avg)
        X       = assemble_vector(rolling_stats, feature_cols, opp_def, int(is_home), week_num, pos)
        pred    = max(0.0, round(float(model.predict(X)[0]), 2))
        actual  = _lookup_actual_2025(display_name, week_num)

        weeks_result.append(SeasonWeek(
            week=week_num, opponent=opponent, is_home=is_home,
            predicted_ppr=pred,
            actual_ppr=actual,
            dnp=(actual is None),  # scheduled game but no stats → player did not play
        ))

    total_pred   = round(sum(w.predicted_ppr for w in weeks_result if not w.is_bye), 2)
    actuals      = [w.actual_ppr for w in weeks_result if not w.is_bye and w.actual_ppr is not None]
    total_actual = round(sum(actuals), 2) if actuals else None

    return SeasonPredictResponse(
        player_name=display_name,
        position=pos,
        team=team,
        model_used=model_used,
        model_mae=round(model_mae, 2),
        weeks=weeks_result,
        season_total_predicted=total_pred,
        season_total_actual=total_actual,
    )
