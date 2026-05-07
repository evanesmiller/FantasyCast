# FantasyCast — PPR Fantasy Football Predictor

A machine learning web app that predicts PPR fantasy points for NFL skill-position players (QB, RB, WR, TE). Built for CPSC 483 · Machine Learning.

Models are trained on 2022–2024 regular-season data and used to project the **2025 season** — a true out-of-sample test, since the 2025 season has already occurred and actual results are available for comparison.

---

## Features

### Game Mode
Enter a player name and a 2025 week number. The backend:
- Looks up the player's 2025 team (handles off-season trades)
- Finds the scheduled opponent and home/away status from the 2025 NFL schedule automatically
- Fetches that opponent's 2024 defensive rating (average PPR allowed to that position)
- Returns the predicted PPR points alongside the actual 2025 result for that week

### Season Mode
Enter a player name. The backend projects all 17 regular-season games using:
- The player's end-of-2024 rolling stats as the feature baseline (simulates pre-draft knowledge)
- Per-opponent defensive ratings from 2024 for each scheduled matchup
- The full 2025 regular-season schedule (17 games + 1 bye week, shown explicitly)

Each projected game is compared against the actual 2025 result, and summary accuracy stats (season error, % predictions within MAE) are displayed.

### Model Performance Charts
Always-visible side panel showing per-position accuracy and model comparison charts generated during training.

---

## Project Structure

```
FantasyCast/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                        # Raw CSVs from nflreadpy (gitignored)
│   └── processed/                  # Feature-engineered CSVs + feature_cols.txt (gitignored)
├── models/                         # Saved .joblib model bundles (gitignored)
├── scripts/
│   ├── collect_data.py             # Fetch 2022–2025 weekly stats + schedules via nflreadpy
│   ├── process_data.py             # PPR scoring, rolling features, train/test split
│   └── train_models.py             # Train per-position models, save best, generate charts
├── backend/
│   ├── __init__.py
│   └── main.py                     # FastAPI — /predict, /predict/season, /players, /charts
└── frontend/
    ├── index.html
    ├── main.jsx
    ├── App.jsx                     # Two-column layout, GAME/SEASON mode toggle
    ├── PlayerSearch.jsx            # Autocomplete player search
    ├── ResultCard.jsx              # Single-game result with matchup + error display
    ├── SeasonCard.jsx              # Full-season projection table with bye week row
    ├── ModelCharts.jsx             # Always-visible accuracy charts panel
    ├── vite.config.js              # Dev proxy: /predict → localhost:8000
    └── package.json
```

> `data/`, `models/`, `venv/`, and `frontend/node_modules/` are gitignored. After cloning, run the pipeline scripts and `npm install` to regenerate them locally.

---

## Local Setup

**Python 3.12 is recommended.** Python 3.14 is not yet fully supported by some ML dependencies.

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 2. Install Python dependencies
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

> **macOS + XGBoost:** XGBoost requires the OpenMP runtime on macOS. If you see a `libomp.dylib` error on startup, run `brew install libomp` first.

```bash
# 3. Install frontend dependencies
cd frontend
npm install
cd ..
```

---

## Running the Pipeline

Run these steps in order — each depends on the previous output.

```bash
# Fetch 2022–2025 weekly player stats and schedules via nflreadpy
python scripts/collect_data.py

# Compute PPR points, engineer rolling features, split train (2022–2024) / test (2025)
python scripts/process_data.py

# Train per-position models (Linear Regression, Random Forest, Gradient Boosting, XGBoost)
# Saves best model per position to models/ and generates accuracy charts
python scripts/train_models.py

# Start the FastAPI backend (http://localhost:8000)
python -m uvicorn backend.main:app --reload

# In a separate terminal — start the Vite frontend (http://localhost:5173)
cd frontend && npm run dev
```

Interactive API docs are available at `http://localhost:8000/docs` once the backend is running.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness check + per-position model load status |
| `GET`  | `/players?q={name}` | Autocomplete player search (searches 2025 roster) |
| `POST` | `/predict` | Single-game PPR prediction with optional week lookup |
| `GET`  | `/predict/season?player_name={name}` | Full 2025 season projection (17 games + bye) |
| `GET`  | `/charts/{filename}` | Serve model performance chart PNGs |

---

### `POST /predict` — Single-game prediction

No need to specify home/away or opponent — when a week is provided, the backend auto-detects the matchup from the 2025 schedule and applies the opponent's 2024 defensive rating.

**Request**
```json
{
  "player_name": "Justin Jefferson",
  "week": 8
}
```

**Response**
```json
{
  "player_name": "Justin Jefferson",
  "position": "WR",
  "predicted_ppr_points": 18.4,
  "actual_ppr_points": 21.1,
  "model_used": "GradientBoosting",
  "model_mae": 7.2,
  "note": "±7.2 pts typical error",
  "opponent": "LAR",
  "is_home": false
}
```

Omitting `week` returns a prediction using career-average features with no actual comparison.

---

### `GET /predict/season` — Full season projection

**Request**
```
GET /predict/season?player_name=Justin%20Jefferson
```

**Response**
```json
{
  "player_name": "Justin Jefferson",
  "position": "WR",
  "team": "MIN",
  "model_used": "GradientBoosting",
  "model_mae": 7.2,
  "season_total_predicted": 289.4,
  "season_total_actual": 312.7,
  "weeks": [
    { "week": 1,  "opponent": "NYG", "is_home": true,  "predicted_ppr": 17.3, "actual_ppr": 22.1, "is_bye": false },
    { "week": 6,  "opponent": "BYE", "is_home": false, "predicted_ppr": null,  "actual_ppr": null,  "is_bye": true },
    ...
  ]
}
```

---

## Edge Case Handling

The system is designed to handle real-world roster and schedule irregularities gracefully rather than crashing or returning misleading results.

### Bye Weeks (Game Mode)
When a week is requested that falls on the player's team bye, the backend detects the absence of a scheduled game and returns a structured response with `predicted_ppr_points: 0.0` and `is_bye: true` instead of throwing an error. The UI renders a result card showing "BYE WEEK" in place of the opponent, a predicted score of 0.0, and a red "Bye Week — No Game Scheduled" banner. Bye weeks are also rendered explicitly as their own rows in the season projection table so the 17-game structure is always visible.

### Did Not Play (DNP)
A player may be on an active roster for a given week but not suit up due to injury, illness, or a coach's decision. In this case the weekly stats dataset contains no row for that player that week, making their actual PPR indistinguishable from a missing value at a glance. The system handles this by checking whether a game was scheduled for the team that week: if a game exists but no actual stats are found, the week is flagged `dnp: true`.

- **Game mode:** The prediction card still displays the projected score (a legitimate pre-game projection), but a yellow "Player did not play this game" banner is shown below it. No actual/error comparison is rendered.
- **Season mode:** The Actual column shows *DNP* in italic rather than a dash. DNP weeks are excluded from both the season actual total and the accuracy badge (% within MAE), so they don't inflate or deflate the model's measured performance.

### Off-Season Trades and Team Changes
When building a season projection or game prediction, the backend first looks up the player's team from 2025 weekly data (if they have appeared in any 2025 game). Only if no 2025 entry exists does it fall back to the player's 2024 team. This means players who were traded or signed as free agents before the 2025 season are projected against the correct team's schedule.

### Players Without 2024 Data
The feature vector is built entirely from 2024 end-of-season rolling stats. If a player has no 2024 data (e.g. retired after 2023, or a 2025 rookie), the backend returns a `404` with a clear message:

```
'<name>' not found in 2024 stats. Only players with 2024 data can be projected.
```

This is surfaced as an error banner in the UI.

### Autocomplete Scope
The player search endpoint queries the 2025 stats dataset, so only players who appeared in at least one 2025 regular-season game are returned as suggestions. This prevents users from accidentally selecting a player the system cannot meaningfully project.

---

## PPR Scoring Formula

| Stat | Points |
|------|--------|
| Passing yard | 0.04 pts (25 yds = 1 pt) |
| Passing TD | +4 pts |
| Interception | −2 pts |
| Rushing yard | 0.1 pts (10 yds = 1 pt) |
| Rushing TD | +6 pts |
| Reception | +1 pt (PPR) |
| Receiving yard | 0.1 pts |
| Receiving TD | +6 pts |
| Fumble lost | −2 pts |

---

## Model Details

Four separate models are trained — one per position (QB, RB, WR, TE). For each position, four model types are evaluated:

- Linear Regression (baseline)
- Random Forest
- Gradient Boosting
- XGBoost

The model with the lowest MAE on the 2025 holdout test set is saved for that position.

**Features used at inference time:**
- 3-game and 5-game rolling averages for all key stats
- Season-to-date averages for all key stats
- Opponent's 2024 average PPR allowed to that position (`opp_def_roll5`)
- Home/away flag (`is_home`)
- Week number

**Train/test split:** 2022–2024 seasons for training, 2025 season as the holdout test — no 2025 data is seen during training.
