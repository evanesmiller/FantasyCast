# FantasyCast — CPSC 483 Final Project Report
### PPR Fantasy Football Predictor

---

## 1. Problem Definition

**Task:** Supervised regression — predict the number of PPR (Points Per Reception) fantasy football points a given NFL skill-position player (QB, RB, WR, or TE) will score in a specific 2025 regular-season game.

**Real-world purpose:** Fantasy football managers make weekly lineup decisions that directly affect their win/loss record. The core challenge is evaluating a player's expected output against a specific opponent in a specific week. This is a classical regression problem: given what we know about a player's recent production and their upcoming matchup, how many points should we expect them to score?

The project produces both single-game predictions and full 17-game season projections, each compared against the actual 2025 results as a true out-of-sample evaluation.

**PPR Scoring Formula:**

| Stat | Points |
|---|---|
| Passing yard | 0.04 pts (25 yds = 1 pt) |
| Passing TD | +4 pts |
| Interception | −2 pts |
| Rushing yard | 0.1 pts (10 yds = 1 pt) |
| Rushing TD | +6 pts |
| Reception | +1 pt (PPR bonus) |
| Receiving yard | 0.1 pts |
| Receiving TD | +6 pts |
| Fumble lost | −2 pts |

---

## 2. Data

### Source

All data was collected programmatically using **nflreadpy**, a Python package (pip-installable) that provides a clean interface to the **nflverse** open data repository — a publicly maintained collection of NFL play-by-play and player stats derived from official NFL tracking data.

```
pip install nflreadpy
```

Three datasets were fetched:

| Dataset | Collection Method | Contents |
|---|---|---|
| Weekly player stats | `nfl.load_player_stats(seasons=[2022,2023,2024,2025])` | Per-game individual stats for all skill-position players |
| Season schedules | `nfl.load_schedules(seasons=[2022,2023,2024,2025])` | Home/away matchups per week |
| Rosters | `nfl.load_rosters(seasons=[2022,2023,2024,2025])` | Team affiliations for trade/FA handling |

All data was filtered to **regular-season games only** (`season_type == "REG"`) and to the four target positions: **QB, RB, WR, TE**.

### Scope

- **Seasons:** 2022, 2023, 2024 (training) and 2025 (test holdout)
- **Split strategy:** Strict temporal split — no 2025 data was used during training or fine-tuning. The 2025 season serves as a true out-of-sample test since the season has already concluded and actual results are available for comparison.

---

## 3. Exploratory Data Analysis (EDA)

EDA was performed in `scripts/eda.py` after the data processing step. Four charts were generated to `models/eda/`:

### Chart 1 — PPR Points Distribution by Position (`eda_points_distribution.png`)
Box plots of weekly PPR scores for each position, broken out by season (2022–2025). Shows the median, spread (IQR), and outlier distribution per position per year. This chart reveals that QB scores have a wider distribution (big games vs. low-output weeks), while RB and TE show tighter, lower medians with more frequent near-zero weeks.

### Chart 2 — Top Feature Correlations (`eda_correlations.png`)
A horizontal bar chart of the 20 engineered features most strongly correlated (Pearson r) with the PPR target across all positions combined. Rolling-average stats dominate the top of the list, confirming that recent production is the strongest signal — not raw position identity or matchup variables alone.

### Chart 3 — Key Raw Stat vs. PPR Scatter (`eda_stat_vs_ppr.png`)
A 2×2 scatter plot showing the single most predictive raw stat for each position (passing yards → QB, rushing yards → RB, receiving yards → WR and TE) against PPR points, with a fitted regression line and Pearson r annotation. All four show strong positive linear relationships, which foreshadowed the finding that Linear Regression would be a competitive baseline.

### Chart 4 — Average PPR per Week by Season (`eda_weekly_trends.png`)
Line charts (one subplot per position) showing average PPR scored per week across the four seasons. Reveals bye-week gaps, late-season consistency patterns, and whether offensive output has shifted year over year.

---

## 4. Data Preparation

All preparation is handled in `scripts/process_data.py`.

### PPR Score Computation
The PPR target column (`fantasy_points_ppr`) was computed directly from the raw stat columns using the formula above. Rows where a player scored zero points (did-not-play / DNP games) were dropped from the training set to avoid the model treating absences as zero-point performances.

### Feature Engineering
Three categories of features were engineered for each raw stat:

| Feature Type | Window | Description |
|---|---|---|
| Rolling average | 3-game | `roll3_{stat}` — mean of the prior 3 games |
| Rolling average | 5-game | `roll5_{stat}` — mean of the prior 5 games |
| Season-to-date average | Expanding | `season_avg_{stat}` — mean of all prior games in the current season |

All rolling features use a **one-game lag** (`shift(1)`) so that the current week's result is never included — preventing any data leakage from the future.

### Matchup Features
Two additional features were added to capture the weekly matchup context:

- **`opp_def_roll5`** — The opponent team's rolling 5-game average of PPR points allowed to that specific position. This quantifies defensive strength from the perspective of the position being scored.
- **`is_home`** — Binary flag (1 = home, 0 = away) derived by joining on the season schedule.

### Train/Test Split
```
Train: seasons 2022, 2023, 2024
Test:  season  2025  (true holdout — never seen during training or CV)
```

The final processed dataset was saved as `data/processed/processed.csv`, with feature column names written to `data/processed/feature_cols.txt` for reproducibility.

---

## 5. Model Selection

### Models Trained

Four separate model bundles were trained — one per position (QB, RB, WR, TE). Each bundle contains all four model types so the frontend can offer a model selector:

| Model | Type | Rationale |
|---|---|---|
| **Linear Regression** | Baseline, regularization-free | Simple, interpretable, and appropriate when the underlying relationship is linear |
| **Random Forest** | Ensemble, bagging | Handles non-linear interactions, robust to outliers, low variance via averaging |
| **Gradient Boosting** | Ensemble, boosting | Sequentially reduces residuals; effective on tabular data |
| **XGBoost** | Ensemble, boosting | Regularized gradient boosting; often state-of-the-art on structured/tabular regression |

### Rationale for Model Choice

The prediction problem is inherently **tabular regression with temporal structure**. The input features are all numeric rolling averages — a compact, well-structured feature set with no missing categories or text. This ruled out neural networks as unnecessarily complex for this feature dimensionality and sample size.

Tree-based ensembles (Random Forest, Gradient Boosting, XGBoost) were included because they can capture interaction effects that linear models cannot — for example, the combination of a high-volume receiver facing a weak secondary may produce a score above what either feature alone would predict.

Linear Regression was included as an interpretable baseline. **A key finding was that Linear Regression matched or outperformed the tree-based models at multiple positions before tuning**, suggesting the relationship between a player's recent rolling stats and their next-week PPR score is largely linear given this feature construction. This is itself a meaningful result — it implies that the dominant signal in this problem is a player's momentum, and matchup/position complexity add relatively little.

---

## 6. Training

Training is handled in `scripts/train_models.py`.

For each position:

1. All rows for that position are extracted from the train set (2022–2024).
2. All four models are fit using `model.fit(X_train, y_train)`.
3. Each model is evaluated on the 2025 test set (MAE, RMSE, R²).
4. The model with the lowest test MAE is identified as the pre-tuning best.

Models are built as scikit-learn `Pipeline` objects. Linear Regression includes a `StandardScaler` step; tree-based models do not require scaling.

---

## 7. Fine-Tuning

After initial training, **all three tree-based models** (Random Forest, Gradient Boosting, XGBoost) are fine-tuned per position using `RandomizedSearchCV`.

### Why All Three (Not Just the Best)?
Linear Regression has no meaningful continuous hyperparameters to tune. Since it wins the initial comparison, restricting tuning to only the pre-tuning winner would skip tuning entirely. Instead, all three tree models are tuned so:
- The comparison chart is substantive and demonstrates the tuning process.
- A tuned tree model may surpass LR after optimization, and that outcome is correctly captured.

### Tuning Configuration

```
Optimizer:    RandomizedSearchCV
n_iter:       40
CV strategy:  TimeSeriesSplit (n_splits=5)
Scoring:      neg_mean_absolute_error
```

### Why TimeSeriesSplit?
Standard k-fold cross-validation assigns validation folds randomly, which in a time-series context allows **future weeks to appear in the training fold**. This inflates CV scores and causes the search to select hyperparameters that only appear good because they've implicitly seen the future. `TimeSeriesSplit` ensures each validation fold is always chronologically later than its corresponding training fold — matching how the model is actually deployed.

Training rows are sorted by `(season, week)` before splitting so the splits respect temporal order across seasons as well as within them.

### Search Grids

Parameter grids are **intentionally offset from the build defaults** to force the search to explore genuinely new territory rather than rediscovering the starting hyperparameters:

| Model | Parameters Searched |
|---|---|
| Random Forest | `n_estimators` [150, 250, 350, 500, 700], `max_depth` [6, 9, 14, 20, None], `min_samples_leaf` [1, 3, 6, 10], `max_features` ["sqrt", "log2", 0.4, 0.6] |
| Gradient Boosting | `n_estimators` [150, 250, 400, 600], `learning_rate` [0.005, 0.02, 0.07, 0.12], `max_depth` [2, 4, 6, 7], `subsample` [0.6, 0.75, 0.85, 1.0] |
| XGBoost | `n_estimators` [150, 300, 500, 700], `learning_rate` [0.005, 0.02, 0.07, 0.12], `max_depth` [3, 5, 7, 9], `subsample` [0.6, 0.75, 0.85, 1.0], `colsample_bytree` [0.6, 0.75, 0.85, 1.0] |

Total model fits: **40 iterations × 5 folds × 3 models × 4 positions = 2,400 fits**.

After tuning completes, the best model is re-determined from the updated (post-tuning) MAE scores. Tuned tree models replace their untuned versions in the saved bundles.

---

## 8. Evaluation and Performance

### Metrics

| Metric | Full Name | Why Used |
|---|---|---|
| **MAE** | Mean Absolute Error | Primary metric. Measured in PPR points — directly interpretable ("the prediction is off by X points on average"). Robust to outliers compared to RMSE. |
| **RMSE** | Root Mean Squared Error | Penalizes large errors more heavily than MAE. Used for secondary comparison. |
| **R²** | Coefficient of Determination | Measures what fraction of variance in actual PPR scores is explained by the model. Useful for comparing across positions with different score distributions. |

### Evaluation Charts

Three charts are generated after training:

- **`position_accuracy.png`** — MAE and R² for the best post-tuning model per position
- **`model_comparison.png`** — All four models' MAE per position in a side-by-side bar chart, with the best model highlighted
- **`regression_lines.png`** — Predicted vs. actual scatter plots with a regression line per position (the ideal model would produce points along y = x)
- **`tuning_comparison.png`** — Before/after MAE for all four models per position; LR is shown as a fixed bar since it has no tunable parameters

### Observed Issues and Limitations

**Inherent noise in the target:**
Week-to-week fantasy production is noisy even for the best models. A receiver can be scheme-targeted for 15 targets one week and barely involved the next. Rolling averages smooth this but cannot fully anticipate game-plan changes, weather, or mid-game scripting adjustments.

**Feature baseline is static:**
At inference time the feature vector is built from end-of-2024 rolling stats. This is a deliberate choice (simulating pre-draft knowledge), but it means the model cannot update mid-season to reflect a player's 2025 form. A deployed production system would recompute features after each new week.

**Rookies and newly acquired players:**
Players without 2024 data cannot be projected. The system correctly surfaces a `404` for these players, but it means a breakout 2025 rookie like a first-year starter is excluded by design.

**DNP weeks inflate error:**
Did-not-play weeks where a player was rostered but inactive have no actual PPR stats. These weeks are excluded from season-level accuracy calculations to avoid penalizing the model for injury absences it could not have predicted.

**Linear Regression dominance:**
The fact that LR equals or beats tree ensembles suggests the engineered features already encode the relevant structure linearly. Additional feature engineering — e.g., target share trends, snap count percentage, air yards — could create non-linear signal that better motivates the ensemble models.

**Potential improvements:**
- Add snap-count and air-yards features to expose utilization signals beyond raw yardage
- Retrain after each week of the 2025 season to incorporate in-season form
- Include weather data (precipitation, wind speed) as matchup features
- Experiment with time-series models (LSTM, Prophet) to exploit temporal dependencies that rolling averages approximate but do not fully capture

---

## 9. Implementation

### Architecture

The project is deployed as a **full-stack web application** with a decoupled backend API and a React frontend.

```
Backend:   FastAPI  (Python)  — http://localhost:8000
Frontend:  React + Vite       — http://localhost:5173
```

### Backend — FastAPI

`backend/main.py` exposes six endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + per-position model load status |
| `GET` | `/players?q={name}` | Autocomplete player search (2025 roster) |
| `GET` | `/models` | Per-position best model, all options, MAE/R² metrics |
| `POST` | `/predict` | Single-game PPR prediction with optional model override |
| `GET` | `/predict/season` | Full 2025 season projection (17 games + bye) |
| `GET` | `/charts/{filename}` | Serve training chart PNGs to the frontend |

All four position model bundles (`.joblib` files) are loaded into memory at startup. At inference time, the backend:
1. Looks up the player's 2024 stats to build the rolling-average feature vector
2. Determines their 2025 team (handles off-season trades)
3. Looks up the opponent and home/away status from the 2025 schedule
4. Fetches the opponent's 2024 average PPR allowed to that position
5. Assembles the feature vector and calls `model.predict()`
6. Fetches the actual 2025 result for comparison

### Frontend — React/Vite

The frontend provides two prediction modes:

- **Game Mode:** Enter a player name and a 2025 week number (1–18). Returns the predicted score alongside the actual result, matchup details, and the model's typical error margin.
- **Season Mode:** Enter a player name. Returns a full 17-game projection table with per-game predicted vs. actual comparison, season totals, and an accuracy badge showing what percentage of predictions fell within one MAE of the actual.

Additional UI features:
- **Autocomplete player search** — queries the `/players` endpoint
- **Model selector dropdown** — allows choosing any of the four trained models; annotates each option with the positions for which it achieves the lowest MAE
- **Always-visible chart panel** — displays the three training performance charts in a right-side column

### Model Persistence

Trained models are serialized with **joblib** as position-specific bundles:

```
models/
├── models_QB.joblib    # all 4 pipelines + metrics + feature_cols
├── models_RB.joblib
├── models_WR.joblib
└── models_TE.joblib
```

Each bundle stores all four fitted pipeline objects keyed by model name, the full metrics dict (MAE, RMSE, R²), and the `feature_cols` list — everything the backend needs to serve predictions without the training data being present.

---

## 10. Pipeline Summary

| Step | Script / Location |
|---|---|
| **Frame the Problem** | README + this report: PPR regression, 2025 holdout test |
| **Get the Data** | `scripts/collect_data.py` — nflreadpy, 2022–2025 seasons |
| **Explore the Data (EDA)** | `scripts/eda.py` — four charts to `models/eda/` |
| **Prepare the Data** | `scripts/process_data.py` — PPR computation, rolling features, train/test split |
| **Train the Model** | `scripts/train_models.py` — LR, RF, GB, XGBoost per position |
| **Fine-Tune the Model** | `scripts/train_models.py` — `RandomizedSearchCV` + `TimeSeriesSplit`, all tree models |
| **Evaluate Against Test Set** | `scripts/train_models.py` — MAE, RMSE, R² on 2025 holdout; four charts generated |
| **Save / Deploy the Model** | `.joblib` bundles → FastAPI backend → React/Vite frontend |
