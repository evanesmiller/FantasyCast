"""
Model Training, Fine-Tuning & Evaluation
==========================================
- Trains one model per position (QB, RB, WR, TE)
- Candidates: Linear Regression, Random Forest, Gradient Boosting, XGBoost
- Fine-tunes the best model per position via RandomizedSearchCV (n_iter=40, TimeSeriesSplit 5-fold)
- Saves the tuned model per position as models/best_model_{POS}.joblib
- Generates per-position accuracy, model comparison, and tuning impact charts

Run: python scripts/train_models.py
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

POSITIONS  = ["QB", "RB", "WR", "TE"]
TARGET_COL = "fantasy_points_ppr"

POS_COLORS = {
    "QB": "#4C72B0",
    "RB": "#55A868",
    "WR": "#C44E52",
    "TE": "#8172B3",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    train = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))
    with open(os.path.join(PROC_DIR, "feature_cols.txt")) as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    return train, test, feature_cols


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models():
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("model", GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42,
            )),
        ]),
        "XGBoost": Pipeline([
            ("model", XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Hyperparameter search spaces (Pipeline step prefix: "model__")
# Linear Regression has no meaningful continuous hyperparameters to search.
# ---------------------------------------------------------------------------

# Grids are intentionally offset from the build_models() defaults so the search
# explores genuinely new territory rather than rediscovering the starting point.
PARAM_GRIDS = {
    "Random Forest": {
        "model__n_estimators":     [150, 250, 350, 500, 700],
        "model__max_depth":        [6, 9, 14, 20, None],
        "model__min_samples_leaf": [1, 3, 6, 10],
        "model__max_features":     ["sqrt", "log2", 0.4, 0.6],
    },
    "Gradient Boosting": {
        "model__n_estimators":  [150, 250, 400, 600],
        "model__learning_rate": [0.005, 0.02, 0.07, 0.12],
        "model__max_depth":     [2, 4, 6, 7],
        "model__subsample":     [0.6, 0.75, 0.85, 1.0],
    },
    "XGBoost": {
        "model__n_estimators":     [150, 300, 500, 700],
        "model__learning_rate":    [0.005, 0.02, 0.07, 0.12],
        "model__max_depth":        [3, 5, 7, 9],
        "model__subsample":        [0.6, 0.75, 0.85, 1.0],
        "model__colsample_bytree": [0.6, 0.75, 0.85, 1.0],
    },
}


def fine_tune_model(model, model_name, X_train, y_train, X_test, y_test,
                    season_col=None, week_col=None):
    """
    RandomizedSearchCV with TimeSeriesSplit over the model's param grid.
    Training rows are sorted chronologically before splitting so each
    validation fold is always later in time than its training fold.
    """
    param_grid = PARAM_GRIDS.get(model_name)
    if param_grid is None:
        print(f"  ⚙️  No tuning grid for {model_name} — skipping")
        return model, None

    # Sort by (season, week) so TimeSeriesSplit respects temporal order
    if season_col is not None and week_col is not None:
        order = X_train.assign(_s=season_col, _w=week_col).sort_values(["_s", "_w"]).index
        X_tr = X_train.loc[order]
        y_tr = y_train.loc[order]
    else:
        X_tr, y_tr = X_train, y_train

    tscv = TimeSeriesSplit(n_splits=5)
    print(f"  🔍 Fine-tuning {model_name} (RandomizedSearchCV n_iter=40, TimeSeriesSplit 5-fold)...",
          flush=True)
    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=40,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_tr, y_tr)
    tuned = search.best_estimator_
    cv_mae = -search.best_score_
    print(f"     CV MAE: {cv_mae:.3f}  Best params: {search.best_params_}")
    metrics = evaluate(tuned, X_test, y_test)
    print(f"     Test  MAE: {metrics['MAE']:.3f}  R²: {metrics['R2']:.3f}")
    return tuned, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "MAE":   mean_absolute_error(y_test, preds),
        "RMSE":  root_mean_squared_error(y_test, preds),
        "R2":    r2_score(y_test, preds),
        "preds": preds,
        "y":     y_test.values,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_model_comparison(all_pos_results: dict):
    """2x2 grid — one subplot per position showing MAE of each model candidate."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison by Position — MAE (lower = better)",
                 fontsize=14, fontweight="bold")

    model_names = list(next(iter(all_pos_results.values())).keys())
    bar_colors  = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    for ax, pos in zip(axes.flat, POSITIONS):
        if pos not in all_pos_results:
            ax.set_visible(False)
            continue
        results = all_pos_results[pos]
        maes    = [results[m]["MAE"] for m in model_names]
        y_min   = max(0, min(maes) * 0.88)          # start just below the best bar
        y_max   = max(maes) + (max(maes) - y_min) * 0.18   # headroom for labels
        bars    = ax.bar(model_names, maes, color=bar_colors)
        ax.set_ylim(y_min, y_max)
        ax.set_title(pos, fontsize=13, fontweight="bold", color=POS_COLORS[pos])
        ax.set_ylabel("MAE (pts)")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=12, ha="right", fontsize=9)
        best_idx = int(np.argmin(maes))
        for i, (bar, val) in enumerate(zip(bars, maes)):
            bar.set_edgecolor("black" if i == best_idx else "none")
            bar.set_linewidth(2 if i == best_idx else 0)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (y_max - y_min) * 0.01,
                    f"{val:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Model comparison chart saved to {path}")
    plt.close()


def plot_position_accuracy(best_pos_results: dict):
    """Bar chart of the winning model's MAE and R² for each position."""
    pos_list = [p for p in POSITIONS if p in best_pos_results]
    maes     = [best_pos_results[p]["MAE"]  for p in pos_list]
    r2s      = [best_pos_results[p]["R2"]   for p in pos_list]
    counts   = [best_pos_results[p]["n"]    for p in pos_list]
    names    = [best_pos_results[p]["name"] for p in pos_list]
    colors   = [POS_COLORS[p] for p in pos_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Per-Position Accuracy (best model per position)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    mae_min = max(0, min(maes) * 0.88)
    mae_max = max(maes) + (max(maes) - mae_min) * 0.18
    bars = ax.bar(pos_list, maes, color=colors)
    ax.set_ylim(mae_min, mae_max)
    ax.set_ylabel("MAE (points)")
    ax.set_title("Mean Absolute Error (lower = better)")
    for bar, val, n, name in zip(bars, maes, counts, names):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (mae_max - mae_min) * 0.01,
                f"{val:.2f}\n{name}\nn={n:,}", ha="center", fontsize=8)

    ax = axes[1]
    r2_min = 0
    r2_max = max(r2s) + max(r2s) * 0.18
    bars = ax.bar(pos_list, r2s, color=colors)
    ax.set_ylim(r2_min, r2_max)
    ax.set_ylabel("R²")
    ax.set_title("R² Score (higher = better)")
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (r2_max - r2_min) * 0.01,
                f"{val:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "position_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Position accuracy chart saved to {path}")
    plt.close()


def plot_regression_lines(all_pos_results: dict, best_pos_results: dict):
    """2x2 grid — predicted vs actual scatter with regression line per position."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Predicted vs Actual PPR Points by Position (2025 Test Set)",
                 fontsize=14, fontweight="bold")

    for ax, pos in zip(axes.flat, POSITIONS):
        if pos not in all_pos_results or pos not in best_pos_results:
            ax.set_visible(False)
            continue

        best_name = best_pos_results[pos]["name"]
        res    = all_pos_results[pos][best_name]
        y_pred = np.clip(res["preds"], 0, None)
        y_true = res["y"]
        color  = POS_COLORS[pos]
        mae    = res["MAE"]
        r2     = res["R2"]

        upper = max(float(y_pred.max()), float(y_true.max())) * 1.08
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)

        # Scatter — semi-transparent to show density
        ax.scatter(y_pred, y_true, alpha=0.28, s=14, color=color, linewidths=0, zorder=2)

        # Perfect prediction reference line (y = x)
        ax.plot([0, upper], [0, upper], "--", color="#aaaaaa", linewidth=1.2,
                label="Perfect prediction", zorder=3)

        # Regression line fitted to predicted vs actual
        m, b   = np.polyfit(y_pred, y_true, 1)
        x_line = np.array([0.0, upper])
        ax.plot(x_line, m * x_line + b, "-", color=color, linewidth=2.2,
                label=f"Regression (slope={m:.2f})", zorder=4)

        ax.set_title(f"{pos} — {best_name}", fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Predicted PPR Points", fontsize=10)
        ax.set_ylabel("Actual PPR Points", fontsize=10)
        ax.text(0.05, 0.95, f"MAE = {mae:.2f}  |  R² = {r2:.3f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.85, edgecolor="#cccccc"))
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25, linestyle=":")

    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "regression_lines.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Regression lines chart saved to {path}")
    plt.close()


MODEL_SHORT   = {"Linear Regression": "LR", "Random Forest": "RF",
                  "Gradient Boosting": "GB", "XGBoost": "XGB"}
MODEL_COLORS  = {"Linear Regression": "#aaaaaa", "Random Forest": "#55A868",
                 "Gradient Boosting": "#C44E52", "XGBoost":        "#8172B3"}
TUNABLE       = ["Random Forest", "Gradient Boosting", "XGBoost"]


def plot_tuning_comparison(all_tuning: dict):
    """
    2×2 grid — one subplot per position.
    Each subplot shows all four models: LR as a single bar (no tuning),
    RF/GB/XGB as a before (gray) + after (colored) pair.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "MAE Before vs. After Hyperparameter Tuning — All Models by Position\n"
        "(RandomizedSearchCV, n_iter=40, TimeSeriesSplit 5-fold  |  LR has no tunable parameters)",
        fontsize=12, fontweight="bold",
    )

    model_order = ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost"]
    width = 0.32

    for ax, pos in zip(axes.flat, POSITIONS):
        if pos not in all_tuning:
            ax.set_visible(False)
            continue

        tuning = all_tuning[pos]
        tick_positions = []
        tick_labels    = []
        x_cursor       = 0.0

        for name in model_order:
            entry  = tuning[name]
            color  = MODEL_COLORS[name]
            short  = MODEL_SHORT[name]

            if name == "Linear Regression":
                bar = ax.bar(x_cursor, entry["before"], width * 1.3,
                             color=color, edgecolor="black", linewidth=0.6, alpha=0.85)
                ax.text(x_cursor, entry["before"] + 0.05,
                        f"{entry['before']:.2f}", ha="center", fontsize=8)
                tick_positions.append(x_cursor)
                tick_labels.append(short)
                x_cursor += 1.1
            else:
                before_x = x_cursor
                after_x  = x_cursor + width + 0.04
                center_x = (before_x + after_x) / 2

                ax.bar(before_x, entry["before"], width,
                       color="#aaaaaa", edgecolor="black", linewidth=0.6, alpha=0.75)
                ax.bar(after_x,  entry["after"],  width,
                       color=color, edgecolor="black", linewidth=0.6, alpha=0.9)

                ax.text(before_x, entry["before"] + 0.05,
                        f"{entry['before']:.2f}", ha="center", fontsize=7.5)

                diff  = entry["before"] - entry["after"]
                label = f"{entry['after']:.2f}"
                if abs(diff) >= 0.005:
                    label += f"\n({'−' if diff >= 0 else '+'}{abs(diff):.2f})"
                ax.text(after_x, entry["after"] + 0.05,
                        label, ha="center", fontsize=7.5, linespacing=1.3)

                tick_positions.append(center_x)
                tick_labels.append(short)
                x_cursor += 1.1

        all_vals = [v for e in tuning.values() for v in (e["before"], e["after"])]
        y_min = max(0, min(all_vals) * 0.9)
        y_max = max(all_vals) * 1.18
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=11)
        ax.set_ylabel("MAE (points)")
        ax.set_title(pos, fontsize=13, fontweight="bold", color=POS_COLORS[pos])
        ax.grid(axis="y", alpha=0.2, linestyle=":")

        # Legend (once, bottom-right)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#aaaaaa", label="Before tuning"),
            Patch(color="#4C72B0", label="After tuning (color = model)"),
        ], fontsize=8, loc="upper right")

    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "tuning_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Tuning comparison chart saved to {path}")
    plt.close()


def plot_feature_importance(model, feature_cols: list, pos: str):
    inner = model.named_steps.get("model", model)
    if not hasattr(inner, "feature_importances_"):
        return
    importances = inner.feature_importances_
    indices     = np.argsort(importances)[-20:]
    plt.figure(figsize=(8, 6))
    plt.barh([feature_cols[i] for i in indices], importances[indices],
             color=POS_COLORS.get(pos, "#55A868"))
    plt.xlabel("Importance")
    plt.title(f"Top 20 Feature Importances — {pos} (best model)")
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Feature importance chart saved to {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("🏈 Fantasy Football Predictor — Model Training")
    print("=" * 55)

    train_df, test_df, feature_cols = load_data()
    print(f"\n   Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    print(f"   Features: {len(feature_cols)}")

    all_pos_results  = {}   # pos → {model_name → metrics}
    best_pos_results = {}   # pos → summary dict for charts
    all_tuning       = {}   # pos → {model_name → {before, after}}
    fi_model         = None
    fi_pos           = None

    for pos in POSITIONS:
        print(f"\n{'─' * 55}")
        print(f"  Position: {pos}")
        print(f"{'─' * 55}")

        train_pos = train_df[train_df["position"] == pos].dropna(
            subset=feature_cols + [TARGET_COL])
        test_pos  = test_df[test_df["position"] == pos].dropna(
            subset=feature_cols + [TARGET_COL])

        if len(train_pos) < 50 or len(test_pos) < 10:
            print(f"  ⚠️  Skipping {pos}: insufficient data "
                  f"(train={len(train_pos)}, test={len(test_pos)})")
            continue

        X_train = train_pos[feature_cols]
        y_train = train_pos[TARGET_COL]
        X_test  = test_pos[feature_cols]
        y_test  = test_pos[TARGET_COL]

        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

        models  = build_models()
        results = {}

        for name, model in models.items():
            print(f"  🔧 {name}...", end=" ", flush=True)
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            results[name] = metrics
            print(f"MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  R²={metrics['R2']:.3f}")

        best_name  = min(results, key=lambda m: results[m]["MAE"])
        best_mae   = results[best_name]["MAE"]
        print(f"\n  🏆 Best: {best_name}  (MAE: {best_mae:.3f})")
        if best_name == "Linear Regression":
            print(f"  ℹ️  Linear Regression wins — no hyperparameters to tune. "
                  f"Fine-tuning all tree-based models for comparison.")

        # Fine-tune every tree-based model; LR is recorded as-is.
        print(f"\n  ── Fine-tuning ──")
        pos_tuning = {"Linear Regression": {
            "before": results["Linear Regression"]["MAE"],
            "after":  results["Linear Regression"]["MAE"],
        }}
        for tree_name in TUNABLE:
            tree_mae = results[tree_name]["MAE"]
            tuned_model, tuned_metrics = fine_tune_model(
                models[tree_name], tree_name, X_train, y_train, X_test, y_test,
                season_col=train_pos["season"],
                week_col=train_pos["week"],
            )
            if tuned_metrics is not None:
                improvement = tree_mae - tuned_metrics["MAE"]
                print(f"  {'📈' if improvement >= 0 else '📉'} {tree_name}: "
                      f"{tree_mae:.3f} → {tuned_metrics['MAE']:.3f} "
                      f"({'−' if improvement >= 0 else '+'}{abs(improvement):.3f})")
                models[tree_name]  = tuned_model
                results[tree_name] = tuned_metrics
                pos_tuning[tree_name] = {"before": tree_mae, "after": tuned_metrics["MAE"]}
            else:
                pos_tuning[tree_name] = {"before": tree_mae, "after": tree_mae}
        all_tuning[pos] = pos_tuning

        # Re-determine best after tuning — a tuned tree model may now outperform LR
        best_name = min(results, key=lambda m: results[m]["MAE"])
        best_mae  = results[best_name]["MAE"]
        print(f"\n  🏆 Post-tuning best: {best_name}  (MAE: {best_mae:.3f})")

        model_path = os.path.join(MODELS_DIR, f"best_model_{pos}.joblib")
        joblib.dump({
            "model":        models[best_name],
            "model_name":   best_name,
            "position":     pos,
            "feature_cols": feature_cols,
            "metrics":      {k: v for k, v in results[best_name].items()
                             if k not in ("preds", "y")},
        }, model_path)
        print(f"  ✅ Saved to {model_path}")

        # Save all-models bundle for model-selector feature
        all_models_path = os.path.join(MODELS_DIR, f"models_{pos}.joblib")
        joblib.dump({
            "models":       models,   # all 4 fitted pipelines keyed by name
            "best":         best_name,
            "position":     pos,
            "feature_cols": feature_cols,
            "metrics": {
                name: {k: v for k, v in res.items() if k not in ("preds", "y")}
                for name, res in results.items()
            },
        }, all_models_path)
        print(f"  ✅ All-models bundle saved to {all_models_path}")

        all_pos_results[pos]  = results
        best_pos_results[pos] = {
            "name": best_name,
            "MAE":  results[best_name]["MAE"],
            "R2":   results[best_name]["R2"],
            "n":    len(X_test),
        }

        # Use the first tree-based best model for feature importance chart
        if fi_model is None and hasattr(
                models[best_name].named_steps.get("model", models[best_name]),
                "feature_importances_"):
            fi_model = models[best_name]
            fi_pos   = pos

    # Summary table
    print(f"\n{'='*55}")
    print(f"{'Position':<10} {'Best Model':<22} {'MAE':>7} {'R²':>7}")
    print("-" * 50)
    for pos, info in best_pos_results.items():
        print(f"{pos:<10} {info['name']:<22} {info['MAE']:>7.3f} {info['R2']:>7.3f}")

    # Charts
    print("\n📊 Generating charts...")
    plot_model_comparison(all_pos_results)
    plot_position_accuracy(best_pos_results)
    plot_regression_lines(all_pos_results, best_pos_results)
    plot_tuning_comparison(all_tuning)
    if fi_model:
        plot_feature_importance(fi_model, feature_cols, fi_pos)

    print("\n✅ Training complete! Next: run the backend with uvicorn backend.main:app --reload")


if __name__ == "__main__":
    main()
