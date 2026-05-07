"""
Model Training & Evaluation
============================
- Trains one model per position (QB, RB, WR, TE)
- Candidates: Linear Regression, Random Forest, Gradient Boosting, XGBoost
- Saves the best model per position as models/best_model_{POS}.joblib
- Generates per-position accuracy and model comparison charts

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
        bars    = ax.bar(model_names, maes, color=bar_colors)
        ax.set_title(pos, fontsize=13, fontweight="bold", color=POS_COLORS[pos])
        ax.set_ylabel("MAE (pts)")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=12, ha="right", fontsize=9)
        best_idx = int(np.argmin(maes))
        for i, (bar, val) in enumerate(zip(bars, maes)):
            bar.set_edgecolor("black" if i == best_idx else "none")
            bar.set_linewidth(2 if i == best_idx else 0)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
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
    bars = ax.bar(pos_list, maes, color=colors)
    ax.set_ylabel("MAE (points)")
    ax.set_title("Mean Absolute Error (lower = better)")
    for bar, val, n, name in zip(bars, maes, counts, names):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}\n{name}\nn={n:,}", ha="center", fontsize=8)

    ax = axes[1]
    bars = ax.bar(pos_list, r2s, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("R²")
    ax.set_title("R² Score (higher = better)")
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "position_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Position accuracy chart saved to {path}")
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
        best_model = models[best_name]
        best_mae   = results[best_name]["MAE"]
        print(f"\n  🏆 Best: {best_name}  (MAE: {best_mae:.3f})")

        model_path = os.path.join(MODELS_DIR, f"best_model_{pos}.joblib")
        joblib.dump({
            "model":        best_model,
            "model_name":   best_name,
            "position":     pos,
            "feature_cols": feature_cols,
            "metrics":      {k: v for k, v in results[best_name].items()
                             if k not in ("preds", "y")},
        }, model_path)
        print(f"  ✅ Saved to {model_path}")

        all_pos_results[pos]  = results
        best_pos_results[pos] = {
            "name": best_name,
            "MAE":  results[best_name]["MAE"],
            "R2":   results[best_name]["R2"],
            "n":    len(X_test),
        }

        # Use the first tree-based best model for feature importance chart
        if fi_model is None and hasattr(
                best_model.named_steps.get("model", best_model), "feature_importances_"):
            fi_model = best_model
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
    if fi_model:
        plot_feature_importance(fi_model, feature_cols, fi_pos)

    print("\n✅ Training complete! Next: run the backend with uvicorn backend.main:app --reload")


if __name__ == "__main__":
    main()
