"""
Exploratory Data Analysis
==========================
Generates four charts from the raw and processed datasets:
  1. PPR points distribution by position (box plots)
  2. Top feature correlations with the PPR target
  3. Key raw stats vs PPR scatter by position
  4. Average PPR per week across seasons by position

Outputs are saved to models/eda/.

Run: python scripts/eda.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
EDA_DIR  = os.path.join(os.path.dirname(__file__), "..", "models", "eda")
os.makedirs(EDA_DIR, exist_ok=True)

POSITIONS = ["QB", "RB", "WR", "TE"]
POS_COLORS = {
    "QB": "#4C72B0",
    "RB": "#55A868",
    "WR": "#C44E52",
    "TE": "#8172B3",
}
SEASON_COLORS = {2022: "#4C72B0", 2023: "#55A868", 2024: "#C44E52", 2025: "#8172B3"}

TARGET = "fantasy_points_ppr"


# ---------------------------------------------------------------------------
# Chart 1 — PPR points distribution by position
# ---------------------------------------------------------------------------

def plot_points_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=False)
    fig.suptitle(
        "PPR Fantasy Points Distribution by Position (2022–2025 Regular Season)",
        fontsize=13, fontweight="bold",
    )

    for ax, pos in zip(axes, POSITIONS):
        data_by_season = [
            df[(df["position"] == pos) & (df["season"] == s)][TARGET].dropna().values
            for s in [2022, 2023, 2024, 2025]
        ]
        labels = [str(s) for s in [2022, 2023, 2024, 2025]]
        colors = [SEASON_COLORS[s] for s in [2022, 2023, 2024, 2025]]

        bp = ax.boxplot(
            data_by_season,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color="#888"),
            capprops=dict(color="#888"),
            flierprops=dict(marker="o", markersize=2, alpha=0.3,
                            markeredgecolor="#888", markerfacecolor="#888"),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(pos, fontsize=13, fontweight="bold", color=POS_COLORS[pos])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("Season", fontsize=9)
        ax.set_ylabel("PPR Points" if pos == "QB" else "", fontsize=9)
        ax.grid(axis="y", alpha=0.2, linestyle=":")

        # Annotate median values
        medians = [np.median(d) for d in data_by_season if len(d) > 0]
        for i, med in enumerate(medians):
            ax.text(i + 1, med + 0.5, f"{med:.1f}", ha="center", fontsize=7.5,
                    color="white", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(EDA_DIR, "eda_points_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✅ {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Chart 2 — Top feature correlations with PPR target
# ---------------------------------------------------------------------------

def plot_correlations(df: pd.DataFrame, feature_cols: list):
    corr = df[feature_cols + [TARGET]].corr()[TARGET].drop(TARGET)
    corr = corr.dropna().sort_values(key=abs, ascending=False).head(20)

    colors = ["#55A868" if v >= 0 else "#C44E52" for v in corr.values]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1])
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with PPR Points", fontsize=10)
    ax.set_title(
        "Top 20 Feature Correlations with PPR Target\n(all positions combined)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.2, linestyle=":")

    for bar, val in zip(bars[::-1], corr.values):
        ax.text(
            val + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left" if val >= 0 else "right",
            fontsize=7.5,
        )

    plt.tight_layout()
    path = os.path.join(EDA_DIR, "eda_correlations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✅ {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Chart 3 — Key raw stats vs PPR scatter, one subplot per position
# ---------------------------------------------------------------------------

# The single most predictive raw stat for each position
POS_KEY_STAT = {
    "QB": ("passing_yards",   "Passing Yards"),
    "RB": ("rushing_yards",   "Rushing Yards"),
    "WR": ("receiving_yards", "Receiving Yards"),
    "TE": ("receiving_yards", "Receiving Yards"),
}


def plot_stat_vs_ppr(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Key Raw Stat vs PPR Points by Position",
        fontsize=13, fontweight="bold",
    )

    for ax, pos in zip(axes.flat, POSITIONS):
        stat_col, stat_label = POS_KEY_STAT[pos]
        sub = df[df["position"] == pos][[stat_col, TARGET]].dropna()

        ax.scatter(
            sub[stat_col], sub[TARGET],
            alpha=0.18, s=10, color=POS_COLORS[pos], linewidths=0,
        )

        # Regression line
        m, b = np.polyfit(sub[stat_col], sub[TARGET], 1)
        x_range = np.linspace(sub[stat_col].min(), sub[stat_col].max(), 200)
        ax.plot(x_range, m * x_range + b, color=POS_COLORS[pos],
                linewidth=2, label=f"slope={m:.3f}")

        r = sub[[stat_col, TARGET]].corr().iloc[0, 1]
        ax.set_title(pos, fontsize=12, fontweight="bold", color=POS_COLORS[pos])
        ax.set_xlabel(stat_label, fontsize=9)
        ax.set_ylabel("PPR Points", fontsize=9)
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="#cccccc"))
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2, linestyle=":")

    plt.tight_layout()
    path = os.path.join(EDA_DIR, "eda_stat_vs_ppr.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✅ {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Chart 4 — Average PPR per week across seasons by position
# ---------------------------------------------------------------------------

def plot_weekly_trends(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Average PPR Points per Week by Season and Position",
        fontsize=13, fontweight="bold",
    )

    for ax, pos in zip(axes.flat, POSITIONS):
        sub = df[df["position"] == pos]
        for season, color in SEASON_COLORS.items():
            season_data = (
                sub[sub["season"] == season]
                .groupby("week")[TARGET]
                .mean()
                .reset_index()
                .sort_values("week")
            )
            if season_data.empty:
                continue
            ax.plot(
                season_data["week"], season_data[TARGET],
                marker="o", markersize=4, linewidth=1.8,
                color=color, label=str(season), alpha=0.85,
            )

        ax.set_title(pos, fontsize=12, fontweight="bold", color=POS_COLORS[pos])
        ax.set_xlabel("Week", fontsize=9)
        ax.set_ylabel("Avg PPR Points", fontsize=9)
        ax.legend(fontsize=8, title="Season", title_fontsize=8)
        ax.grid(alpha=0.2, linestyle=":")

    plt.tight_layout()
    path = os.path.join(EDA_DIR, "eda_weekly_trends.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✅ {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("🏈 Fantasy Football Predictor — EDA")
    print("=" * 55)

    print("\n📂 Loading processed data...")
    df = pd.read_csv(os.path.join(PROC_DIR, "processed.csv"))
    with open(os.path.join(PROC_DIR, "feature_cols.txt")) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    print(f"   {len(df):,} rows · {df['position'].value_counts().to_dict()}")
    print(f"   Seasons: {sorted(df['season'].unique().tolist())}")
    print(f"   PPR points — mean: {df[TARGET].mean():.2f}  "
          f"median: {df[TARGET].median():.2f}  max: {df[TARGET].max():.2f}")

    print("\n📊 Generating EDA charts...")
    plot_points_distribution(df)
    plot_correlations(df, feature_cols)
    plot_stat_vs_ppr(df)
    plot_weekly_trends(df)

    print(f"\n✅ EDA complete — charts saved to models/eda/")
    print("   Next: run scripts/train_models.py")


if __name__ == "__main__":
    main()
