"""
Reconstruct Experiment 2 derived artifacts from raw CSV outputs.

Use this script when regen_exp2_importances_and_nullswap_rmse_pairs.py has
already produced the two raw data files but the downstream steps (summary,
rankings, figures) need to be regenerated — for example after a crash,
after changing figure aesthetics, or after a partial run.

Does NOT refit any models.

Required inputs (bundle-relative):
  - data/exp2_nullswap_rmse_pairs_long.csv
  - data/exp2_importance_measures.csv

Outputs:
  - data/exp2_nullswap_rmse_pairs_summary.csv
  - data/exp2_rankings.csv
  - figures/exp2_nullswap_pairs_heatmap.png
  - figures/exp2_nullswap_target_aggregate.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bundle-relative paths
_BUNDLE_DIR = Path(__file__).resolve().parents[1]
_FIG_DIR = _BUNDLE_DIR / "figures"
_DATA_DIR = _BUNDLE_DIR / "data"
_FIG_DIR.mkdir(parents=True, exist_ok=True)
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Canonical feature order for axes
_FEATURE_ORDER = [
    "X_causal",
    "X_conf",
    "X_syn1",
    "X_syn2",
    "Z_noise1",
    "Z_noise2",
    "Z_noise3",
]


# ---------------------------------------------------------------------------
# Summary / aggregation helpers (mirrors regen_exp2_importances_and_nullswap…)
# ---------------------------------------------------------------------------

def summarize_pairs(df_long: pd.DataFrame, *, eps: float = 0.0) -> pd.DataFrame:
    g = df_long.groupby(
        ["context_feature", "target_feature", "context_idx", "target_idx"],
        sort=False,
    )["delta_rmse"]
    df = g.agg(["mean", "std", "median", "min", "max", "count"]).reset_index()
    df = df.rename(columns={"count": "n"})
    df["p_gt_0"] = g.apply(lambda x: float(np.mean(x.to_numpy() > 0.0))).to_numpy()
    df["p_gt_eps"] = g.apply(lambda x: float(np.mean(x.to_numpy() > eps))).to_numpy()
    return df


def aggregate_target_scores(df_pairs_summary: pd.DataFrame) -> pd.DataFrame:
    g = df_pairs_summary.groupby(["target_feature", "target_idx"], sort=False)
    out = g["mean"].agg(["mean", "median", "max", "min"]).reset_index()
    out = out.rename(
        columns={
            "mean": "nullswap_target_mean_over_context",
            "median": "nullswap_target_median_over_context",
            "max": "nullswap_target_max_over_context",
            "min": "nullswap_target_min_over_context",
        }
    )
    return out


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """Rank features with 1 = most important (largest value)."""
    order = np.argsort(-values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    path_long = _DATA_DIR / "exp2_nullswap_rmse_pairs_long.csv"
    path_imp = _DATA_DIR / "exp2_importance_measures.csv"

    if not path_long.exists():
        raise FileNotFoundError(
            f"Missing: {path_long}\n"
            "Run regen_exp2_importances_and_nullswap_rmse_pairs.py first."
        )
    if not path_imp.exists():
        raise FileNotFoundError(
            f"Missing: {path_imp}\n"
            "Run regen_exp2_importances_and_nullswap_rmse_pairs.py first."
        )

    df_long = pd.read_csv(path_long)
    df_imp = pd.read_csv(path_imp)
    print(f"Loaded {len(df_long):,} rows from {path_long.name}")
    print(f"Loaded {len(df_imp)} features from {path_imp.name}")

    # Infer feature order from the data (fall back to _FEATURE_ORDER constant).
    if "feature_idx" in df_imp.columns:
        ordered_features = (
            df_imp.sort_values("feature_idx")["feature"].tolist()
        )
    else:
        ordered_features = _FEATURE_ORDER

    # ------------------------------------------------------------------
    # 1. Pairwise summary
    # ------------------------------------------------------------------
    df_summary = summarize_pairs(df_long, eps=0.0)
    out_summary = _DATA_DIR / "exp2_nullswap_rmse_pairs_summary.csv"
    df_summary.to_csv(out_summary, index=False)
    print(f"Saved {out_summary.name}  ({len(df_summary)} pairs)")

    # ------------------------------------------------------------------
    # 2. Per-target aggregate scores
    # ------------------------------------------------------------------
    df_target = aggregate_target_scores(df_summary)

    # ------------------------------------------------------------------
    # 3. Rankings
    # ------------------------------------------------------------------
    df_rank = df_imp.merge(
        df_target,
        left_on=["feature", "feature_idx"],
        right_on=["target_feature", "target_idx"],
    )
    df_rank = df_rank.drop(columns=["target_feature", "target_idx"])

    df_rank["rank_gini"] = _rank_desc(df_rank["importance_gini"].to_numpy())
    df_rank["rank_permutation"] = _rank_desc(
        df_rank["importance_permutation_rmse"].to_numpy()
    )
    df_rank["rank_shap"] = _rank_desc(df_rank["importance_shap_mean_abs"].to_numpy())
    df_rank["rank_lime"] = _rank_desc(
        df_rank["importance_lime_mean_abscoef"].to_numpy()
    )
    df_rank["rank_nullswap_pairs_rmse"] = _rank_desc(
        df_rank["nullswap_target_mean_over_context"].to_numpy()
    )
    df_rank["rank_boruta"] = df_rank["boruta_rank"].to_numpy()
    df_rank["rank_mean_across_methods"] = (
        df_rank[
            [
                "rank_gini",
                "rank_permutation",
                "rank_shap",
                "rank_lime",
                "rank_nullswap_pairs_rmse",
                "rank_boruta",
            ]
        ]
        .mean(axis=1)
        .to_numpy()
    )
    df_rank = df_rank.sort_values("rank_mean_across_methods", ascending=True)

    out_rank = _DATA_DIR / "exp2_rankings.csv"
    df_rank.to_csv(out_rank, index=False)
    print(f"Saved {out_rank.name}")

    # ------------------------------------------------------------------
    # 4. Figures
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="talk")

    # Heatmap: rows = context (kept), cols = target (nulled)
    mat = (
        df_summary.pivot(
            index="context_feature", columns="target_feature", values="mean"
        )
        .reindex(index=ordered_features, columns=ordered_features)
        .astype(float)
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        mat,
        ax=ax,
        cmap="viridis",
        center=0.0,
        annot=False,
        cbar_kws={"label": "mean ΔRMSE"},
    )
    ax.set_title(
        "Experiment 2: null-swap on pairs (RMSE), nullifying the 2nd feature"
    )
    ax.set_xlabel("target feature (nulled)")
    ax.set_ylabel("context feature (kept)")
    fig.tight_layout()
    out_heatmap = _FIG_DIR / "exp2_nullswap_pairs_heatmap.png"
    fig.savefig(out_heatmap, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_heatmap.name}")

    # Barplot: per-target aggregate score
    df_target_plot = df_target.sort_values(
        "nullswap_target_mean_over_context", ascending=False
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        data=df_target_plot,
        x="target_feature",
        y="nullswap_target_mean_over_context",
        ax=ax,
        color="#4c72b0",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("")
    ax.set_ylabel("mean ΔRMSE over contexts")
    ax.set_title("Experiment 2: target feature effect (pairwise null-swap, RMSE)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out_bar = _FIG_DIR / "exp2_nullswap_target_aggregate.png"
    fig.savefig(out_bar, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_bar.name}")


if __name__ == "__main__":
    main()
