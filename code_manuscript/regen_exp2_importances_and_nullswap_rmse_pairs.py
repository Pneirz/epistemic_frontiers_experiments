"""
Experiment 2 (Epistemic dissociation): compare feature-importance measures to a
null-swap baseline computed at the level of RMSE.

Requested setup:
  - Importance measures: SHAP, LIME, Gini (RF impurity), permutation importance, Boruta.
  - Do NOT use mutual information.
  - Null-swap computed on RMSE, using 10-fold CV.
  - Evaluate features "in pairs": S = {i, j} (ordered), and nullify the SECOND feature j.
    This matches the paper's subset null-swap definition:
      Δ~_{n,S,j} = L(M^{(S, j<-null)}) - L(M^{(S)})

Outputs (bundle-relative):
  - data/exp2_importance_measures.csv
  - data/exp2_nullswap_rmse_pairs_long.csv
  - data/exp2_nullswap_rmse_pairs_summary.csv
  - data/exp2_rankings.csv
  - figures/exp2_nullswap_pairs_heatmap.png
  - figures/exp2_nullswap_target_aggregate.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from boruta import BorutaPy
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split

SEED = 42
EXPERIMENT2_ANALYSIS_N = 8192

# Bundle-relative paths
_BUNDLE_DIR = Path(__file__).resolve().parents[1]
_FIG_DIR = _BUNDLE_DIR / "figures"
_DATA_DIR = _BUNDLE_DIR / "data"
_FIG_DIR.mkdir(parents=True, exist_ok=True)
_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_epistemic_dgp(
    *,
    n: int = EXPERIMENT2_ANALYSIS_N,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate data from the Complete Epistemic Dissociation DGP (binary Y)."""
    rng = np.random.default_rng(seed)

    # Latent confounder
    U = rng.choice([0, 1], size=n, p=[0.5, 0.5])

    # X_causal: causal but observationally cancelled
    X_causal = U.astype(float) + rng.normal(0, 0.1, n)
    Y_causal_component = X_causal * (1 - U)

    # X_conf: confounded (informative but not causal)
    Z_common = rng.normal(0, 1, n)
    X_conf = Z_common + rng.normal(0, 0.3, n)
    Y_conf_component = 1.5 * Z_common

    # Synergistic variables (XOR pattern)
    X_syn1 = rng.uniform(0, 1, n)
    X_syn2 = rng.uniform(0, 1, n)
    Y_syn_component = 2.0 * ((X_syn1 > 0.5) ^ (X_syn2 > 0.5)).astype(float)

    # Noise variables
    Z_noise1 = rng.normal(0, 1, n)
    Z_noise2 = rng.normal(0, 1, n)
    Z_noise3 = rng.normal(0, 1, n)

    # Combine into Y
    Y_latent = Y_causal_component + Y_conf_component + Y_syn_component + rng.normal(0, 0.5, n)
    prob_Y = 1 / (1 + np.exp(-Y_latent))
    y = (rng.random(n) < prob_Y).astype(int)

    X = np.column_stack([X_causal, X_conf, X_syn1, X_syn2, Z_noise1, Z_noise2, Z_noise3])
    feature_names = [
        "X_causal",
        "X_conf",
        "X_syn1",
        "X_syn2",
        "Z_noise1",
        "Z_noise2",
        "Z_noise3",
    ]
    return X, y, feature_names


def rmse_from_proba(y_true: np.ndarray, p1: np.ndarray) -> float:
    """RMSE between binary labels and predicted probabilities."""
    err = y_true.astype(float) - p1.astype(float)
    return float(np.sqrt(np.mean(err * err)))


def make_probe_by_permutation(X: np.ndarray, *, col_idx: int, seed: int) -> np.ndarray:
    """Create a null copy by permuting one column within the provided sample."""
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    perm = rng.permutation(X.shape[0])
    Xp[:, col_idx] = Xp[perm, col_idx]
    return Xp


def _fit_predict_proba1(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    mdl = clone(model)
    mdl.set_params(n_jobs=1)  # prevent nested parallelism when called inside joblib threads
    mdl.fit(X_train, y_train)
    return mdl.predict_proba(X_test)[:, 1]


@dataclass(frozen=True)
class PairTrial:
    context_feature: str
    target_feature: str
    context_idx: int
    target_idx: int
    repeat: int
    fold: int
    rmse_full: float
    rmse_probe: float
    delta_rmse: float


def _process_fold(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    rep: int,
    fold_idx: int,
    seed: int,
) -> list[dict]:
    """Compute all ordered-pair null-swap trials for one (repeat, fold) slice."""
    p = X.shape[1]
    Xtr_all = X[train_idx]
    ytr = y[train_idx]
    Xte_all = X[test_idx]
    yte = y[test_idx]

    out: list[dict] = []
    for i in range(p):
        for j in range(p):
            if i == j:
                continue

            cols = [i, j]
            Xtr = Xtr_all[:, cols]
            Xte = Xte_all[:, cols]

            p_full = _fit_predict_proba1(model, Xtr, ytr, Xte)
            rmse_full = rmse_from_proba(yte, p_full)

            Xtr_probe = make_probe_by_permutation(Xtr, col_idx=1, seed=seed + 10_000 + rep * 100 + fold_idx)
            Xte_probe = make_probe_by_permutation(Xte, col_idx=1, seed=seed + 20_000 + rep * 100 + fold_idx)
            p_probe = _fit_predict_proba1(model, Xtr_probe, ytr, Xte_probe)
            rmse_probe = rmse_from_proba(yte, p_probe)

            out.append(
                dict(
                    context_feature=feature_names[i],
                    target_feature=feature_names[j],
                    context_idx=i,
                    target_idx=j,
                    repeat=rep,
                    fold=fold_idx,
                    rmse_full=rmse_full,
                    rmse_probe=rmse_probe,
                    delta_rmse=rmse_probe - rmse_full,
                )
            )
    return out


def compute_nullswap_pairs_rmse(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    model,
    n_splits: int = 10,
    n_repeats: int = 20,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Compute ordered-pair null-swap deltas (RMSE) under repeated K-fold CV.

    Each (repeat, fold) slice is processed independently and can be
    parallelised with joblib.  n_jobs=-1 uses all available CPU cores;
    set n_jobs=1 to disable parallelism.
    """
    work: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + 1_000 * rep)
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            work.append((rep, fold_idx, train_idx, test_idx))

    batches = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_fold)(X, y, feature_names, model, tr, te, rep, fi, seed)
        for rep, fi, tr, te in work
    )

    all_rows = [row for batch in batches for row in batch]
    return pd.DataFrame(all_rows)


def summarize_pairs(df_long: pd.DataFrame, *, eps: float = 0.0) -> pd.DataFrame:
    """Summarize pairwise deltas with stability-style stats."""
    g = df_long.groupby(["context_feature", "target_feature", "context_idx", "target_idx"], sort=False)["delta_rmse"]
    df = g.agg(["mean", "std", "median", "min", "max", "count"]).reset_index()
    df = df.rename(columns={"count": "n"})
    df["p_gt_0"] = g.apply(lambda x: float(np.mean(x.to_numpy() > 0.0))).to_numpy()
    df["p_gt_eps"] = g.apply(lambda x: float(np.mean(x.to_numpy() > eps))).to_numpy()
    return df


def aggregate_target_scores(df_pairs_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pairwise null-swap deltas into a per-target score by averaging across contexts."""
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


def compute_importance_measures(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    model: RandomForestClassifier,
    seed: int = 42,
    test_size: float = 0.2,
    perm_repeats: int = 30,
    shap_background_n: int = 200,
    shap_explain_n: int = 600,
    lime_points: int = 80,
    lime_perturbations: int = 800,
    lime_kernel_width: float | None = None,
    boruta_max_iter: int = 60,
) -> pd.DataFrame:
    """Compute global feature-importance scores for multiple methods."""
    rng = np.random.default_rng(seed)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    model.fit(Xtr, ytr)

    # 1) Gini / impurity importance (RF built-in).
    imp_gini = model.feature_importances_.astype(float)

    # 2) Permutation importance: use RMSE on predicted probabilities as the loss.
    def _rmse_scorer(estimator, X_eval, y_eval) -> float:
        p1 = estimator.predict_proba(X_eval)[:, 1]
        return -rmse_from_proba(y_eval, p1)

    perm = permutation_importance(
        model,
        Xte,
        yte,
        scoring=_rmse_scorer,
        n_repeats=perm_repeats,
        random_state=seed,
        n_jobs=1,
    )
    # Higher is "more important": since scoring is negative RMSE, importances_mean is (score_original - score_permuted).
    imp_perm = perm.importances_mean.astype(float)

    # 3) SHAP (tree explainer): mean(|phi|) for class 1.
    bg_idx = rng.choice(Xtr.shape[0], size=min(shap_background_n, Xtr.shape[0]), replace=False)
    ex_idx = rng.choice(Xte.shape[0], size=min(shap_explain_n, Xte.shape[0]), replace=False)
    explainer = shap.TreeExplainer(model, data=Xtr[bg_idx])
    shap_values = explainer.shap_values(Xte[ex_idx])
    if isinstance(shap_values, list):
        # Older SHAP: list of arrays per class
        sv = shap_values[1]
    else:
        sv = shap_values
    # Handle 3D array (n_samples, n_features, n_classes) from newer SHAP versions
    if sv.ndim == 3:
        sv = sv[:, :, 1]  # Take class 1 SHAP values
    imp_shap = np.mean(np.abs(sv), axis=0).astype(float)

    # 4) LIME (simple in-house implementation): average |coef| of local linear surrogates.
    #    This uses the model's p(Y=1|x) as the target.
    x_std = Xtr.std(axis=0, ddof=1)
    x_std = np.where(x_std == 0, 1.0, x_std)
    if lime_kernel_width is None:
        lime_kernel_width = float(np.sqrt(Xtr.shape[1]) * 0.75)

    def _kernel(dist: np.ndarray) -> np.ndarray:
        return np.exp(-(dist * dist) / (lime_kernel_width * lime_kernel_width))

    # Choose reference points to explain.
    ref_idx = rng.choice(Xte.shape[0], size=min(lime_points, Xte.shape[0]), replace=False)
    imp_lime_accum = np.zeros(X.shape[1], dtype=float)

    ridge = Ridge(alpha=1.0, fit_intercept=True, random_state=seed)
    for idx in ref_idx:
        x0 = Xte[idx]
        Z = rng.normal(loc=x0, scale=x_std, size=(lime_perturbations, X.shape[1]))
        pz = model.predict_proba(Z)[:, 1]
        d = np.linalg.norm((Z - x0) / x_std, axis=1)
        w = _kernel(d)
        ridge.fit(Z, pz, sample_weight=w)
        imp_lime_accum += np.abs(ridge.coef_)

    imp_lime = (imp_lime_accum / max(1, len(ref_idx))).astype(float)

    # 5) Boruta: wrapper selection around RF importances.
    boruta_est = RandomForestClassifier(
        n_estimators=800,
        random_state=seed,
        n_jobs=1,
        class_weight="balanced",
        max_depth=None,
    )
    boruta = BorutaPy(
        estimator=boruta_est,
        n_estimators="auto",
        max_iter=boruta_max_iter,
        random_state=seed,
        verbose=0,
    )
    boruta.fit(Xtr, ytr)
    boruta_rank = boruta.ranking_.astype(int)
    boruta_support = boruta.support_.astype(bool)

    df = pd.DataFrame(
        {
            "feature_idx": np.arange(X.shape[1], dtype=int),
            "feature": feature_names,
            "importance_gini": imp_gini,
            "importance_permutation_rmse": imp_perm,
            "importance_shap_mean_abs": imp_shap,
            "importance_lime_mean_abscoef": imp_lime,
            "boruta_rank": boruta_rank,
            "boruta_selected": boruta_support,
        }
    )
    return df


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """Rank with 1=best (largest value)."""
    order = np.argsort(-values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def main() -> None:
    # Use a large sample so Experiment 2 can focus on epistemic dissociation
    # rather than on finite-sample instability, which is already studied in Experiment 1.
    n = EXPERIMENT2_ANALYSIS_N
    X, y, feature_names = generate_epistemic_dgp(n=n, seed=SEED)

    # Base model for comparisons (kept consistent across procedures).
    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    # -------------------------------------------------------------------------
    # A) Importance measures (global)
    # -------------------------------------------------------------------------
    df_imp = compute_importance_measures(
        X,
        y,
        feature_names,
        model=clone(base_model),
        seed=SEED,
        perm_repeats=30,
        shap_background_n=200,
        shap_explain_n=600,
        lime_points=80,
        lime_perturbations=800,
        boruta_max_iter=60,
    )
    out_imp = _DATA_DIR / "exp2_importance_measures.csv"
    df_imp.to_csv(out_imp, index=False)

    # -------------------------------------------------------------------------
    # B) Null-swap RMSE on ordered pairs: S={i,j}, nullify j (the second feature)
    # -------------------------------------------------------------------------
    df_pairs_long = compute_nullswap_pairs_rmse(
        X,
        y,
        feature_names,
        model=clone(base_model),
        n_splits=10,
        n_repeats=20,
        seed=SEED,
        n_jobs=-1,
    )
    out_pairs_long = _DATA_DIR / "exp2_nullswap_rmse_pairs_long.csv"
    df_pairs_long.to_csv(out_pairs_long, index=False)

    df_pairs_summary = summarize_pairs(df_pairs_long, eps=0.0)
    out_pairs_summary = _DATA_DIR / "exp2_nullswap_rmse_pairs_summary.csv"
    df_pairs_summary.to_csv(out_pairs_summary, index=False)

    df_target = aggregate_target_scores(df_pairs_summary)

    # -------------------------------------------------------------------------
    # C) Rankings
    # -------------------------------------------------------------------------
    df_rank = df_imp.merge(df_target, left_on=["feature", "feature_idx"], right_on=["target_feature", "target_idx"])
    df_rank = df_rank.drop(columns=["target_feature", "target_idx"])

    df_rank["rank_gini"] = _rank_desc(df_rank["importance_gini"].to_numpy())
    df_rank["rank_permutation"] = _rank_desc(df_rank["importance_permutation_rmse"].to_numpy())
    df_rank["rank_shap"] = _rank_desc(df_rank["importance_shap_mean_abs"].to_numpy())
    df_rank["rank_lime"] = _rank_desc(df_rank["importance_lime_mean_abscoef"].to_numpy())
    df_rank["rank_nullswap_pairs_rmse"] = _rank_desc(df_rank["nullswap_target_mean_over_context"].to_numpy())
    df_rank["rank_boruta"] = df_rank["boruta_rank"].to_numpy()

    # Simple aggregate rank: average of ranks (lower is better).
    df_rank["rank_mean_across_methods"] = (
        df_rank[["rank_gini", "rank_permutation", "rank_shap", "rank_lime", "rank_nullswap_pairs_rmse", "rank_boruta"]]
        .mean(axis=1)
        .to_numpy()
    )
    df_rank = df_rank.sort_values("rank_mean_across_methods", ascending=True)

    out_rank = _DATA_DIR / "exp2_rankings.csv"
    df_rank.to_csv(out_rank, index=False)

    # -------------------------------------------------------------------------
    # D) Figures
    # -------------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="talk")

    # Heatmap of mean ΔRMSE (rows=context i, cols=target j).
    mat = (
        df_pairs_summary.pivot(index="context_feature", columns="target_feature", values="mean")
        .reindex(index=feature_names, columns=feature_names)
        .astype(float)
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(mat, ax=ax, cmap="viridis", center=0.0, annot=False, cbar_kws={"label": "mean ΔRMSE"})
    ax.set_title("Experiment 2: null-swap on pairs (RMSE), nullifying the 2nd feature")
    ax.set_xlabel("target feature (nulled)")
    ax.set_ylabel("context feature (kept)")
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "exp2_nullswap_pairs_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Barplot of aggregated target scores.
    df_target_plot = df_target.sort_values("nullswap_target_mean_over_context", ascending=False)
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
    fig.savefig(_FIG_DIR / "exp2_nullswap_target_aggregate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(f"  - {out_imp}")
    print(f"  - {out_pairs_long}")
    print(f"  - {out_pairs_summary}")
    print(f"  - {out_rank}")
    print(f"  - {_FIG_DIR / 'exp2_nullswap_pairs_heatmap.png'}")
    print(f"  - {_FIG_DIR / 'exp2_nullswap_target_aggregate.png'}")


if __name__ == "__main__":
    main()


