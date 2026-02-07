"""
Generate figures for Experiment 2: Bayes-Consistency of Experimental Protocols.
Professional styling for Nature.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
from scipy import stats
from pathlib import Path

SEED = 42
np.random.seed(SEED)

# Bundle-relative paths
_BUNDLE_DIR = Path(__file__).resolve().parents[1]
_FIG_DIR = _BUNDLE_DIR / "figures"
_DATA_DIR = _BUNDLE_DIR / "data"
_FIG_DIR.mkdir(parents=True, exist_ok=True)
_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_dgp(n=5000, p_noise=3, seed=0):
    """Generate data from a simple DGP.
    
    X_1: Informative variable (Y depends on it)
    Z_1, ..., Z_k: Noise variables (independent of Y)
    """
    rng = np.random.default_rng(seed)
    
    X_1 = rng.normal(0, 1, n)
    prob_y = 1 / (1 + np.exp(-2 * X_1))
    Y = (rng.random(n) < prob_y).astype(int)
    
    Z = rng.normal(0, 1, size=(n, p_noise))
    X = np.column_stack([X_1, Z])
    feature_names = [r"$X_1$"] + [rf"$Z_{{{i+1}}}$" for i in range(p_noise)]
    
    return X, Y, feature_names


def compute_mi(X_j, y, random_state=0):
    """Estimate MI using sklearn's mutual_info_classif."""
    if X_j.ndim == 1:
        X_j = X_j.reshape(-1, 1)
    mi = mutual_info_classif(X_j, y, discrete_features=False, n_neighbors=5, random_state=random_state)
    return float(mi[0])


def create_null_copy(X, rng):
    """Permute rows to break X-Y association."""
    perm_idx = rng.permutation(len(X))
    return X[perm_idx] if X.ndim > 1 else X[perm_idx]


def run_bootstrap_protocol(X, y, feature_names, n_bootstrap=50, seed=42):
    """Protocol 1: Bootstrap (NOT Bayes-consistent)."""
    rng = np.random.default_rng(seed)
    results = []
    
    for b in range(n_bootstrap):
        idx = rng.integers(0, len(y), size=len(y))
        X_b = X[idx]
        y_b = y[idx]
        
        for j, fname in enumerate(feature_names):
            X_j = X_b[:, j]
            mi_original = compute_mi(X_j, y_b, random_state=seed + b)
            X_j_null = create_null_copy(X_j, rng)
            mi_null = compute_mi(X_j_null, y_b, random_state=seed + b)
            
            results.append({
                "feature": fname,
                "mi_original": mi_original,
                "mi_null": mi_null,
                "delta": mi_original - mi_null,
                "iteration": b,
                "protocol": "Bootstrap",
                "is_informative": j == 0
            })
    
    return pd.DataFrame(results)


def run_kfold_protocol(X, y, feature_names, n_splits=5, n_repeats=10, seed=42):
    """Protocol 2: K-Fold Cross-Validation (Partially Bayes-consistent)."""
    rng = np.random.default_rng(seed)
    results = []
    
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_fold = X[test_idx]
            y_fold = y[test_idx]
            
            for j, fname in enumerate(feature_names):
                X_j = X_fold[:, j]
                mi_original = compute_mi(X_j, y_fold, random_state=seed + rep * 100 + fold_idx)
                X_j_null = create_null_copy(X_j, rng)
                mi_null = compute_mi(X_j_null, y_fold, random_state=seed + rep * 100 + fold_idx)
                
                results.append({
                    "feature": fname,
                    "mi_original": mi_original,
                    "mi_null": mi_null,
                    "delta": mi_original - mi_null,
                    "iteration": rep * n_splits + fold_idx,
                    "protocol": "K-Fold CV",
                    "is_informative": j == 0
                })
    
    return pd.DataFrame(results)


def run_dgp_protocol(n_samples, p_noise, feature_names, n_realizations=50, seed=42):
    """Protocol 3: DGP Realizations (Bayes-consistent)."""
    results = []
    
    for r in range(n_realizations):
        rng = np.random.default_rng(seed + r * 1000)
        X, y, _ = generate_dgp(n=n_samples, p_noise=p_noise, seed=seed + r * 1000)
        
        for j, fname in enumerate(feature_names):
            X_j = X[:, j]
            mi_original = compute_mi(X_j, y, random_state=seed + r)
            X_j_null = create_null_copy(X_j, rng)
            mi_null = compute_mi(X_j_null, y, random_state=seed + r)
            
            results.append({
                "feature": fname,
                "mi_original": mi_original,
                "mi_null": mi_null,
                "delta": mi_original - mi_null,
                "iteration": r,
                "protocol": "DGP Realizations",
                "is_informative": j == 0
            })
    
    return pd.DataFrame(results)


def main():
    print("Generating data and running protocols...")
    
    # Generate fixed dataset
    n_samples = 5000
    p_noise = 3
    X_fixed, Y_fixed, feature_names = generate_dgp(n=n_samples, p_noise=p_noise, seed=SEED)
    
    # Run protocols
    print("  Running Bootstrap protocol...")
    results_bootstrap = run_bootstrap_protocol(X_fixed, Y_fixed, feature_names, n_bootstrap=50, seed=SEED)
    
    print("  Running K-Fold CV protocol...")
    results_kfold = run_kfold_protocol(X_fixed, Y_fixed, feature_names, n_splits=5, n_repeats=10, seed=SEED)
    
    print("  Running DGP Realizations protocol...")
    results_dgp = run_dgp_protocol(n_samples, p_noise, feature_names, n_realizations=50, seed=SEED)
    
    # Combine results
    results_all = pd.concat([results_bootstrap, results_kfold, results_dgp], ignore_index=True)

    # Save a compact table used by the manuscript bundle (optional but convenient)
    results_all.to_csv(_DATA_DIR / "bayes_consistency_protocol_results_long.csv", index=False)
    
    # =========================================================================
    # Figure 1: Protocol Comparison (Boxplot)
    # =========================================================================
    print("  Creating Figure 1: Protocol comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    
    protocols = ["Bootstrap", "K-Fold CV", "DGP Realizations"]
    subtitles = ["Not Bayes-consistent", "Partially consistent", "Bayes-consistent"]
    
    palette = {True: "#27ae60", False: "#e74c3c"}
    
    for ax, protocol, subtitle in zip(axes, protocols, subtitles):
        data = results_all[results_all["protocol"] == protocol]
        
        sns.boxplot(
            data=data,
            x="feature",
            y="delta",
            hue="is_informative",
            ax=ax,
            palette=palette,
            legend=False,
            linewidth=0.8
        )
        
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_title(f"{protocol}\n({subtitle})", fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, axis='y', alpha=0.3)
    
    axes[0].set_ylabel(r"$\widetilde{\Delta} = I(Y; X_j) - I(Y; X_j^{\mathrm{null}})$", fontsize=10)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", edgecolor="black", label=r"Informative ($X_1$)"),
        Patch(facecolor="#e74c3c", edgecolor="black", label=r"Noise ($Z_j$)")
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    
    plt.suptitle("Swap delta by experimental protocol", fontsize=11, y=1.08)
    plt.tight_layout()
    fig.savefig(_FIG_DIR / "bayes_consistency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # =========================================================================
    # Figure 2: MI Original vs MI Null (Scatter)
    # =========================================================================
    print("  Creating Figure 2: MI scatter plot...")
    
    noise_data = results_all[~results_all["is_informative"]]
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    
    global_max = max(noise_data["mi_original"].max(), noise_data["mi_null"].max()) * 1.15
    
    for ax, protocol, subtitle in zip(axes, protocols, subtitles):
        data = noise_data[noise_data["protocol"] == protocol]
        
        ax.scatter(data["mi_original"], data["mi_null"], alpha=0.5, s=25, c="#e74c3c", edgecolor="white", linewidth=0.3)
        ax.plot([0, global_max], [0, global_max], "k--", linewidth=1.2, label=r"$\widetilde{\Delta} = 0$")
        
        ax.set_xlabel(r"$I(Y; Z_j)$", fontsize=10)
        ax.set_xlim(0, global_max)
        ax.set_ylim(0, global_max)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{protocol}\n({subtitle})", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel(r"$I(Y; Z_j^{\mathrm{null}})$", fontsize=10)
    
    plt.suptitle("Noise variables: original vs null MI", fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(_FIG_DIR / "bayes_consistency_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # =========================================================================
    # Print summary statistics
    # =========================================================================
    print("\nSummary Statistics:")
    print("=" * 70)
    
    for protocol in protocols:
        noise_deltas = results_all[(results_all["protocol"] == protocol) & (~results_all["is_informative"])]["delta"]
        info_deltas = results_all[(results_all["protocol"] == protocol) & (results_all["is_informative"])]["delta"]
        
        # t-test for noise
        t_stat, p_val = stats.ttest_1samp(noise_deltas, 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        
        print(f"\n{protocol}:")
        print(f"  Noise:       mean = {noise_deltas.mean():.4f}, t = {t_stat:.2f}, p = {p_val:.4f} {sig}")
        print(f"  Informative: mean = {info_deltas.mean():.4f}")
    
    print("\n" + "=" * 70)
    print("Figures saved:")
    print(f"  - {_FIG_DIR / 'bayes_consistency_comparison.png'}")
    print(f"  - {_FIG_DIR / 'bayes_consistency_scatter.png'}")


if __name__ == "__main__":
    main()

