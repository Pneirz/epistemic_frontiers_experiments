"""
Sensitivity analysis for Experiment 1: all three protocols across sample sizes
and iteration counts.

Protocols and iteration counts (symmetric: 10, 50, 100 evaluations each):
  - Bootstrap      : B        in {10, 50, 100} resamples from fixed D_n
  - K-Fold CV      : n_rep x 10 folds, n_rep in {1, 5, 10} -> {10, 50, 100} total
  - DGP realisations: R       in {10, 50, 100} independent draws from P(X,Y)

Sample sizes: n in {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}

Figure: 3 columns (protocols) x 3 rows (iter counts).
Each panel: boxplots of delta_tilde for X1 (green) and Z_pooled (red) per n,
medians connected with lines.

Addresses R1 (FPR vs n curve) and R2 (robustness to iteration count > 50).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

SEED     = 42
N_SEEDS  = 10
N_SIZES  = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
P_NOISE  = 3
N_FOLDS  = 10                        # fixed folds for K-Fold; vary repetitions
ITER_COUNTS = [10, 50, 100]          # total evaluations per protocol

_BUNDLE_DIR = Path(__file__).resolve().parents[1]
_FIG_DIR    = _BUNDLE_DIR / "figures"
_DATA_DIR   = _BUNDLE_DIR / "data"
_FIG_DIR.mkdir(parents=True, exist_ok=True)
_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def generate_dgp(n, p_noise=P_NOISE, seed=0):
    rng    = np.random.default_rng(seed)
    X1     = rng.normal(0, 1, n)
    prob_y = 1 / (1 + np.exp(-2 * X1))
    Y      = (rng.random(n) < prob_y).astype(int)
    Z      = rng.normal(0, 1, size=(n, p_noise))
    return np.column_stack([X1, Z]), Y


def compute_mi(X_j, y, n_neighbors=5, random_state=0):
    k = min(n_neighbors, len(y) - 1)
    if k < 1:
        return 0.0
    X_j = X_j.reshape(-1, 1)
    mi  = mutual_info_classif(
        X_j, y, discrete_features=False, n_neighbors=k, random_state=random_state
    )
    return float(mi[0])


def delta(X_j, y, rng, rs=0):
    mi_orig = compute_mi(X_j, y, random_state=rs)
    X_null  = X_j[rng.permutation(len(X_j))]
    mi_null = compute_mi(X_null, y, random_state=rs + 1)
    return mi_orig - mi_null


# =============================================================================
# Protocol runners — return list of delta_tilde per (iteration, variable)
# Shape: (n_iter, 1+P_NOISE)
# =============================================================================

def run_bootstrap(X, Y, n_iter, dgp_seed):
    """n_iter resamples with replacement from fixed (X, Y)."""
    rng    = np.random.default_rng(dgp_seed)
    n      = len(Y)
    deltas = np.zeros((n_iter, 1 + P_NOISE))
    for b in range(n_iter):
        idx = rng.integers(0, n, size=n)
        Xb, Yb = X[idx], Y[idx]
        for j in range(1 + P_NOISE):
            deltas[b, j] = delta(Xb[:, j], Yb, rng, rs=dgp_seed + b)
    return deltas


def run_kfold(X, Y, n_iter, dgp_seed):
    """
    n_iter = n_repeats * N_FOLDS  (n_repeats = n_iter // N_FOLDS).
    Each iteration = one test fold.
    """
    n_repeats = max(1, n_iter // N_FOLDS)
    rng = np.random.default_rng(dgp_seed)
    deltas = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=dgp_seed + rep)
        for fold_idx, (_, test_idx) in enumerate(kf.split(X)):
            Xf, Yf = X[test_idx], Y[test_idx]
            row = []
            for j in range(1 + P_NOISE):
                row.append(delta(Xf[:, j], Yf, rng, rs=dgp_seed + rep * 100 + fold_idx))
            deltas.append(row)
    return np.array(deltas[:n_iter])


def run_dgp(n, n_iter, dgp_seed):
    """n_iter independent draws from P(X,Y)."""
    deltas = []
    for r in range(n_iter):
        rng    = np.random.default_rng(dgp_seed + r * 1000)
        Xr, Yr = generate_dgp(n, seed=dgp_seed + r * 1000)
        row = []
        for j in range(1 + P_NOISE):
            row.append(delta(Xr[:, j], Yr, rng, rs=dgp_seed + r))
        deltas.append(row)
    return np.array(deltas)


# =============================================================================
# Main data collection
# =============================================================================

def collect_data():
    var_names  = ["$X_1$"] + [f"$Z_{j+1}$" for j in range(P_NOISE)]
    protocols  = ["Bootstrap", "K-Fold CV", "DGP Realizations"]
    records    = []

    total = len(N_SIZES) * N_SEEDS * len(ITER_COUNTS)
    pbar = tqdm(total=total, desc="collecting", unit="run")

    for n in N_SIZES:
        for s in range(N_SEEDS):
            seed = SEED + s * 1000
            X, Y = generate_dgp(n, seed=seed)      # fixed dataset for Bootstrap & KFold

            for n_iter in ITER_COUNTS:
                # Bootstrap
                d_bs = run_bootstrap(X, Y, n_iter, seed)
                # K-Fold
                d_kf = run_kfold(X, Y, n_iter, seed)
                # DGP
                d_dg = run_dgp(n, n_iter, seed)

                for proto, d_mat in zip(protocols, [d_bs, d_kf, d_dg]):
                    for it in range(len(d_mat)):
                        for j, vname in enumerate(var_names):
                            records.append({
                                "n":        n,
                                "seed":     s,
                                "protocol": proto,
                                "n_iter":   n_iter,
                                "variable": vname,
                                "is_info":  j == 0,
                                "delta":    d_mat[it, j],
                            })
                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


# =============================================================================
# Figure
# =============================================================================

def make_figure(df):
    protocols  = ["Bootstrap", "K-Fold CV", "DGP Realizations"]
    subtitles  = ["Not Bayes-consistent", "Partially consistent", "Bayes-consistent"]
    row_labels = {10: "$B = 10$", 50: "$B = 50$", 100: "$B = 100$"}

    color_info  = "#27ae60"
    color_noise = "#e74c3c"

    Y_LIM = (-0.25, 0.70)   # clip K-Fold outliers at small n; shared across all panels

    # x-tick labels: abbreviate thousands
    x_tick_labels = []
    for n in N_SIZES:
        x_tick_labels.append(f"{n//1000}K" if n >= 1000 else str(n))

    n_rows = len(ITER_COUNTS)
    n_cols = len(protocols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 10),
        sharey=True, sharex=True,
        gridspec_kw={"hspace": 0.10, "wspace": 0.06}
    )

    x_pos   = np.arange(len(N_SIZES))
    offsets = [-0.18, 0.18]
    width   = 0.26

    for row, n_iter in enumerate(ITER_COUNTS):
        for col, (proto, subtitle) in enumerate(zip(protocols, subtitles)):
            ax = axes[row, col]

            sub = df[(df["protocol"] == proto) & (df["n_iter"] == n_iter)]

            med_info  = []
            med_noise = []

            for xi, n in enumerate(N_SIZES):
                sub_n = sub[sub["n"] == n]
                info_vals  = sub_n[sub_n["is_info"]]["delta"].values
                noise_vals = sub_n[~sub_n["is_info"]]["delta"].values

                for vals, color, xoff in [
                    (info_vals,  color_info,  offsets[0]),
                    (noise_vals, color_noise, offsets[1]),
                ]:
                    if len(vals) == 0:
                        continue
                    ax.boxplot(
                        vals,
                        positions=[x_pos[xi] + xoff],
                        widths=width,
                        patch_artist=True,
                        manage_ticks=False,
                        showfliers=False,
                        boxprops=dict(facecolor=color, alpha=0.45, linewidth=0.5),
                        medianprops=dict(color="black", linewidth=1.2),
                        whiskerprops=dict(linewidth=0.5, color="#555555"),
                        capprops=dict(linewidth=0.5, color="#555555"),
                    )

                med_info.append(np.median(info_vals)  if len(info_vals)  else np.nan)
                med_noise.append(np.median(noise_vals) if len(noise_vals) else np.nan)

            # Connect medians
            ax.plot(x_pos + offsets[0], med_info,  color=color_info,
                    lw=1.4, ls="-",  zorder=3)
            ax.plot(x_pos + offsets[1], med_noise, color=color_noise,
                    lw=1.4, ls="--", zorder=3)

            ax.axhline(0, color="#333333", lw=0.9, ls=":", alpha=0.7, zorder=2)
            ax.set_ylim(Y_LIM)
            ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)

            # Column titles (top row only)
            if row == 0:
                ax.set_title(f"{proto}\n({subtitle})", fontsize=9, pad=6)

            # X-axis labels (bottom row only)
            if row == n_rows - 1:
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_tick_labels, fontsize=8)
                ax.set_xlabel("Sample size $n$", fontsize=9, labelpad=4)
            else:
                ax.set_xticks(x_pos)
                ax.set_xticklabels([])

            # Left y-label (left column only)
            if col == 0:
                ax.set_ylabel(
                    r"$\widetilde{\Delta} = I(Y;X_j) - I(Y;X_j^{\mathrm{null}})$",
                    fontsize=8
                )

            # Row label on right margin
            if col == n_cols - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(row_labels[n_iter], fontsize=9, rotation=270,
                              labelpad=14, color="#444444")
                ax.yaxis.set_label_position("right")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=color_info,  edgecolor="black", alpha=0.6,
              label=r"Informative ($X_1$)"),
        Patch(facecolor=color_noise, edgecolor="black", alpha=0.6,
              label=r"Noise ($Z_1, Z_2, Z_3$ pooled)"),
        Line2D([0], [0], color=color_info,  lw=1.4, ls="-",  label="Median — informative"),
        Line2D([0], [0], color=color_noise, lw=1.4, ls="--", label="Median — noise"),
    ]
    fig.legend(
        handles=legend_elements, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, 1.00), fontsize=10, framealpha=0.9
    )

    out = _FIG_DIR / "sensitivity_exp1_boxplots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    print("Sensitivity analysis — Experiment 1")
    print(f"n      : {N_SIZES}")
    print(f"Iters  : {ITER_COUNTS}")
    print(f"Seeds  : {N_SEEDS}")
    print()

    df = collect_data()
    df.to_csv(_DATA_DIR / "sensitivity_exp1_raw.csv", index=False)
    print(f"\nData saved: {_DATA_DIR / 'sensitivity_exp1_raw.csv'}")

    make_figure(df)


if __name__ == "__main__":
    main()
