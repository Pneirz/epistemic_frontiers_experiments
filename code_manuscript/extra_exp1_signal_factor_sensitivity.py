"""
Sensitivity analysis for Experiment 1 under varying signal factors.

This script varies the coefficient that multiplies X_1 in

    Y ~ Bernoulli(sigmoid(alpha * X_1))

with alpha in {0.5, 2.0, 10.0}, and evaluates the same three protocols used
in Experiment 1 at the reference evaluation count B=100:
  - Bootstrap
  - K-Fold CV
  - DGP Realizations

Outputs are saved with the prefix "extra_":
  - data/extra_exp1_signal_factor_raw.csv
  - data/extra_exp1_signal_factor_b100_summary.csv
  - figures/extra_exp1_signal_factor_b100_boxplots.png
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
N_SEEDS = 10
N_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
P_NOISE = 3
N_FOLDS = 10
ITER_COUNTS = [100]
SIGNAL_FACTORS = [0.5, 2.0, 10.0]

_BUNDLE_DIR = Path(__file__).resolve().parents[1]
_FIG_DIR = _BUNDLE_DIR / "figures"
_DATA_DIR = _BUNDLE_DIR / "data"
_FIG_DIR.mkdir(parents=True, exist_ok=True)
_DATA_DIR.mkdir(parents=True, exist_ok=True)

_RAW_OUT = _DATA_DIR / "extra_exp1_signal_factor_raw.csv"
_SUMMARY_OUT = _DATA_DIR / "extra_exp1_signal_factor_b100_summary.csv"
_FIG_OUT = _FIG_DIR / "extra_exp1_signal_factor_b100_boxplots.png"


# =============================================================================
# Helpers
# =============================================================================

def generate_dgp(n: int, signal_factor: float, p_noise: int = P_NOISE, seed: int = 0):
    """Generate the Experiment 1 DGP with a variable signal factor."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    prob_y = 1 / (1 + np.exp(-signal_factor * x1))
    y = (rng.random(n) < prob_y).astype(int)
    z = rng.normal(0, 1, size=(n, p_noise))
    return np.column_stack([x1, z]), y


def compute_mi(x_j: np.ndarray, y: np.ndarray, n_neighbors: int = 5, random_state: int = 0) -> float:
    """Estimate mutual information for one feature."""
    k = min(n_neighbors, len(y) - 1)
    if k < 1:
        return 0.0
    x_j = x_j.reshape(-1, 1)
    mi = mutual_info_classif(
        x_j,
        y,
        discrete_features=False,
        n_neighbors=k,
        random_state=random_state,
        n_jobs=1,
    )
    return float(mi[0])


def delta(x_j: np.ndarray, y: np.ndarray, rng: np.random.Generator, rs: int = 0) -> float:
    """Compute the null-swap contrast for a single variable."""
    mi_orig = compute_mi(x_j, y, random_state=rs)
    x_null = x_j[rng.permutation(len(x_j))]
    mi_null = compute_mi(x_null, y, random_state=rs + 1)
    return mi_orig - mi_null


# =============================================================================
# Protocol runners
# =============================================================================

def run_bootstrap(x: np.ndarray, y: np.ndarray, n_iter: int, dgp_seed: int) -> np.ndarray:
    """Run bootstrap resampling on a fixed dataset."""
    rng = np.random.default_rng(dgp_seed)
    n = len(y)
    deltas = np.zeros((n_iter, 1 + P_NOISE))
    for b in range(n_iter):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        for j in range(1 + P_NOISE):
            deltas[b, j] = delta(xb[:, j], yb, rng, rs=dgp_seed + b)
    return deltas


def run_kfold(x: np.ndarray, y: np.ndarray, n_iter: int, dgp_seed: int) -> np.ndarray:
    """Run repeated K-Fold CV and return per-fold delta values."""
    n_repeats = max(1, n_iter // N_FOLDS)
    rng = np.random.default_rng(dgp_seed)
    deltas = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=dgp_seed + rep)
        for fold_idx, (_, test_idx) in enumerate(kf.split(x)):
            xf, yf = x[test_idx], y[test_idx]
            row = []
            for j in range(1 + P_NOISE):
                row.append(delta(xf[:, j], yf, rng, rs=dgp_seed + rep * 100 + fold_idx))
            deltas.append(row)
    return np.array(deltas[:n_iter])


def run_dgp(n: int, signal_factor: float, n_iter: int, dgp_seed: int) -> np.ndarray:
    """Run independent DGP realizations."""
    deltas = []
    for r in range(n_iter):
        rng = np.random.default_rng(dgp_seed + r * 1000)
        xr, yr = generate_dgp(n, signal_factor=signal_factor, seed=dgp_seed + r * 1000)
        row = []
        for j in range(1 + P_NOISE):
            row.append(delta(xr[:, j], yr, rng, rs=dgp_seed + r))
        deltas.append(row)
    return np.array(deltas)


# =============================================================================
# Main data collection
# =============================================================================

def collect_data(signal_factors: list[float], n_sizes: list[int]) -> pd.DataFrame:
    """Collect raw delta values for all protocol conditions."""
    var_names = ["$X_1$"] + [f"$Z_{j+1}$" for j in range(P_NOISE)]
    protocols = ["Bootstrap", "K-Fold CV", "DGP Realizations"]
    records = []

    total = len(signal_factors) * len(n_sizes) * N_SEEDS * len(ITER_COUNTS)
    pbar = tqdm(total=total, desc="collecting", unit="run")

    for signal_factor in signal_factors:
        for n in n_sizes:
            for s in range(N_SEEDS):
                seed = SEED + s * 1000
                x, y = generate_dgp(n, signal_factor=signal_factor, seed=seed)

                for n_iter in ITER_COUNTS:
                    d_bs = run_bootstrap(x, y, n_iter, seed)
                    d_kf = run_kfold(x, y, n_iter, seed)
                    d_dg = run_dgp(n, signal_factor, n_iter, seed)

                    for proto, d_mat in zip(protocols, [d_bs, d_kf, d_dg]):
                        for it in range(len(d_mat)):
                            for j, vname in enumerate(var_names):
                                records.append(
                                    {
                                        "signal_factor": signal_factor,
                                        "n": n,
                                        "seed": s,
                                        "protocol": proto,
                                        "n_iter": n_iter,
                                        "variable": vname,
                                        "is_info": j == 0,
                                        "delta": d_mat[it, j],
                                    }
                                )
                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def build_b100_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact summary at the reference evaluation count."""
    df = df[df["n_iter"] == 100].copy()
    grouped = (
        df.groupby(["signal_factor", "protocol", "n", "is_info"], as_index=False)["delta"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    grouped["feature_group"] = np.where(grouped["is_info"], "informative", "noise")
    return grouped


# =============================================================================
# Figure
# =============================================================================

def make_figure(df: pd.DataFrame) -> None:
    """Create B=100 boxplots by signal factor and protocol."""
    df = df[df["n_iter"] == 100].copy()

    protocols = ["Bootstrap", "K-Fold CV", "DGP Realizations"]
    factor_labels = {
        0.5: r"$\alpha = 0.5$",
        2.0: r"$\alpha = 2$",
        10.0: r"$\alpha = 10$",
    }
    color_info = "#27ae60"
    color_noise = "#e74c3c"
    y_lim = (-0.25, 0.9)

    x_tick_labels = [f"{n // 1000}K" if n >= 1000 else str(n) for n in N_SIZES]
    x_pos = np.arange(len(N_SIZES))
    offsets = [-0.18, 0.18]
    width = 0.26

    fig, axes = plt.subplots(
        len(SIGNAL_FACTORS),
        len(protocols),
        figsize=(14, 10),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.10, "wspace": 0.06},
    )
    axes = np.atleast_2d(axes)

    for row, signal_factor in enumerate(SIGNAL_FACTORS):
        for col, proto in enumerate(protocols):
            ax = axes[row, col]
            sub = df[(df["signal_factor"] == signal_factor) & (df["protocol"] == proto)]

            med_info = []
            med_noise = []

            for xi, n in enumerate(N_SIZES):
                sub_n = sub[sub["n"] == n]
                info_vals = sub_n[sub_n["is_info"]]["delta"].values
                noise_vals = sub_n[~sub_n["is_info"]]["delta"].values

                for vals, color, xoff in [
                    (info_vals, color_info, offsets[0]),
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

                med_info.append(np.median(info_vals) if len(info_vals) else np.nan)
                med_noise.append(np.median(noise_vals) if len(noise_vals) else np.nan)

            ax.plot(x_pos + offsets[0], med_info, color=color_info, lw=1.4, ls="-", zorder=3)
            ax.plot(x_pos + offsets[1], med_noise, color=color_noise, lw=1.4, ls="--", zorder=3)

            ax.axhline(0, color="#333333", lw=0.9, ls=":", alpha=0.7, zorder=2)
            ax.set_ylim(y_lim)
            ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)

            if row == 0:
                ax.set_title(proto, fontsize=9, pad=6)

            if row == len(SIGNAL_FACTORS) - 1:
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_tick_labels, fontsize=8)
                ax.set_xlabel("Sample size $n$", fontsize=9, labelpad=4)
            else:
                ax.set_xticks(x_pos)
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(
                    factor_labels[signal_factor] + "\n" + r"$\widetilde{\Delta}$",
                    fontsize=8,
                )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_info, edgecolor="black", alpha=0.6, label=r"Informative ($X_1$)"),
        Patch(facecolor=color_noise, edgecolor="black", alpha=0.6, label=r"Noise ($Z_1, Z_2, Z_3$ pooled)"),
        Line2D([0], [0], color=color_info, lw=1.4, ls="-", label="Median - informative"),
        Line2D([0], [0], color=color_noise, lw=1.4, ls="--", label="Median - noise"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.00),
        fontsize=10,
        framealpha=0.9,
    )

    fig.savefig(_FIG_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {_FIG_OUT}")


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signal-factors",
        type=float,
        nargs="*",
        default=SIGNAL_FACTORS,
        help="Subset of signal factors to run.",
    )
    parser.add_argument(
        "--n-sizes",
        type=int,
        nargs="*",
        default=N_SIZES,
        help="Subset of sample sizes to run.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append raw results to existing file instead of overwriting.",
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Skip summary and figure generation for partial runs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the experiment and save raw data, summary, and figure."""
    args = parse_args()
    signal_factors = args.signal_factors
    n_sizes = args.n_sizes

    print("Extra sensitivity analysis - Experiment 1 signal factor")
    print(f"signal factors: {signal_factors}")
    print(f"n            : {n_sizes}")
    print(f"iters        : {ITER_COUNTS}")
    print(f"seeds        : {N_SEEDS}")
    print()

    df = collect_data(signal_factors=signal_factors, n_sizes=n_sizes)

    if args.append and _RAW_OUT.exists():
        existing = pd.read_csv(_RAW_OUT)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(
            subset=["signal_factor", "n", "seed", "protocol", "n_iter", "variable", "is_info", "delta"]
        )

    df.to_csv(_RAW_OUT, index=False)
    print(f"Raw data saved: {_RAW_OUT}")

    if args.skip_figure:
        return

    summary = build_b100_summary(df)
    summary.to_csv(_SUMMARY_OUT, index=False)
    print(f"Summary saved : {_SUMMARY_OUT}")

    make_figure(df)


if __name__ == "__main__":
    main()
