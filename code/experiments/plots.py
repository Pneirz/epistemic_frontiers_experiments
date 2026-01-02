from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def savefig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stability_bars(
    *,
    title: str,
    labels: list[str],
    p_gt_eps: list[float],
    means: list[float],
    out_path: Path,
) -> None:
    setup_style()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, p_gt_eps, color=sns.color_palette("deep", n_colors=len(labels)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"$\Pr_{E\sim\Pi}[\Delta(E)>\varepsilon]$")
    ax.set_title(title)

    # Annotate with mean delta
    for i, (p, m) in enumerate(zip(p_gt_eps, means, strict=True)):
        ax.text(i, min(0.98, p + 0.03), f"mean={m:.3g}", ha="center", va="bottom", fontsize=11)

    savefig(fig, out_path)


def plot_delta_histogram(
    *,
    title: str,
    deltas: dict[str, np.ndarray],
    eps: float,
    out_path: Path,
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    palette = sns.color_palette("deep", n_colors=len(deltas))
    for (name, arr), c in zip(deltas.items(), palette, strict=True):
        sns.kdeplot(arr, ax=ax, label=name, color=c, fill=False)

    ax.axvline(eps, color="black", linestyle="--", linewidth=1, label=r"$\varepsilon$")
    ax.set_title(title)
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel("density")
    ax.legend()
    savefig(fig, out_path)


def plot_stability_curve(
    *,
    title: str,
    n_grid: np.ndarray,
    p_grid: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, p in p_grid.items():
        ax.plot(n_grid, p, marker="o", linewidth=2, label=name)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("n (train)")
    ax.set_ylabel(r"$\Pr_{E\sim\Pi}[\Delta(E)>\varepsilon]$")
    ax.set_title(title)
    ax.legend()
    savefig(fig, out_path)


