"""
Recompute null-swap contrasts (including a "one-null-at-a-time" joint baseline)
for the epistemic dissociation DGP used in the paper.

This script exists to make the LaTeX-reported numbers reproducible and to clarify
the exact null construction used for a synergistic pair (X_syn1, X_syn2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score


@dataclass(frozen=True)
class NullSwapSummary:
    mi_original: float
    mi_null_mean: float
    delta: float


def generate_epistemic_dgp(*, n: int = 5000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate data from the Complete Epistemic Dissociation DGP."""
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
    Y = (rng.random(n) < prob_Y).astype(int)

    X = np.column_stack([X_causal, X_conf, X_syn1, X_syn2, Z_noise1, Z_noise2, Z_noise3])
    return X, Y


def null_swap_single_feature(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_idx: int,
    n_permutations: int = 30,
    seed: int = 42,
) -> NullSwapSummary:
    """Compute null-swap delta for a single feature using sklearn's MI estimator."""
    rng = np.random.default_rng(seed)

    x = X[:, feature_idx]
    mi_original = float(mutual_info_classif(x.reshape(-1, 1), y, discrete_features=False, random_state=seed)[0])

    mi_nulls: list[float] = []
    for _ in range(n_permutations):
        x_null = x.copy()
        rng.shuffle(x_null)
        mi_null = float(mutual_info_classif(x_null.reshape(-1, 1), y, discrete_features=False, random_state=seed)[0])
        mi_nulls.append(mi_null)

    mi_null_mean = float(np.mean(mi_nulls))
    return NullSwapSummary(mi_original=mi_original, mi_null_mean=mi_null_mean, delta=mi_original - mi_null_mean)


def _bin2(x: np.ndarray) -> np.ndarray:
    """Discretize into 2 bins with the XOR-relevant threshold at 0.5."""
    return (x > 0.5).astype(int)


def _joint_code_2bit(a01: np.ndarray, b01: np.ndarray) -> np.ndarray:
    """Encode (a,b) in {0,1}^2 into a single integer in {0,1,2,3}."""
    return (a01.astype(int) << 1) | b01.astype(int)


def null_swap_joint_pair(
    X: np.ndarray,
    y: np.ndarray,
    *,
    idx_a: int,
    idx_b: int,
    n_permutations: int = 30,
    seed: int = 42,
    baseline: str,
) -> NullSwapSummary:
    """
    Joint null-swap for a pair (a,b) using a discretized joint MI estimate.

    baseline:
      - "both": permute the JOINT code w.r.t. y (equivalent to permuting rows of (a,b) together)
      - "a_null": permute only a (keep b intact)
      - "b_null": permute only b (keep a intact)
      - "one_null_avg": average of the "a_null" and "b_null" null baselines
    """
    if baseline not in {"both", "a_null", "b_null", "one_null_avg"}:
        raise ValueError(f"Unknown baseline: {baseline!r}")

    rng = np.random.default_rng(seed)

    a = _bin2(X[:, idx_a])
    b = _bin2(X[:, idx_b])
    joint = _joint_code_2bit(a, b)
    mi_original = float(mutual_info_score(joint, y))

    def _estimate_null_mean(kind: str) -> float:
        mi_nulls: list[float] = []
        for _ in range(n_permutations):
            if kind == "both":
                perm = rng.permutation(len(y))
                joint_null = joint[perm]
            elif kind == "a_null":
                a_null = a.copy()
                rng.shuffle(a_null)
                joint_null = _joint_code_2bit(a_null, b)
            elif kind == "b_null":
                b_null = b.copy()
                rng.shuffle(b_null)
                joint_null = _joint_code_2bit(a, b_null)
            else:
                raise ValueError(kind)
            mi_nulls.append(float(mutual_info_score(joint_null, y)))
        return float(np.mean(mi_nulls))

    if baseline == "one_null_avg":
        mi_null_mean = 0.5 * (_estimate_null_mean("a_null") + _estimate_null_mean("b_null"))
    else:
        mi_null_mean = _estimate_null_mean(baseline)

    return NullSwapSummary(mi_original=mi_original, mi_null_mean=mi_null_mean, delta=mi_original - mi_null_mean)


def main() -> None:
    X, y = generate_epistemic_dgp(n=5000, seed=42)
    names = [
        "X_causal",
        "X_conf",
        "X_syn1",
        "X_syn2",
        "Z_noise1",
        "Z_noise2",
        "Z_noise3",
    ]

    # Use ASCII-only output for Windows consoles with legacy encodings.
    print("NULL-SWAP (single-variable):  Delta~ = I(Y;X_j) - E[I(Y;X_j^null)]")
    for j, name in enumerate(names):
        s = null_swap_single_feature(X, y, feature_idx=j, n_permutations=30, seed=42)
        print(f"- {name:<8}  I={s.mi_original:.4f}  I_null={s.mi_null_mean:.4f}  Delta~={s.delta:.4f}")

    print("\nNULL-SWAP (joint pair): (X_syn1, X_syn2)")
    for baseline in ["both", "a_null", "b_null", "one_null_avg"]:
        s = null_swap_joint_pair(X, y, idx_a=2, idx_b=3, n_permutations=30, seed=42, baseline=baseline)
        print(f"- baseline={baseline:<11}  I={s.mi_original:.4f}  I_null={s.mi_null_mean:.4f}  Delta~={s.delta:.4f}")


if __name__ == "__main__":
    main()


