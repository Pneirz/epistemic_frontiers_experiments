from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Dataset:
    """A simple container for train/test splits plus feature names."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_spurious_classification(
    *,
    n_train: int,
    n_test: int,
    p_noise: int = 50,
    snr: float = 1.0,
    seed: int,
) -> Dataset:
    """Binary classification with one true signal feature + many pure-noise features.

    Goal: show finite-sample "predictive" improvements that are not stable and are
    not distinguishable from a probe under repeated experiments.
    """
    rng = np.random.default_rng(seed)

    x_signal_tr = rng.normal(size=(n_train, 1))
    x_noise_tr = rng.normal(size=(n_train, p_noise))
    X_train = np.concatenate([x_signal_tr, x_noise_tr], axis=1)

    x_signal_te = rng.normal(size=(n_test, 1))
    x_noise_te = rng.normal(size=(n_test, p_noise))
    X_test = np.concatenate([x_signal_te, x_noise_te], axis=1)

    # True conditional depends only on the first feature.
    logits_tr = snr * x_signal_tr[:, 0]
    p_tr = _sigmoid(logits_tr)
    y_train = (rng.uniform(size=n_train) < p_tr).astype(int)

    logits_te = snr * x_signal_te[:, 0]
    p_te = _sigmoid(logits_te)
    y_test = (rng.uniform(size=n_test) < p_te).astype(int)

    feature_names = ["x_signal"] + [f"x_noise_{k:03d}" for k in range(p_noise)]
    return Dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, feature_names=feature_names)


def make_group_leakage_classification(
    *,
    n_groups: int = 40,
    n_per_group: int = 30,
    test_groups_fraction: float = 0.25,
    split: str = "group_holdout",
    seed: int,
) -> tuple[Dataset, np.ndarray, np.ndarray]:
    """Binary classification with a group-id feature that leaks label *within* groups.

    We generate:
    - a per-group label bias that makes group id highly predictive on random splits
      (train/test share groups),
    - but weak on group-holdout splits (test uses unseen groups).

    Returns (dataset, group_train, group_test).
    """
    rng = np.random.default_rng(seed)

    n_total = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)

    # Per-group logit bias -> label depends strongly on group.
    group_bias = rng.normal(loc=0.0, scale=1.5, size=n_groups)
    # One "content" feature with mild signal shared across groups.
    x_content = rng.normal(size=n_total)
    logits = 0.7 * x_content + group_bias[groups]
    p = _sigmoid(logits)
    y = (rng.uniform(size=n_total) < p).astype(int)

    # Features:
    # - content feature (legit)
    # - group id encoded as integer (a deliberate "leaky" representation)
    X = np.stack([x_content, groups.astype(int)], axis=1)
    feature_names = ["x_content", "x_group_id"]

    if split not in {"group_holdout", "random"}:
        raise ValueError(f"split must be 'group_holdout' or 'random', got: {split!r}")

    if split == "group_holdout":
        # Split by groups (holdout)
        n_test_groups = max(1, int(round(test_groups_fraction * n_groups)))
        perm = rng.permutation(n_groups)
        test_groups = set(perm[:n_test_groups].tolist())
        is_test = np.array([g in test_groups for g in groups], dtype=bool)
    else:
        # Random split by samples (train/test share groups -> leakage looks predictive)
        idx = rng.permutation(n_total)
        n_test = max(1, int(round(test_groups_fraction * n_total)))
        is_test = np.zeros(n_total, dtype=bool)
        is_test[idx[:n_test]] = True

    X_test = X[is_test]
    y_test = y[is_test]
    group_test = groups[is_test]

    X_train = X[~is_test]
    y_train = y[~is_test]
    group_train = groups[~is_test]

    return (
        Dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, feature_names=feature_names),
        group_train,
        group_test,
    )


def make_xor_classification(
    *,
    n_train: int,
    n_test: int,
    seed: int,
) -> Dataset:
    """Binary XOR: y = 1[x1 xor x2], so each variable is marginally uninformative.

    This illustrates why conditional/incremental information matters.
    """
    rng = np.random.default_rng(seed)

    def _sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        x1 = rng.integers(0, 2, size=n)
        x2 = rng.integers(0, 2, size=n)
        y = (x1 ^ x2).astype(int)
        X = np.stack([x1.astype(float), x2.astype(float)], axis=1)
        return X, y

    X_train, y_train = _sample(n_train)
    X_test, y_test = _sample(n_test)
    return Dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, feature_names=["x1", "x2"])


