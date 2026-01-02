from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import clone
from sklearn.metrics import log_loss

from experiments.datasets import Dataset


@dataclass(frozen=True)
class TrialResult:
    delta_drop_one: float
    delta_swap_probe: float


def make_probe_by_permutation(X_train: np.ndarray, *, col_idx: int, seed: int) -> np.ndarray:
    """Create a 'null copy' of a feature by permuting it within the training sample.

    This is an *experimental* (finite-sample) probe used to compute the swap contrast:
    replace x_j by permuted values while keeping the pipeline fixed.

    IMPORTANT: For a fair swap/probe contrast, the same transformation should be applied
    consistently to the data used by the probe model (train and evaluation). Otherwise,
    you change the data-generating process between training and test in a way that is
    unrelated to "distinguishability from noise" and can inflate the measured delta.
    """
    rng = np.random.default_rng(seed)
    Xp = X_train.copy()
    perm = rng.permutation(X_train.shape[0])
    Xp[:, col_idx] = Xp[perm, col_idx]
    return Xp


def _fit_predict_proba(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    mdl = clone(model)
    mdl.fit(X_train, y_train)
    proba = mdl.predict_proba(X_test)
    return proba


def compute_trial(
    *,
    dataset: Dataset,
    model,
    feature_idx: int,
    seed: int,
    loss_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> TrialResult:
    """Compute drop-one and swap/probe deltas for one trial.

    Delta convention (consistent with the paper):
      Δ = L(model_without_feature) - L(model_with_feature)
    so positive Δ means the feature helps.
    """
    if loss_fn is None:
        loss_fn = lambda y_true, proba: log_loss(y_true, proba, labels=[0, 1])

    Xtr = dataset.X_train
    ytr = dataset.y_train
    Xte = dataset.X_test
    yte = dataset.y_test

    # Full model
    p_full = _fit_predict_proba(model, Xtr, ytr, Xte)
    L_full = loss_fn(yte, p_full)

    # Drop-one model
    keep = [k for k in range(Xtr.shape[1]) if k != feature_idx]
    p_drop = _fit_predict_proba(model, Xtr[:, keep], ytr, Xte[:, keep])
    L_drop = loss_fn(yte, p_drop)

    delta_drop_one = L_drop - L_full

    # Swap/probe model: replace x_j in training with a permuted version
    Xtr_probe = make_probe_by_permutation(Xtr, col_idx=feature_idx, seed=seed + 991)
    Xte_probe = make_probe_by_permutation(Xte, col_idx=feature_idx, seed=seed + 992)
    p_probe = _fit_predict_proba(model, Xtr_probe, ytr, Xte_probe)
    L_probe = loss_fn(yte, p_probe)

    delta_swap_probe = L_probe - L_full
    return TrialResult(delta_drop_one=float(delta_drop_one), delta_swap_probe=float(delta_swap_probe))


def summarize_stability(deltas: np.ndarray, *, eps: float) -> dict[str, float]:
    """Return stability summary for Pr[Δ > eps] plus mean/std."""
    return {
        "mean": float(np.mean(deltas)),
        "std": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
        "p_gt_eps": float(np.mean(deltas > eps)),
        "n": int(deltas.size),
    }


