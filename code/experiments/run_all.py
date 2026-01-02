from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.base import clone

from experiments.metrics import compute_trial, make_probe_by_permutation, summarize_stability
from experiments.plots import plot_delta_histogram, plot_stability_bars, plot_stability_curve
from experiments.protocols import (
    ProtocolSpec,
    iter_group_leakage_protocol_holdout,
    iter_group_leakage_protocol_random,
    iter_spurious_protocol,
    iter_xor_protocol,
)


def _logreg_l2() -> Pipeline:
    # Simple, stable baseline classifier for log-loss.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )


def _logreg_with_group_onehot() -> Pipeline:
    """Logistic regression that can exploit group-id leakage via one-hot encoding."""
    pre = ColumnTransformer(
        transformers=[
            ("content", StandardScaler(with_mean=True, with_std=True), [0]),
            ("group", OneHotEncoder(handle_unknown="ignore"), [1]),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("pre", pre), ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))])


def run_spurious(*, out_dir: Path, trials: int, seed: int) -> None:
    model = _logreg_l2()
    spec = ProtocolSpec(name="spurious_iid", trials=trials, seed=seed)

    # We'll show stability vs n (train size).
    n_grid = np.array([80, 150, 300, 600])
    eps = 0.01
    idx_signal = 0
    idx_noise = 1

    p_sig_drop = []
    p_sig_probe = []
    p_noise_drop = []
    p_noise_probe = []

    for n_train in n_grid:
        datasets = iter_spurious_protocol(spec=spec, n_train=int(n_train), n_test=2000, p_noise=80, snr=1.1)
        d_sig_drop = []
        d_sig_probe = []
        d_noise_drop = []
        d_noise_probe = []
        for t, ds in enumerate(datasets):
            r_sig = compute_trial(dataset=ds, model=model, feature_idx=idx_signal, seed=seed + 100 + t)
            r_noise = compute_trial(dataset=ds, model=model, feature_idx=idx_noise, seed=seed + 200 + t)
            d_sig_drop.append(r_sig.delta_drop_one)
            d_sig_probe.append(r_sig.delta_swap_probe)
            d_noise_drop.append(r_noise.delta_drop_one)
            d_noise_probe.append(r_noise.delta_swap_probe)

        p_sig_drop.append(summarize_stability(np.array(d_sig_drop), eps=eps)["p_gt_eps"])
        p_sig_probe.append(summarize_stability(np.array(d_sig_probe), eps=eps)["p_gt_eps"])
        p_noise_drop.append(summarize_stability(np.array(d_noise_drop), eps=eps)["p_gt_eps"])
        p_noise_probe.append(summarize_stability(np.array(d_noise_probe), eps=eps)["p_gt_eps"])

    plot_stability_curve(
        title="Espurio finito-muestra: señal vs ruido (estabilidad vs n)",
        n_grid=n_grid,
        p_grid={
            "señal: drop-one Δ": np.array(p_sig_drop),
            "señal: swap/probe Δ~": np.array(p_sig_probe),
            "ruido: drop-one Δ": np.array(p_noise_drop),
            "ruido: swap/probe Δ~": np.array(p_noise_probe),
        },
        out_path=out_dir / "spurious_signal_vs_noise_stability_curve.png",
    )

    # Histogram at a fixed n
    ds_list = iter_spurious_protocol(spec=spec, n_train=150, n_test=2000, p_noise=80, snr=1.1)
    d_sig_drop = []
    d_sig_probe = []
    d_noise_drop = []
    d_noise_probe = []
    for t, ds in enumerate(ds_list):
        r_sig = compute_trial(dataset=ds, model=model, feature_idx=idx_signal, seed=seed + 300 + t)
        r_noise = compute_trial(dataset=ds, model=model, feature_idx=idx_noise, seed=seed + 400 + t)
        d_sig_drop.append(r_sig.delta_drop_one)
        d_sig_probe.append(r_sig.delta_swap_probe)
        d_noise_drop.append(r_noise.delta_drop_one)
        d_noise_probe.append(r_noise.delta_swap_probe)
    plot_delta_histogram(
        title="Espurio finito-muestra: distribuciones de deltas (n_train=150)",
        deltas={
            "señal: drop-one Δ": np.array(d_sig_drop),
            "señal: swap/probe Δ~": np.array(d_sig_probe),
            "ruido: drop-one Δ": np.array(d_noise_drop),
            "ruido: swap/probe Δ~": np.array(d_noise_probe),
        },
        eps=eps,
        out_path=out_dir / "spurious_signal_vs_noise_delta_hist.png",
    )


def run_leakage(*, out_dir: Path, trials: int, seed: int) -> None:
    model = _logreg_with_group_onehot()
    eps = 0.01

    # In this scenario, "x_group_id" is feature 1.
    feature_idx = 1

    spec_random = ProtocolSpec(name="leakage_random_split", trials=trials, seed=seed)
    spec_holdout = ProtocolSpec(name="leakage_group_holdout", trials=trials, seed=seed)

    def _fit_predict_proba(mdl, Xtr, ytr, Xte):
        m = clone(mdl)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)

    def _loss(mdl, Xtr, ytr, Xte, yte) -> float:
        proba = _fit_predict_proba(mdl, Xtr, ytr, Xte)
        return float(log_loss(yte, proba, labels=[0, 1]))

    # Drop-one model for leakage must have a compatible preprocessing schema (only x_content).
    model_drop_content_only = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )

    def _eval(datasets, seed_offset: int) -> tuple[dict[str, float], dict[str, float]]:
        dd: list[float] = []
        dp: list[float] = []
        for t, ds in enumerate(datasets):
            Xtr = ds.X_train
            ytr = ds.y_train
            Xte = ds.X_test
            yte = ds.y_test

            L_full = _loss(model, Xtr, ytr, Xte, yte)
            L_drop = _loss(model_drop_content_only, Xtr[:, [0]], ytr, Xte[:, [0]], yte)

            Xtr_probe = make_probe_by_permutation(Xtr, col_idx=feature_idx, seed=seed + seed_offset + 10_000 + t)
            Xte_probe = make_probe_by_permutation(Xte, col_idx=feature_idx, seed=seed + seed_offset + 20_000 + t)
            L_probe = _loss(model, Xtr_probe, ytr, Xte_probe, yte)

            dd.append(L_drop - L_full)
            dp.append(L_probe - L_full)

        return summarize_stability(np.array(dd), eps=eps), summarize_stability(np.array(dp), eps=eps)

    s_drop_rand, s_probe_rand = _eval(
        iter_group_leakage_protocol_random(spec=spec_random, n_groups=60, n_per_group=25), seed_offset=300
    )
    s_drop_hold, s_probe_hold = _eval(
        iter_group_leakage_protocol_holdout(spec=spec_holdout, n_groups=60, n_per_group=25), seed_offset=600
    )

    plot_stability_bars(
        title="Leakage/protocolo: la relevancia depende del protocolo \u03A0 (random vs group-holdout)",
        labels=[
            "random split: drop-one Δ",
            "random split: swap/probe \u0394~",
            "group-holdout: drop-one Δ",
            "group-holdout: swap/probe \u0394~",
        ],
        p_gt_eps=[s_drop_rand["p_gt_eps"], s_probe_rand["p_gt_eps"], s_drop_hold["p_gt_eps"], s_probe_hold["p_gt_eps"]],
        means=[s_drop_rand["mean"], s_probe_rand["mean"], s_drop_hold["mean"], s_probe_hold["mean"]],
        out_path=out_dir / "leakage_protocol_comparison.png",
    )


def run_xor(*, out_dir: Path, trials: int, seed: int) -> None:
    model = _logreg_l2()
    spec = ProtocolSpec(name="xor_iid", trials=trials, seed=seed)
    eps = 0.01

    # Logistic regression cannot learn XOR without feature engineering.
    # This demonstrates the "informative but not exploitable predictively" case.
    datasets = iter_xor_protocol(spec=spec, n_train=500, n_test=2000)

    deltas_drop_x1 = []
    deltas_drop_x2 = []
    for t, ds in enumerate(datasets):
        r1 = compute_trial(dataset=ds, model=model, feature_idx=0, seed=seed + 400 + t)
        r2 = compute_trial(dataset=ds, model=model, feature_idx=1, seed=seed + 500 + t)
        deltas_drop_x1.append(r1.delta_drop_one)
        deltas_drop_x2.append(r2.delta_drop_one)

    s1 = summarize_stability(np.array(deltas_drop_x1), eps=eps)
    s2 = summarize_stability(np.array(deltas_drop_x2), eps=eps)

    plot_stability_bars(
        title="XOR: informatividad condicional, pero no explotable por el modelo (logreg lineal)",
        labels=["drop-one Δ (x1)", "drop-one Δ (x2)"],
        p_gt_eps=[s1["p_gt_eps"], s2["p_gt_eps"]],
        means=[s1["mean"], s2["mean"]],
        out_path=out_dir / "xor_not_exploitable_logreg.png",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="figures", help="Output directory for figures")
    ap.add_argument("--trials", type=int, default=200, help="Number of trials per protocol")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_spurious(out_dir=out_dir, trials=args.trials, seed=args.seed)
    run_leakage(out_dir=out_dir, trials=args.trials, seed=args.seed)
    run_xor(out_dir=out_dir, trials=args.trials, seed=args.seed)

    print(f"OK: wrote figures to {out_dir.resolve()}")


if __name__ == "__main__":
    main()


