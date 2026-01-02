from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.datasets import Dataset, make_group_leakage_classification, make_spurious_classification, make_xor_classification


@dataclass(frozen=True)
class ProtocolSpec:
    """Represents an experimental protocol Pi: distribution over data/configs."""

    name: str
    trials: int
    seed: int


def iter_spurious_protocol(
    *,
    spec: ProtocolSpec,
    n_train: int,
    n_test: int,
    p_noise: int,
    snr: float,
) -> list[Dataset]:
    out: list[Dataset] = []
    for t in range(spec.trials):
        out.append(
            make_spurious_classification(
                n_train=n_train,
                n_test=n_test,
                p_noise=p_noise,
                snr=snr,
                seed=spec.seed + 10_000 + t,
            )
        )
    return out


def iter_xor_protocol(*, spec: ProtocolSpec, n_train: int, n_test: int) -> list[Dataset]:
    out: list[Dataset] = []
    for t in range(spec.trials):
        out.append(make_xor_classification(n_train=n_train, n_test=n_test, seed=spec.seed + 20_000 + t))
    return out


def iter_group_leakage_protocol(*, spec: ProtocolSpec, n_groups: int, n_per_group: int) -> list[Dataset]:
    raise RuntimeError(
        "Use iter_group_leakage_protocol_random or iter_group_leakage_protocol_holdout "
        "to make the protocol dependence explicit."
    )


def iter_group_leakage_protocol_random(*, spec: ProtocolSpec, n_groups: int, n_per_group: int) -> list[Dataset]:
    out: list[Dataset] = []
    for t in range(spec.trials):
        ds, _, _ = make_group_leakage_classification(
            n_groups=n_groups,
            n_per_group=n_per_group,
            test_groups_fraction=0.25,
            split="random",
            seed=spec.seed + 30_000 + t,
        )
        out.append(ds)
    return out


def iter_group_leakage_protocol_holdout(*, spec: ProtocolSpec, n_groups: int, n_per_group: int) -> list[Dataset]:
    out: list[Dataset] = []
    for t in range(spec.trials):
        ds, _, _ = make_group_leakage_classification(
            n_groups=n_groups,
            n_per_group=n_per_group,
            test_groups_fraction=0.25,
            split="group_holdout",
            seed=spec.seed + 31_000 + t,
        )
        out.append(ds)
    return out


