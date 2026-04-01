import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


SEED = 42
EXPERIMENT2_ANALYSIS_N = 8192

# Bundle-relative paths
_BUNDLE_DIR = Path(__file__).resolve().parents[1]
_FIG_DIR = _BUNDLE_DIR / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)


def simulate_intervention(
    var_name: str,
    intervention_value: float,
    n: int = EXPERIMENT2_ANALYSIS_N,
    seed: int = 42,
):
    """
    Simulate do(Variable = intervention_value) by regenerating data with the variable fixed.
    This breaks all incoming causal arrows to the variable.
    """
    rng = np.random.default_rng(seed)

    # Regenerate the same random structure
    U = rng.choice([0, 1], size=n, p=[0.5, 0.5])
    Z_common = rng.normal(0, 1, n)
    X_syn1 = rng.uniform(0, 1, n)
    X_syn2 = rng.uniform(0, 1, n)

    if var_name == "X_causal":
        # do(X_causal = x): break U -> X_causal
        X_causal_do = np.full(n, intervention_value)
        Y_causal_component = X_causal_do * (1 - U)
        Y_conf_component = 1.5 * Z_common
    elif var_name == "X_conf":
        # do(X_conf = x): break Z_common -> X_conf, but Z_common -> Y remains
        X_causal_natural = U.astype(float) + rng.normal(0, 0.1, n)
        Y_causal_component = X_causal_natural * (1 - U)
        Y_conf_component = 1.5 * Z_common
    else:
        raise ValueError(f"Unknown variable: {var_name}")

    Y_syn_component = 2.0 * ((X_syn1 > 0.5) ^ (X_syn2 > 0.5)).astype(float)

    Y_latent = Y_causal_component + Y_conf_component + Y_syn_component + rng.normal(0, 0.5, n)
    prob_Y = 1 / (1 + np.exp(-Y_latent))
    Y = (rng.random(n) < prob_Y).astype(int)
    return float(Y.mean())


def main():
    intervention_values = [-2, -1, 0, 1, 2]

    # Use a large sample so the intervention plot foregrounds the SCM mechanism
    # rather than sampling variation, which is analysed in Experiment 1.
    results_conf = [simulate_intervention("X_conf", v, seed=SEED) for v in intervention_values]
    results_causal = [simulate_intervention("X_causal", v, seed=SEED) for v in intervention_values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: X_conf (confounded, not causal)
    ax = axes[0]
    ax.plot(intervention_values, results_conf, "o-", color="#3498db", linewidth=2, markersize=8, zorder=3)
    ax.axhline(np.mean(results_conf), color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(
        intervention_values,
        [min(results_conf) - 0.01] * len(intervention_values),
        [max(results_conf) + 0.01] * len(intervention_values),
        alpha=0.2,
        color="#3498db",
        zorder=1,
    )
    ax.set_xlabel(r"$\mathrm{do}(X_{\mathrm{conf}} = x)$", fontsize=11)
    ax.set_ylabel(r"$P(Y=1)$", fontsize=11)
    ax.set_title(
        r"$X_{\mathrm{conf}}$ (confounded, high SHAP)" + "\n"
        + f"$\\Delta P(Y=1) = {max(results_conf) - min(results_conf):.3f}$",
        fontsize=10,
    )
    ax.set_ylim(0.3, 0.8)
    ax.set_xlim(-2.5, 2.5)
    ax.grid(True, alpha=0.3)

    # Right: X_causal (causal variable)
    ax = axes[1]
    ax.plot(intervention_values, results_causal, "o-", color="#e74c3c", linewidth=2, markersize=8, zorder=3)
    ax.fill_between(
        intervention_values,
        [min(results_causal)] * len(intervention_values),
        [max(results_causal)] * len(intervention_values),
        alpha=0.2,
        color="#e74c3c",
        zorder=1,
    )
    ax.set_xlabel(r"$\mathrm{do}(X_{\mathrm{causal}} = x)$", fontsize=11)
    ax.set_ylabel(r"$P(Y=1)$", fontsize=11)
    ax.set_title(
        r"$X_{\mathrm{causal}}$ (causal, low SHAP)" + "\n"
        + f"$\\Delta P(Y=1) = {max(results_causal) - min(results_causal):.3f}$",
        fontsize=10,
    )
    ax.set_ylim(0.3, 0.8)
    ax.set_xlim(-2.5, 2.5)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Simulated causal intervention analysis",
        fontsize=12,
        y=0.98,
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(_FIG_DIR / "epistemic_dissociation_intervention.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Regenerated 'epistemic_dissociation_intervention.png'.")


if __name__ == "__main__":
    main()


