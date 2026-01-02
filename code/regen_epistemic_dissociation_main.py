"""
Regenerate epistemic_dissociation_main.png with professional styling for Nature.
Only includes SHAP importance plots (A and B), as the table and summary text
are already in the LaTeX document.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import shap
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# Bundle-relative paths
_BUNDLE_DIR = Path(__file__).resolve().parents[1]
_FIG_DIR = _BUNDLE_DIR / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_epistemic_dgp(n: int = 5000, seed: int = 42) -> tuple:
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
    Y_latent = (
        Y_causal_component +
        Y_conf_component +
        Y_syn_component +
        rng.normal(0, 0.5, n)
    )
    prob_Y = 1 / (1 + np.exp(-Y_latent))
    Y = (rng.random(n) < prob_Y).astype(int)
    
    X = np.column_stack([X_causal, X_conf, X_syn1, X_syn2, Z_noise1, Z_noise2, Z_noise3])
    
    feature_names = [
        r'$X_{\mathrm{causal}}$',
        r'$X_{\mathrm{conf}}$',
        r'$X_{\mathrm{syn1}}$',
        r'$X_{\mathrm{syn2}}$',
        r'$Z_{\mathrm{noise1}}$',
        r'$Z_{\mathrm{noise2}}$',
        r'$Z_{\mathrm{noise3}}$',
    ]
    
    feature_info = {
        'names': feature_names,
        'causal': [True, False, False, False, False, False, False],
        'category': [
            'Causal', 'Confounded', 'Synergistic', 'Synergistic',
            'Noise', 'Noise', 'Noise'
        ],
    }
    
    return X, Y, feature_info


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerically stable logit transform."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _predict_proba_class1(model, X: np.ndarray) -> np.ndarray:
    """Return P(Y=1|X) for sklearn-style classifiers."""
    return model.predict_proba(X)[:, 1]


def _compute_mean_abs_shap(
    trained_models: dict,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    output_space: str,
) -> dict:
    """
    Compute mean(|phi|) per feature for each model using the SAME SHAP algorithm.

    output_space:
      - "probability": explains P(Y=1|X)
      - "log_odds": explains logit(P(Y=1|X))
    """
    if output_space not in {"probability", "log_odds"}:
        raise ValueError("output_space must be 'probability' or 'log_odds'.")

    results = {}
    for name, model in trained_models.items():
        if output_space == "probability":
            f = lambda x: _predict_proba_class1(model, x)
        else:
            f = lambda x: _logit(_predict_proba_class1(model, x))

        explainer = shap.Explainer(f, X_background, algorithm="permutation")
        exp = explainer(X_explain)
        vals = exp.values
        if vals.ndim != 2:
            raise ValueError(f"Unexpected SHAP shape for {name}: {vals.shape}")
        results[name] = np.abs(vals).mean(axis=0)
    return results


def _plot_main_figure(
    mean_abs_by_model: dict,
    feature_info: dict,
    title_suffix: str,
    ylabel: str,
    out_path: Path,
) -> None:
    feature_names = feature_info["names"]

    # Color mapping
    category_colors = {
        "Causal": "#c0392b",
        "Confounded": "#2980b9",
        "Synergistic": "#d35400",
        "Noise": "#7f8c8d",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: use Random Forest as the single-model example
    ax = axes[0]
    rf_mean_abs = mean_abs_by_model["Random Forest"]
    sorted_idx = list(np.argsort(rf_mean_abs)[::-1])

    colors = [category_colors[feature_info["category"][i]] for i in sorted_idx]
    ax.barh(
        range(len(feature_names)),
        [rf_mean_abs[i] for i in sorted_idx],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.invert_yaxis()
    ax.set_xlabel(ylabel, fontsize=11)
    ax.set_title(f"(A) SHAP feature importance (Random Forest){title_suffix}", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)

    # Panel B: model comparison
    ax = axes[1]
    model_names = list(mean_abs_by_model.keys())
    x = np.arange(len(feature_names))
    width = 0.2

    short_names = ["causal", "conf", "syn1", "syn2", "noise1", "noise2", "noise3"]
    for i, model_name in enumerate(model_names):
        mean_abs = mean_abs_by_model[model_name]
        ax.bar(x + i * width, mean_abs, width, label=model_name, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([r"$X_{\mathrm{" + n + "}}$" for n in short_names], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(f"(B) SHAP importance across models{title_suffix}", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Category legend (panel A)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=category_colors["Causal"], edgecolor="black", label="Causal"),
        Patch(facecolor=category_colors["Confounded"], edgecolor="black", label="Confounded"),
        Patch(facecolor=category_colors["Synergistic"], edgecolor="black", label="Synergistic"),
        Patch(facecolor=category_colors["Noise"], edgecolor="black", label="Noise"),
    ]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    # Generate data
    X, Y, feature_info = generate_epistemic_dgp(n=5000, seed=SEED)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=SEED, stratify=Y
    )
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=SEED),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED),
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    # Use a shared background and a shared evaluation subset for all models/spaces
    X_background = shap.sample(X_train, 200, random_state=SEED)
    X_explain = X_test[:400]

    # Version B: probability space (bounded, but not the canonical additive scale for logit models)
    mean_abs_prob = _compute_mean_abs_shap(
        trained_models=trained_models,
        X_background=X_background,
        X_explain=X_explain,
        output_space="probability",
    )
    _plot_main_figure(
        mean_abs_by_model=mean_abs_prob,
        feature_info=feature_info,
        title_suffix=" (probability)",
        ylabel=r"Mean $|\phi_j|$ (probability)",
        out_path=_FIG_DIR / "epistemic_dissociation_main.png",
    )

    # Version A: log-odds space (unbounded, typically the natural scale for tree boosting / logistic models)
    mean_abs_logodds = _compute_mean_abs_shap(
        trained_models=trained_models,
        X_background=X_background,
        X_explain=X_explain,
        output_space="log_odds",
    )
    _plot_main_figure(
        mean_abs_by_model=mean_abs_logodds,
        feature_info=feature_info,
        title_suffix=" (log-odds)",
        ylabel=r"Mean $|\phi_j|$ (log-odds)",
        out_path=_FIG_DIR / "epistemic_dissociation_main_logodds.png",
    )

    print("Regenerated:")
    print(f"- {_FIG_DIR / 'epistemic_dissociation_main.png'}")
    print(f"- {_FIG_DIR / 'epistemic_dissociation_main_logodds.png'}")


if __name__ == "__main__":
    main()

