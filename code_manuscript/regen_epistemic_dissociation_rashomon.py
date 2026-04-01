"""
Shows SHAP ranking comparison across models (Rashomon effect).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import shap
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

SEED = 27
EXPERIMENT2_ANALYSIS_N = 8192
np.random.seed(SEED)

# Bundle-relative paths
_BUNDLE_DIR = Path(__file__).resolve().parents[1]
_FIG_DIR = _BUNDLE_DIR / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_DPI = 300


def generate_epistemic_dgp(n: int = EXPERIMENT2_ANALYSIS_N, seed: int = 42) -> tuple:
    """Generate data from the Complete Epistemic Dissociation DGP."""
    rng = np.random.default_rng(seed)
    
    U = rng.choice([0, 1], size=n, p=[0.5, 0.5])
    X_causal = U.astype(float) + rng.normal(0, 0.1, n)
    Y_causal_component = X_causal * (1 - U)
    
    Z_common = rng.normal(0, 1, n)
    X_conf = Z_common + rng.normal(0, 0.3, n)
    Y_conf_component = 1.5 * Z_common
    
    X_syn1 = rng.uniform(0, 1, n)
    X_syn2 = rng.uniform(0, 1, n)
    Y_syn_component = 2.0 * ((X_syn1 > 0.5) ^ (X_syn2 > 0.5)).astype(float)
    
    Z_noise1 = rng.normal(0, 1, n)
    Z_noise2 = rng.normal(0, 1, n)
    Z_noise3 = rng.normal(0, 1, n)
    
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
    
    return X, Y, feature_names


def main():
    # Use a large sample so the ranking comparison reflects representation choices
    # rather than small-sample instability, which is studied in Experiment 1.
    X, Y, feature_names = generate_epistemic_dgp(n=EXPERIMENT2_ANALYSIS_N, seed=SEED)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=SEED, stratify=Y
    )
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=SEED),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED),
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    # Compute SHAP values with the SAME SHAP algorithm across models (probability space)
    X_background = shap.sample(X_train, 200, random_state=SEED)
    X_explain = X_test[:400]
    shap_values_dict = {}

    for name, model in trained_models.items():
        f = lambda x: model.predict_proba(x)[:, 1]
        explainer = shap.Explainer(f, X_background, algorithm="permutation")
        exp = explainer(X_explain)
        vals = exp.values
        if vals.ndim != 2:
            raise ValueError(f"Unexpected SHAP shape for {name}: {vals.shape}")
        shap_values_dict[name] = vals
    
    # Compute rankings
    model_names = list(shap_values_dict.keys())
    rankings = {}
    for name, shap_vals in shap_values_dict.items():
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.flatten()
        ranking = list(np.argsort(mean_abs)[::-1])
        rankings[name] = ranking
    
    # Create ranking dataframe
    ranking_df = pd.DataFrame()
    for name, ranking in rankings.items():
        rank_dict = {feature_names[i]: pos + 1 for pos, i in enumerate(ranking)}
        ranking_df[name] = [rank_dict[f] for f in feature_names]
    ranking_df.index = feature_names
    
    # Create a slightly larger canvas so labels and cell annotations remain crisp
    # after LaTeX rescales the final PNG in the manuscript.
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    
    # Heatmap of rankings
    ranking_matrix = ranking_df.values
    im = ax.imshow(
        ranking_matrix,
        cmap='RdYlGn_r',
        aspect='auto',
        vmin=1,
        vmax=7,
        interpolation='nearest',
        resample=False,
    )
    
    # Labels
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=35, ha='right', rotation_mode='anchor', fontsize=11)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=11)
    
    # Add text annotations
    for i in range(len(feature_names)):
        for j in range(len(model_names)):
            ax.text(
                j,
                i,
                int(ranking_matrix[i, j]),
                ha='center',
                va='center',
                color='black',
                fontsize=11,
                fontweight='semibold',
            )

    # Thin white separators keep the heatmap readable after downscaling.
    ax.set_xticks(np.arange(-0.5, len(model_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(feature_names), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.0)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('SHAP rank (1 = highest importance)', rotation=270, labelpad=18, fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title('SHAP feature ranking across models', fontsize=13)
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Variable', fontsize=11)
    
    fig.savefig(
        _FIG_DIR / 'epistemic_dissociation_rashomon.png',
        dpi=EXPORT_DPI,
        bbox_inches='tight',
        facecolor='white',
        pad_inches=0.04,
    )
    plt.close(fig)
    
    print("Regenerated 'epistemic_dissociation_rashomon.png'.")


if __name__ == "__main__":
    main()

