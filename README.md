## Manuscript experiment bundle

This folder groups **the code, figures, and data artifacts** used by the experiments reported in manuscript, so everything can be shared as a single directory.

### Figures referenced by the manuscript 

- **Experiment 1 (Epistemic dissociation)**
  - `figures/epistemic_dissociation_main.png`
    - Repro: `code_manuscript/regen_epistemic_dissociation_main.py` (probability-scale SHAP; unified explainer across models)
  - `figures/epistemic_dissociation_main_logodds.png`
    - Repro: `code_manuscript/regen_epistemic_dissociation_main.py` (log-odds-scale SHAP; unified explainer across models)
  - `figures/epistemic_dissociation_intervention.png`
    - Repro: `code_manuscript/regen_epistemic_dissociation_intervention.py`
  - `figures/epistemic_dissociation_rashomon.png`
    - Repro: `code_manuscript/regen_epistemic_dissociation_rashomon.py` (probability-scale SHAP; unified explainer across models)

- **Experiment 2 (Bayes-consistency of protocols)**
  - `figures/bayes_consistency_comparison.png`
    - Repro: `code_manuscript/regen_bayes_consistency.py`
  - `figures/bayes_consistency_scatter.png`
    - Repro: `code_manuscript/regen_bayes_consistency.py`

- **Experiment 2 (Null-swap RMSE pairs)**
  - `figures/exp2_nullswap_pairs_heatmap.png`
    - Repro: `code_manuscript/regen_exp2_importances_and_nullswap_rmse_pairs.py`
  - `figures/exp2_nullswap_target_aggregate.png`
    - Repro: `code_manuscript/regen_exp2_importances_and_nullswap_rmse_pairs.py`

### Data included

- **Experiment 2 (Bayes-consistency)**
  - `data/bayes_consistency_protocol_results_long.csv`

- **Experiment 2 (Null-swap / importance measures)**
  - `data/exp2_importance_measures.csv`
  - `data/exp2_nullswap_rmse_pairs_long.csv`
  - `data/exp2_nullswap_rmse_pairs_summary.csv`
  - `data/exp2_rankings.csv`

### Code included

- **Figure regeneration scripts**
  - `code_manuscript/regen_epistemic_dissociation_main.py`
  - `code_manuscript/regen_epistemic_dissociation_intervention.py`
  - `code_manuscript/regen_epistemic_dissociation_rashomon.py`
  - `code_manuscript/regen_epistemic_dissociation_null_swap.py`
  - `code_manuscript/regen_bayes_consistency.py`
  - `code_manuscript/regen_exp2_importances_and_nullswap_rmse_pairs.py`

### Environment

- `code_manuscript/requirements.txt` contains the Python dependencies for running the scripts.


