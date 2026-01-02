## Manuscript experiment bundle

This folder groups **the code, figures, and data artifacts** used by the experiments reported in
`sn-article-nature_revised.tex`, so everything can be shared as a single directory.

### Figures referenced by the manuscript (`sn-article-nature_revised.tex`)

- **Experiment 1 (Epistemic dissociation)**
  - `figures/epistemic_dissociation_main.png`
    - Repro: `code/regen_epistemic_dissociation_main.py` (probability-scale SHAP; unified explainer across models)
  - `figures/epistemic_dissociation_main_logodds.png`
    - Repro: `code/regen_epistemic_dissociation_main.py` (log-odds-scale SHAP; unified explainer across models)
  - `figures/epistemic_dissociation_intervention.png`
    - Repro: `code/regen_epistemic_dissociation_intervention.py`
  - `figures/epistemic_dissociation_rashomon.png`
    - Repro: `code/regen_epistemic_dissociation_rashomon.py` (probability-scale SHAP; unified explainer across models)

- **Experiment 2 (Protocol / Bayes-consistency)**
  - `figures/bayes_consistency_comparison.png`
    - Repro: `code/regen_bayes_consistency.py`
    - Source table: `data/bayes_consistency_comparison.csv`
  - `figures/bayes_consistency_scatter.png`
    - Repro: `code/regen_bayes_consistency.py`

### Data included

- **Experiment 1**
  - `data/epistemic_dissociation_results.csv`

- **Swap/null-swap related**
  - `data/swap_protocol_rf_results.csv`

### Code included

- **Figure regeneration scripts**
  - `code/regen_epistemic_dissociation_main.py`
  - `code/regen_epistemic_dissociation_intervention.py`
  - `code/regen_epistemic_dissociation_rashomon.py`
  - `code/regen_bayes_consistency.py`

- **Notebooks**
  - `code/epistemic_dissociation_experiment.ipynb`
  - `code/bayes_consistency_protocols_rev.ipynb`
  - `code/swap_protocol_rf_experiment.ipynb`
  - `code/swap_protocol_mi_experiment_rev.ipynb`

- **Reusable experiment utilities**
  - `code/experiments/` (Python package used by the scripts/notebooks)

### Environment

- `code/requirements.txt` contains the Python dependencies for running the scripts/notebooks.


