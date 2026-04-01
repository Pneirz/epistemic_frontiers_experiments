"""
Build a compact protocol-comparison table for mean noise delta in Experiment 1.

Inputs:
  - data/sensitivity_exp1_raw.csv

Outputs:
  - data/sensitivity_exp1_noise_delta_b100_summary.csv
  - data/sensitivity_exp1_noise_delta_b100_table.tex
"""

from pathlib import Path

import pandas as pd


_BUNDLE_DIR = Path(__file__).resolve().parents[1]
_DATA_DIR = _BUNDLE_DIR / "data"
_INPUT = _DATA_DIR / "sensitivity_exp1_raw.csv"
_OUT_CSV = _DATA_DIR / "sensitivity_exp1_noise_delta_b100_summary.csv"
_OUT_TEX = _DATA_DIR / "sensitivity_exp1_noise_delta_b100_table.tex"


def format_pm(mean: float, std: float) -> str:
    """Format mean +/- sd for manuscript tables."""
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["n_iter"] == 100) & (~df["is_info"])].copy()
    summary = (
        df.groupby(["protocol", "n"], as_index=False)["delta"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary["mean_pm_sd"] = [
        format_pm(mean, std) for mean, std in zip(summary["mean"], summary["std"])
    ]
    return summary


def build_latex_table(summary: pd.DataFrame) -> str:
    table = (
        summary.pivot(index="n", columns="protocol", values="mean_pm_sd")
        .sort_index()
        .reindex(columns=["Bootstrap", "K-Fold CV", "DGP Realizations"])
    )
    table.index = [str(n) for n in table.index]
    return table.to_latex(
        escape=False,
        column_format="lccc",
        caption=(
            "Mean $\\widetilde{\\Delta}$ for noise variables in Experiment 1 at "
            "$B=100$. Entries report mean $\\pm$ s.d."
        ),
        label="tab:exp1_noise_delta",
    )


def main() -> None:
    df = pd.read_csv(_INPUT)
    summary = build_summary(df)
    summary.to_csv(_OUT_CSV, index=False)

    latex_table = build_latex_table(summary)
    _OUT_TEX.write_text(latex_table, encoding="utf-8")

    print(f"Read: {_INPUT}")
    print(f"Saved summary: {_OUT_CSV}")
    print(f"Saved LaTeX table: {_OUT_TEX}")


if __name__ == "__main__":
    main()
