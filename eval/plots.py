"""Plotting and visualization utilities for model evaluation (Member B - Tasks B2 & B3).

Generates:
  1. Reliability Diagrams (ECE calibration curve).
  2. Risk-Coverage Curves (selective prediction trade-off).
  3. Benchmark metric comparison bar charts for the final presentation.
"""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_reliability_diagram(
    bin_accs: np.ndarray,
    bin_confs: np.ndarray,
    ece: float,
    save_path: Optional[str] = "results/reliability_diagram.png",
    title: str = "Reliability Diagram (ECE)",
) -> None:
    """Plot ECE reliability diagram comparing confidence against empirical accuracy."""
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.bar(
        bin_confs,
        bin_accs,
        width=0.06,
        alpha=0.7,
        color="#2563eb",
        edgecolor="#1d4ed8",
        label=f"Model (ECE = {ece:.4f})",
    )
    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_risk_coverage(
    coverages: np.ndarray,
    error_rates: np.ndarray,
    save_path: Optional[str] = "results/risk_coverage_curve.png",
    title: str = "Risk-Coverage Tradeoff (Abstention)",
) -> None:
    """Plot error rate as a function of data coverage (selective classification)."""
    plt.figure(figsize=(7, 5))
    plt.plot(coverages * 100, error_rates * 100, marker="o", color="#dc2626", linewidth=2)
    plt.xlabel("Coverage (% of samples answered)", fontsize=12)
    plt.ylabel("Risk / Error Rate (%)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_benchmark_comparison(
    results_csv_path: str = "results.csv",
    save_path: Optional[str] = "results/benchmark_comparison.png",
) -> None:
    """Generates comparison bar chart across models for Macro-F1 and Accuracy."""
    if not os.path.exists(results_csv_path):
        print(f"File {results_csv_path} not found.")
        return

    df = pd.read_csv(results_csv_path)
    test_df = df[df["split"] == "test"]
    summary = test_df.groupby(["exp_id", "metric"])["value"].mean().unstack()

    if "macro_f1" in summary.columns and "accuracy" in summary.columns:
        ax = summary[["accuracy", "macro_f1"]].plot(kind="bar", figsize=(10, 5), colormap="viridis")
        plt.title("Model Benchmark on Test Set (Contract #3)", fontsize=14, fontweight="bold")
        plt.ylabel("Score", fontsize=12)
        plt.xlabel("Experiment ID", fontsize=12)
        plt.ylim(0.5, 1.05)
        plt.xticks(rotation=25, ha="right")
        plt.legend(["Accuracy", "Macro-F1"], loc="lower right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
