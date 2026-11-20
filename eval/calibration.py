"""Calibration and Abstention module (Member B - Task B3 / Contribution C3).

Implements:
  1. Temperature Scaling: post-hoc optimization on validation logits.
  2. Expected Calibration Error (ECE): binned difference between confidence and accuracy.
  3. Reliability Diagram: calibration curve for reporting.
  4. Selective Prediction (Abstention): risk-coverage trade-off curve.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling to calibrate model logits on validation set."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50) -> float:
        """Find optimal temperature T using NLL loss on validation set."""
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self.temperature.item()


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error (ECE) and bin statistics for reliability diagrams."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)

    return float(ece), np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


def risk_coverage_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 50,
) -> Dict[str, np.ndarray]:
    """Compute error rate when answering only the most confident samples (selective classification)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    is_correct = (predictions == labels)

    coverages = []
    error_rates = []

    thresholds = np.linspace(0.5, 1.0, n_thresholds)
    for t in thresholds:
        accepted = confidences >= t
        coverage = np.mean(accepted)
        if coverage > 0:
            error_rate = 1.0 - np.mean(is_correct[accepted])
        else:
            error_rate = 0.0
        coverages.append(coverage)
        error_rates.append(error_rate)

    return {
        "thresholds": thresholds,
        "coverage": np.array(coverages),
        "error_rate": np.array(error_rates),
    }
