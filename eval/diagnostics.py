"""Diagnostics and Shortcut Audit module (Member B - Task B2 / Contribution C1).

Implements:
  1. Swap-the-Image Test: On real multimodal pairs, randomly swap images within the batch.
     A genuine multimodal model will notice the cross-modal mismatch and shift prediction towards fake.
     A shortcut (text-only) model will be completely blind to the swapped image!
  2. Modality Dropout: Zero out image embeddings or text embeddings to measure unimodal drop.
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn


def swap_the_image_test(
    model: nn.Module,
    text_embs: np.ndarray,
    image_embs: np.ndarray,
    labels: np.ndarray,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model sensitivity to visual-textual consistency on genuine pairs."""
    model.eval()
    # Filter to real pairs (label == 0)
    real_mask = (labels == 0)
    te_real = torch.tensor(text_embs[real_mask], dtype=torch.float32).to(device)
    ve_real = torch.tensor(image_embs[real_mask], dtype=torch.float32).to(device)

    # 1. Prediction on intact pairs
    with torch.no_grad():
        logits_orig = model(te_real, ve_real)
        if isinstance(logits_orig, tuple):
            logits_orig = logits_orig[0]
        preds_orig = torch.argmax(logits_orig, dim=-1).cpu().numpy()
        fake_rate_orig = float(np.mean(preds_orig == 1))

    # 2. Prediction on swapped pairs (permutation of image embeddings)
    perm = torch.randperm(ve_real.size(0))
    ve_swapped = ve_real[perm]

    with torch.no_grad():
        logits_swapped = model(te_real, ve_swapped)
        if isinstance(logits_swapped, tuple):
            logits_swapped = logits_swapped[0]
        preds_swapped = torch.argmax(logits_swapped, dim=-1).cpu().numpy()
        fake_rate_swapped = float(np.mean(preds_swapped == 1))

    mismatch_sensitivity = fake_rate_swapped - fake_rate_orig
    return {
        "fake_rate_intact": fake_rate_orig,
        "fake_rate_swapped": fake_rate_swapped,
        "mismatch_sensitivity": mismatch_sensitivity,
    }


def modality_dropout_audit(
    model: nn.Module,
    text_embs: np.ndarray,
    image_embs: np.ndarray,
    labels: np.ndarray,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure performance degradation when zeroing out either modality."""
    model.eval()
    te = torch.tensor(text_embs, dtype=torch.float32).to(device)
    ve = torch.tensor(image_embs, dtype=torch.float32).to(device)

    with torch.no_grad():
        # Full multimodal
        out_both = model(te, ve)
        logits_both = out_both[0] if isinstance(out_both, tuple) else out_both
        acc_both = float(np.mean(torch.argmax(logits_both, dim=-1).cpu().numpy() == labels))

        # Text-only (zeroed vision)
        out_text = model(te, torch.zeros_like(ve))
        logits_text = out_text[0] if isinstance(out_text, tuple) else out_text
        acc_text = float(np.mean(torch.argmax(logits_text, dim=-1).cpu().numpy() == labels))

        # Vision-only (zeroed text)
        out_vision = model(torch.zeros_like(te), ve)
        logits_vision = out_vision[0] if isinstance(out_vision, tuple) else out_vision
        acc_vision = float(np.mean(torch.argmax(logits_vision, dim=-1).cpu().numpy() == labels))

    return {
        "acc_multimodal": acc_both,
        "acc_text_only": acc_text,
        "acc_vision_only": acc_vision,
        "text_reliance_gap": acc_both - acc_text,
    }
