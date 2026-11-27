"""Cross-modal heatmap generation module (Member B - Task B4 / Contribution C3).

Computes and visualizes spatial cross-modal alignment between headline text tokens
and image patches using CLIP representations.
"""

import io
import os
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def compute_token_patch_similarity(
    text_features: np.ndarray,
    image_features: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity matrix between text tokens and image patches.

    Args:
        text_features: [n_tokens, dim]
        image_features: [n_patches, dim]

    Returns:
        similarity_matrix: [n_tokens, n_patches]
    """
    t_norm = text_features / (np.linalg.norm(text_features, axis=-1, keepdims=True) + 1e-8)
    v_norm = image_features / (np.linalg.norm(image_features, axis=-1, keepdims=True) + 1e-8)
    return np.dot(t_norm, v_norm.T)


def plot_cross_modal_heatmap(
    similarity_matrix: np.ndarray,
    tokens: List[str],
    save_path: Optional[str] = "results/cross_modal_heatmap.png",
    title: str = "Cross-Modal Alignment Heatmap (Text-Vision)",
) -> None:
    """Plot cross-modal similarity heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Cosine Similarity")
    plt.yticks(range(len(tokens)), tokens, fontsize=10)
    plt.xlabel("Image Spatial Patches", fontsize=12)
    plt.ylabel("Headline Tokens", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
