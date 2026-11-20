"""PyTorch Dataset definitions for multimodal paired data."""

from typing import Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class PrecomputedEmbeddingDataset(Dataset):
    """Loads pre-extracted 512-d CLIP embeddings directly into memory."""

    def __init__(
        self,
        text_embs: np.ndarray,
        image_embs: np.ndarray,
        labels: np.ndarray,
    ):
        self.text_embs = torch.tensor(text_embs, dtype=torch.float32)
        self.image_embs = torch.tensor(image_embs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.text_embs[idx], self.image_embs[idx], self.labels[idx]
