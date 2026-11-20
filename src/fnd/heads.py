"""Baseline classification heads (Text-only, Image-only, Concat MLP)."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """2-layer MLP head with Dropout and LayerNorm."""

    def __init__(self, in_features: int = 512, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EarlyConcatFusion(nn.Module):
    """Multimodal early concatenation baseline (E3)."""

    def __init__(self, text_dim: int = 512, image_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.head = ClassificationHead(
            in_features=text_dim + image_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([text_emb, image_emb], dim=-1)
        return self.head(fused)
