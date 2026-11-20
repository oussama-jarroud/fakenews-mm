"""Consistency-Aware Gated Multimodal Fusion (Contribution C2).

Dynamically gates textual and visual features conditioned on the cross-modal
cosine similarity score between text and image representations.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConsistencyFusion(nn.Module):
    """Gated fusion architecture conditioned on cross-modal cosine similarity."""

    def __init__(
        self,
        text_dim: int = 512,
        image_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Unimodal projections
        self.proj_t = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj_v = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gate network: inputs are text_emb (512) + image_emb (512) + cosine_similarity (1)
        self.gate_net = nn.Sequential(
            nn.Linear(text_dim + image_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Classification classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns:
            logits: [batch_size, num_classes]
            gate_weights: [batch_size, hidden_dim]
            cos_sim: [batch_size, 1]
        """
        # 1. Compute normalized cosine similarity feature
        t_norm = F.normalize(text_emb, p=2, dim=-1)
        v_norm = F.normalize(image_emb, p=2, dim=-1)
        cos_sim = (t_norm * v_norm).sum(dim=-1, keepdim=True)

        # 2. Compute dynamic gate
        gate_input = torch.cat([text_emb, image_emb, cos_sim], dim=-1)
        gate = self.gate_net(gate_input)

        # 3. Project representations
        h_t = self.proj_t(text_emb)
        h_v = self.proj_v(image_emb)

        # 4. Gated fusion
        fused = gate * h_t + (1.0 - gate) * h_v

        # 5. Classify
        logits = self.classifier(fused)
        return logits, gate, cos_sim
