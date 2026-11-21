"""Inference Detector for Multimodal Fake News Detection.

Loads the pretrained CLIP encoder and the trained GatedConsistencyFusion model
to perform real-time predictions on text and optional image inputs.
"""

import io
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import requests
import torch
import torch.nn.functional as F

from src.fnd.fusion import GatedConsistencyFusion

logger = logging.getLogger("fnd.detector")
logging.basicConfig(level=logging.INFO)


def to_feature_tensor(output: Any) -> torch.Tensor:
    """Extract 2D pooled tensor from HuggingFace output objects across versions."""
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    if isinstance(output, torch.Tensor):
        return output
    raise ValueError(f"Unsupported output container type: {type(output)}")


def is_hf_cached(model_name: str) -> bool:
    """Check if model directory exists in local HuggingFace cache without network."""
    try:
        cache_dir = os.environ.get(
            "HF_HUB_CACHE",
            os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub")),
        )
        folder = "models--" + model_name.replace("/", "--")
        return os.path.isdir(os.path.join(cache_dir, folder))
    except Exception:
        return False


class FakeNewsDetector:
    """Production detector service for consistency-aware multimodal fake news classification."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/best_gated_fusion.pt",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        text_dim: int = 512,
        image_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        mock_encoders: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model_name = clip_model_name
        self.checkpoint_path = checkpoint_path
        self.processor = None
        self.clip = None

        logger.info("Initializing FakeNewsDetector on device: %s", self.device)

        # 1. Determine whether to use real CLIP or mock encoder
        env_mock = os.getenv("MOCK_ENCODERS", "").lower() in ("1", "true")
        cached = is_hf_cached(self.clip_model_name)

        if mock_encoders or env_mock or not cached:
            logger.info("Running FakeNewsDetector with mock feature encoders (cached=%s, mock=%s).", cached, mock_encoders)
            self.mock_encoders = True
        else:
            try:
                from transformers import CLIPModel, CLIPProcessor
                logger.info("Loading cached CLIP encoder: %s", self.clip_model_name)
                self.processor = CLIPProcessor.from_pretrained(self.clip_model_name, local_files_only=True)
                self.clip = CLIPModel.from_pretrained(self.clip_model_name, local_files_only=True).to(self.device)
                self.clip.eval()
                self.mock_encoders = False
            except Exception as e:
                logger.warning("Could not load local CLIP model: %s. Using mock mode.", e)
                self.mock_encoders = True

        # 2. Initialize Gated Consistency Fusion Head
        self.model = GatedConsistencyFusion(
            text_dim=text_dim,
            image_dim=image_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        ).to(self.device)

        # 3. Load Checkpoint if present
        self.is_trained = False
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info("Loading model checkpoint from: %s", checkpoint_path)
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.is_trained = True
            logger.info("Checkpoint loaded successfully!")
        else:
            logger.warning(
                "Checkpoint file '%s' not found! Using initialized weights for development/testing.",
                checkpoint_path,
            )

        self.model.eval()

    def fetch_image(self, image_url: Optional[str]) -> Image.Image:
        """Fetch image from URL, local path, or fallback to neutral placeholder."""
        if not image_url:
            return Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Local file path support
        if os.path.exists(image_url):
            try:
                return Image.open(image_url).convert("RGB")
            except Exception as e:
                logger.warning("Failed to open local image '%s': %s", image_url, e)

        # HTTP(S) URL
        if image_url.startswith(("http://", "https://")):
            try:
                resp = requests.get(image_url, timeout=5, headers={"User-Agent": "FakeNewsDetector/1.0"})
                if resp.status_code == 200:
                    return Image.open(io.BytesIO(resp.content)).convert("RGB")
                logger.warning("HTTP GET %s returned status %d", image_url, resp.status_code)
            except Exception as e:
                logger.warning("Failed to download image from '%s': %s", image_url, e)

        # Fallback neutral image
        return Image.new("RGB", (224, 224), color=(128, 128, 128))

    @torch.no_grad()
    def predict(
        self,
        text: str,
        image_url: Optional[str] = None,
        abstention_margin: float = 0.05,
    ) -> Dict[str, Any]:
        """Runs end-to-end inference on a single multimodal input."""
        # 1. Feature Extraction
        if self.mock_encoders or self.clip is None:
            seed = sum(ord(c) for c in text) % 100000
            g = torch.Generator().manual_seed(seed)
            t_feat = torch.randn(1, 512, generator=g, device=self.device)
            t_feat = F.normalize(t_feat, p=2, dim=-1)

            v_seed = (seed + 42) % 100000 if image_url else (seed + 999) % 100000
            g_v = torch.Generator().manual_seed(v_seed)
            v_feat = torch.randn(1, 512, generator=g_v, device=self.device)
            v_feat = F.normalize(v_feat, p=2, dim=-1)
        else:
            image = self.fetch_image(image_url)
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            text_out = self.clip.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            t_feat = to_feature_tensor(text_out)
            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)

            img_out = self.clip.get_image_features(pixel_values=inputs["pixel_values"])
            v_feat = to_feature_tensor(img_out)
            v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)

        # 2. Gated Fusion Forward Pass
        logits, gate, cos_sim = self.model(t_feat, v_feat)

        # 3. Probabilities & Decisions
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        prob_real = float(probs[0])
        prob_fake = float(probs[1])

        pred_label2 = int(np.argmax(probs))
        confidence = prob_fake if pred_label2 == 1 else prob_real
        prediction_str = "FAKE" if pred_label2 == 1 else "REAL"

        # Abstention check
        abstained = bool(abs(confidence - 0.5) < abstention_margin)

        gate_weight = float(gate.mean().cpu().item())
        cross_modal_similarity = float(cos_sim.squeeze().cpu().item())

        return {
            "prediction": prediction_str,
            "confidence": round(confidence, 4),
            "label2": pred_label2,
            "label6": None,
            "gate_weight": round(gate_weight, 4),
            "cross_modal_similarity": round(cross_modal_similarity, 4),
            "abstained": abstained,
            "explanations": {
                "text_gate_weight": round(gate_weight, 4),
                "image_gate_weight": round(1.0 - gate_weight, 4),
                "cross_modal_alignment": round(cross_modal_similarity, 4),
                "prob_real": round(prob_real, 4),
                "prob_fake": round(prob_fake, 4),
            },
        }
