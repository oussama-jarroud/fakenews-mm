"""API Data Contracts for Fake News Prediction Service (Contract #4).

Fully aligned with Section 14 (Web Application) and Appendix A7 (FastAPI skeleton)
of the course guide.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Payload sent by client to FastAPI /predict endpoint."""
    text: str = Field(..., max_length=2000, description="News headline or post text")
    image_url: Optional[str] = Field(None, description="Optional image URL for multimodal check")


class PredictionResponse(BaseModel):
    """JSON response returned to web UI and evaluation clients.
    
    Includes both the official PDF specification fields (Appendix A7)
    and detailed explainability attributes.
    """
    # PDF Appendix A7 core fields
    verdict: str = Field(..., description="'misleading' | 'consistent' | 'uncertain'")
    p_fake: float = Field(..., description="Calibrated fake probability in [0.0, 1.0]")
    consistency: float = Field(..., description="CLIP cosine similarity between text and image")
    modality_weights: Dict[str, float] = Field(..., description="Attention weights e.g. {'text': 0.71, 'image': 0.29}")
    heatmap_png_b64: Optional[str] = Field(None, description="Base64-encoded cross-modal alignment heatmap PNG")
    top_tokens: List[Tuple[str, float]] = Field(default_factory=list, description="Top influential words with attribution scores")

    # Companion contract attributes
    prediction: str = Field(..., description="'REAL' or 'FAKE'")
    confidence: float = Field(..., description="Calibrated confidence score [0.0, 1.0]")
    label2: int = Field(..., description="0 for real (consistent), 1 for fake (misleading)")
    label6: Optional[str] = Field(None, description="6-way fine-grained class label")
    gate_weight: float = Field(..., description="Multimodal gate weight (0=image, 1=text)")
    cross_modal_similarity: float = Field(..., description="Alias for consistency")
    abstained: bool = Field(False, description="True if model abstained due to ambiguity")
    explanations: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Detailed explanatory feature breakdown")


class DemoExample(BaseModel):
    """Curated demo case for offline demonstration (Page 13 of course guide)."""
    id: str
    title: str
    text: str
    image_url: Optional[str]
    expected_verdict: str
    description: str
