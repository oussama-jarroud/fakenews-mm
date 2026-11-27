"""API Data Contracts for Fake News Prediction Service (Contract #4)."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Payload sent by React UI to FastAPI /predict endpoint."""
    text: str = Field(..., description="News headline or post text")
    image_url: Optional[str] = Field(None, description="Optional image URL")


class PredictionResponse(BaseModel):
    """JSON response returned to React UI."""
    prediction: str = Field(..., description="'REAL' or 'FAKE'")
    confidence: float = Field(..., description="Calibrated confidence score [0.0, 1.0]")
    label2: int = Field(..., description="0 for real, 1 for fake")
    label6: Optional[str] = Field(None, description="6-way fine-grained class label")
    gate_weight: float = Field(..., description="Multimodal gate weight (0=image, 1=text)")
    cross_modal_similarity: float = Field(..., description="Cosine similarity between text and image CLIP embeddings")
    abstained: bool = Field(False, description="True if model abstained due to low confidence")
    explanations: Optional[Dict[str, float]] = Field(default_factory=dict, description="Word/feature importance scores")
