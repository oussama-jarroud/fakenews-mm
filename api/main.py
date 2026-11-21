"""FastAPI Application for Multimodal Fake News Detection Service.

Provides real-time inference endpoints conforming to Data Contract #4.
"""

from contextlib import asynccontextmanager
import logging
from typing import Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictionRequest, PredictionResponse
from serve.detector import FakeNewsDetector

logger = logging.getLogger("fnd.api")
logging.basicConfig(level=logging.INFO)

# Global detector instance
detector: FakeNewsDetector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application."""
    global detector
    if detector is None:
        logger.info("Starting up Fake News Detection Service...")
        detector = FakeNewsDetector()
        app.state.detector = detector
        logger.info("Service initialized and ready to receive requests.")
    else:
        app.state.detector = detector
    yield
    logger.info("Shutting down Fake News Detection Service...")


app = FastAPI(
    title="Multimodal Fake News Detection API",
    description="Consistency-Aware Multimodal Fake News Detector exposing predictions, calibrated confidence, cross-modal similarity, and gating weights.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React UI integration (Contract #4)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Root info endpoint."""
    return {
        "service": "Multimodal Fake News Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint reporting model and device status."""
    global detector
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detector model is not yet initialized.",
        )
    return {
        "status": "ok",
        "model_loaded": True,
        "is_trained": detector.is_trained,
        "device": detector.device,
        "clip_model": detector.clip_model_name,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest):
    """Predict whether a multimodal news post is REAL or FAKE.

    Calculates:
      - Calibrated classification confidence
      - Dynamic gate weight (modality reliance: text vs. image)
      - Cross-modal cosine similarity score
      - Abstention flag for ambiguous predictions
    """
    global detector
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detector model is not available.",
        )

    text_cleaned = request.text.strip()
    if not text_cleaned:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input 'text' must not be empty or whitespace only.",
        )

    try:
        result = detector.predict(text=text_cleaned, image_url=request.image_url)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error("Inference error during /predict: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failed: {str(e)}",
        )
