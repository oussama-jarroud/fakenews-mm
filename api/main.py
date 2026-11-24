"""FastAPI Application for Multimodal Fake News Detection Service.

Provides real-time inference endpoints conforming to Section 14 & Appendix A7
of the Course Guide.
"""

from contextlib import asynccontextmanager
import logging
from typing import Dict, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import DemoExample, PredictionRequest, PredictionResponse
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

# Enable CORS for React UI integration (Contract #4 / Section 14)
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
            "examples": "/examples",
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


@app.get("/examples", response_model=List[DemoExample], tags=["Demo"])
async def get_curated_examples():
    """Returns 3 curated demo cases guaranteeing a working demo offline (Page 13)."""
    return [
        DemoExample(
            id="case_1_real",
            title="Consistent Genuine News",
            text="NASA James Webb Space Telescope reveals deepest infrared image of the early universe.",
            image_url="https://images.nasa.gov/details-PIA25325.html",
            expected_verdict="consistent",
            description="High semantic alignment between headline claim and astronomy imagery.",
        ),
        DemoExample(
            id="case_2_mismatch",
            title="Out-of-Context Cheapfake",
            text="Breaking: Catastrophic earthquake levels downtown buildings in capital city today.",
            image_url="https://images.unsplash.com/photo-1546527868-ccb7ee7dfa6a",
            expected_verdict="misleading",
            description="Unrelated photograph paired with sensational claim; cross-modal similarity collapses.",
        ),
        DemoExample(
            id="case_3_uncertain",
            title="Ambiguous / Borderline Claim",
            text="Reports suggest unusual atmospheric phenomenon observed near coastal region.",
            image_url=None,
            expected_verdict="uncertain",
            description="Vague textual description without accompanying visual evidence triggers abstention.",
        ),
    ]


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest):
    """Predict whether a multimodal news post is consistent (REAL) or misleading (FAKE).

    Calculates:
      - verdict: 'misleading' | 'consistent' | 'uncertain' (abstention)
      - p_fake: calibrated probability of falsehood
      - consistency: CLIP cosine similarity
      - modality_weights: dynamic attention allocation (text vs. image)
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
