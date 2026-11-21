"""Unit and integration tests for FastAPI Fake News Detection Service."""

from PIL import Image
import pytest
from fastapi.testclient import TestClient

from api.main import app
from serve.detector import FakeNewsDetector


@pytest.fixture(scope="module")
def client():
    """TestClient fixture with test detector instance."""
    # Ensure detector runs with mock_encoders=True for fast offline unit testing
    app.state.detector = FakeNewsDetector(mock_encoders=True)
    import api.main
    api.main.detector = app.state.detector

    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    """Verifies that GET / returns service information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_endpoint(client: TestClient):
    """Verifies that GET /health returns operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "device" in data


def test_predict_text_only(client: TestClient):
    """Verifies /predict inference when only headline text is provided."""
    payload = {
        "text": "NASA rover discovers evidence of ancient microbial life on Mars.",
        "image_url": None,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in ["REAL", "FAKE"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["label2"] in [0, 1]
    assert 0.0 <= data["gate_weight"] <= 1.0
    assert isinstance(data["cross_modal_similarity"], float)
    assert isinstance(data["abstained"], bool)
    assert isinstance(data["explanations"], dict)
    assert "text_gate_weight" in data["explanations"]


def test_predict_multimodal(client: TestClient):
    """Verifies /predict with multimodal input."""
    payload = {
        "text": "Scientists announce groundbreaking renewable battery that charges in seconds.",
        "image_url": "https://example.com/battery.jpg",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in ["REAL", "FAKE"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["gate_weight"] <= 1.0
    assert "cross_modal_alignment" in data["explanations"]


def test_predict_validation_empty_text(client: TestClient):
    """Verifies that empty or whitespace-only text returns HTTP 422."""
    payload = {"text": "   ", "image_url": None}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_validation_missing_text(client: TestClient):
    """Verifies that missing 'text' field returns HTTP 422."""
    payload = {"image_url": "https://example.com/test.jpg"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
