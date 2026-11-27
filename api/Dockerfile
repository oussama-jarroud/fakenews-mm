# ==============================================================================
# Production Dockerfile for Multimodal Fake News Detection Service
# ==============================================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies for image processing and network health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install python package dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy application modules
COPY src/ ./src/
COPY serve/ ./serve/
COPY api/ ./api/

# Optional: copy checkpoints directory if populated
COPY checkpoints/ ./checkpoints/

EXPOSE 8000

# Docker healthcheck querying /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
