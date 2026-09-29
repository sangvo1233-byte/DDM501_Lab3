# =============================================================================
# Dockerfile for Movie Rating Prediction API
# DDM501 - Lab 3: Testing & CI/CD
# =============================================================================
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# scikit-surprise ships as source only: it needs a C compiler to build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the trained model
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# python:3.10-slim has no curl, so the health check uses the stdlib instead
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import json,sys,urllib.request; \
d=json.load(urllib.request.urlopen('http://localhost:8000/health', timeout=5)); \
sys.exit(0 if d.get('model_loaded') else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
