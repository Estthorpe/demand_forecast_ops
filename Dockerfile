# Dockerfile
# Builds the demand-forecast-ops serving container.
# Built by Railway.app from this file — never run locally.
#
# Multi-stage build:
# Stage 1 (builder): installs all dependencies including compilers
# Stage 2 (runtime): copies only what is needed to run the server
# Result: smaller final image, faster deploys, less attack surface

FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer
# If requirements.txt hasn't changed, pip install is skipped on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml .

# Copy trained model artefacts
# These are committed to the repo after running scripts/train.py
COPY models/ ./models/

# Add user-installed packages to PATH
ENV PATH=/root/.local/bin:$PATH

# Port the FastAPI server listens on
# Railway.app reads this to know which port to expose
EXPOSE 8000

# Liveness check — Docker marks container unhealthy if this fails
# Gives the server 30 seconds to start before checking
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the FastAPI server
# --host 0.0.0.0 makes it accessible outside the container
# --workers 1 is correct for a single-model serving container
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
