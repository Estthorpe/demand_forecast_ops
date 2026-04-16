FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

COPY src/ ./src/
COPY pyproject.toml .
COPY .dvc/ ./.dvc/
COPY start.sh .

ENV PATH=/root/.local/bin:$PATH

RUN pip install --no-cache-dir -e .
RUN sed -i 's/\r//' start.sh && chmod +x start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
