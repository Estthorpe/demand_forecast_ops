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
COPY configs/ ./configs/
COPY pyproject.toml .
COPY .dvc/ ./.dvc/

ENV PATH=/root/.local/bin:$PATH

# Install project so src.* imports resolve
RUN pip install --no-cache-dir -e .

# Pull model artefacts from DagsHub at build time
ARG DAGSHUB_USERNAME
ARG DAGSHUB_TOKEN
RUN dvc remote modify origin --local auth basic \
    && dvc remote modify origin --local user ${DAGSHUB_USERNAME} \
    && dvc remote modify origin --local password ${DAGSHUB_TOKEN} \
    && dvc pull

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]git status
