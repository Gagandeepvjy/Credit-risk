# ── Base image ──────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="credit-risk-team"
LABEL description="Credit Risk Scoring API"

# ── System deps ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ──────────────────────────────────────────────────────
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

# ── Expose & run ─────────────────────────────────────────────────────
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
