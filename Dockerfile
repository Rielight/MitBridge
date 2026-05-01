# ============================================================
#  DocScan API — Hugging Face Spaces Dockerfile
#  Target: Free-tier cloud (2 vCPU, 16GB RAM for HF CPU tier)
# ============================================================

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 1. Install system deps FIRST while we are still the 'root' user
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 2. --- HUGGING FACE REQUIRED: Non-root user setup ---
RUN useradd -m -u 1000 user

# Switch to the new user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory inside the user's home
WORKDIR $HOME/app

# 3. Install Python deps (Make sure to use --chown=user)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application code and models (Make sure to use --chown=user)
COPY --chown=user app/ app/
COPY --chown=user models/ models/

# 5. --- HUGGING FACE REQUIRED: Port 7860 ---
EXPOSE 7860

# Update Healthcheck to use the new HF port
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run with single worker on port 7860
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "30", \
     "--log-level", "info"]