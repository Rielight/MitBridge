"""
FastAPI application for Document Scanning API.

Endpoints
---------
POST /scan          Upload image → get scan result + processed image
GET  /health        Liveness / readiness probe
GET  /              Root info
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.pipeline import ScanPipeline
from app.config import settings

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("docscan")

# ---------------------------------------------------------------------------
#  Application lifespan  (model loading at startup)
# ---------------------------------------------------------------------------

pipeline: ScanPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once on startup, release on shutdown."""
    global pipeline

    logger.info("Loading models …")
    t0 = time.perf_counter()

    pipeline = ScanPipeline(
        detector_path=settings.DETECTOR_MODEL_PATH,
        classifier_path=settings.CLASSIFIER_MODEL_PATH,
        det_imgsz=settings.DET_IMGSZ,
        clf_imgsz=settings.CLF_IMGSZ,
        det_conf=settings.DET_CONF_THRESHOLD,
        clf_threshold=settings.CLF_THRESHOLD,
    )

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("Models loaded in %.0f ms", elapsed)

    yield  # app runs here

    pipeline = None
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
#  FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DocScan API",
    description="Scan student assignment documents from camera photos. "
                "Detects the document, corrects perspective, and checks readability.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp",
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _decode_image(raw: bytes) -> np.ndarray:
    """bytes → BGR numpy array via OpenCV."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    """BGR numpy → JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "DocScan API",
        "version": "1.0.0",
        "endpoints": {
            "POST /scan": "Upload an image to scan a document",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": pipeline is not None,
    }


@app.post("/scan")
async def scan_document(file: UploadFile = File(..., description="Camera photo of a document")):
    """
    Main scanning endpoint.

    **Request:** ``multipart/form-data`` with field ``file`` (JPEG/PNG).

    **Response (document detected & readable):**
    Returns the processed image directly as ``image/jpeg``
    with metadata in custom headers:

    - ``X-Detected``: ``true``
    - ``X-Readable``: ``true``
    - ``X-Detection-Confidence``: float
    - ``X-Readability-Score``: float (P(unreadable), lower = better)
    - ``X-Inference-Ms``: total pipeline time

    **Response (not detected / unreadable):**
    Returns JSON with ``detected``, ``readable``, ``stage_failed``, scores.
    """

    if pipeline is None:
        raise HTTPException(503, "Models not loaded yet")

    # ── Validate input ──────────────────────────────────────────────────
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            415,
            f"Unsupported file type: {file.content_type}. "
            f"Accepted: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    raw = await file.read()

    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large ({len(raw)} bytes). Max: {MAX_FILE_SIZE}")

    if len(raw) == 0:
        raise HTTPException(400, "Empty file")

    # ── Decode ──────────────────────────────────────────────────────────
    try:
        image = _decode_image(raw)
    except ValueError:
        raise HTTPException(400, "Cannot decode image. Send a valid JPEG/PNG.")

    # ── Run pipeline ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    result = pipeline.run(image)
    inference_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "scan  detected=%s  readable=%s  det_conf=%s  read_score=%s  %.0fms",
        result.detected, result.readable,
        result.detection_confidence, result.readability_score,
        inference_ms,
    )

    # ── Response: success → return image ────────────────────────────────
    if result.detected and result.readable and result.image is not None:
        jpeg_bytes = _encode_jpeg(result.image)

        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers={
                "X-Detected": "true",
                "X-Readable": "true",
                "X-Detection-Confidence": str(round(result.detection_confidence, 4)),
                "X-Readability-Score": str(round(result.readability_score, 4)),
                "X-Inference-Ms": str(round(inference_ms, 1)),
            },
        )

    # ── Response: failure → return JSON ─────────────────────────────────
    body = result.to_api_dict()
    body["inference_ms"] = round(inference_ms, 1)

    return JSONResponse(content=body, status_code=200)
