"""
Configuration — all tuneable values in one place.

Override any setting via environment variables (useful for Docker / cloud).
Example:  DET_CONF_THRESHOLD=0.6 uvicorn app.main:app
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # ── Model paths ──────────────────────────────────────────────────────
    DETECTOR_MODEL_PATH: str = os.getenv(
        "DETECTOR_MODEL_PATH", "models/document_detector.onnx"
    )
    CLASSIFIER_MODEL_PATH: str = os.getenv(
        "CLASSIFIER_MODEL_PATH", "models/readability_classifier.onnx"
    )

    # ── Model 1 (YOLO OBB) ──────────────────────────────────────────────
    DET_IMGSZ: int = int(os.getenv("DET_IMGSZ", "640"))
    DET_CONF_THRESHOLD: float = float(os.getenv("DET_CONF_THRESHOLD", "0.55"))

    # ── Model 2 (MobileNetV3) ───────────────────────────────────────────
    CLF_IMGSZ: int = int(os.getenv("CLF_IMGSZ", "224"))
    CLF_THRESHOLD: float = float(os.getenv("CLF_THRESHOLD", "0.5"))


settings = Settings()
