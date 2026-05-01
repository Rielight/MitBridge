"""
Document Scanning Pipeline — Full Inference Chain
===================================================
Model 1 (YOLO11n-OBB)  → Detect & locate document via Oriented Bounding Box
     ↓ warp + CLAHE
Model 2 (MobileNetV3-Small) → Classify readable vs unreadable

Dependencies (production-only, NO PyTorch):
    numpy, opencv-python-headless, onnxruntime
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Immutable result returned by the pipeline."""

    detected: bool = False
    readable: bool = False
    image: np.ndarray | None = None  # BGR, warped + CLAHE'd document

    # Diagnostics (optional, for debugging / logging)
    detection_confidence: float | None = None
    readability_score: float | None = None  # P(unreadable), lower = more readable
    corners: np.ndarray | None = None       # (4, 2) in original-image coords
    stage_failed: str | None = None         # which stage returned early

    def to_api_dict(self) -> dict:
        """Serialise for JSON response (image excluded — sent separately)."""
        return {
            "detected": self.detected,
            "readable": self.readable,
            "detection_confidence": round(self.detection_confidence, 4)
            if self.detection_confidence is not None
            else None,
            "readability_score": round(self.readability_score, 4)
            if self.readability_score is not None
            else None,
            "stage_failed": self.stage_failed,
        }


# ---------------------------------------------------------------------------
#  Model 1 — YOLO11n-OBB Document Detector
# ---------------------------------------------------------------------------

class DocumentDetector:
    """
    YOLO11n-OBB via ONNX Runtime.

    ONNX exported with ``nms=False``  →  output shape (1, 6, 8400)
    Format per anchor (after transpose): [cx, cy, w, h, class_logit, angle_rad]
    Coordinates are in 640×640 letterbox space.
    Single-class OBB → column 4 is the raw logit for class "document",
    column 5 is the angle in radians.
    """

    def __init__(self, model_path: str, imgsz: int = 640, conf: float = 0.5):
        self.imgsz = imgsz
        self.conf = conf

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1          # 1 vCPU constraint
        so.inter_op_num_threads = 1
        so.enable_mem_pattern = True

        self.session = ort.InferenceSession(
            model_path, sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info("DocumentDetector loaded  (%s)", model_path)

    # -- preprocessing -----------------------------------------------------

    def _letterbox(self, img: np.ndarray):
        """Resize + pad to (imgsz, imgsz) keeping aspect ratio.  Returns
        blob (1,3,H,W) float32, scale ratio, and (pad_w, pad_h)."""
        h, w = img.shape[:2]
        ratio = min(self.imgsz / h, self.imgsz / w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        pad_w = (self.imgsz - new_w) / 2
        pad_h = (self.imgsz - new_h) / 2

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # BGR→RGB, /255, HWC→CHW, batch dim
        blob = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[np.newaxis])
        return blob, ratio, (pad_w, pad_h)

    # -- postprocessing ----------------------------------------------------

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    @staticmethod
    def _xywhr_to_corners(cx, cy, w, h, angle):
        """OBB centre repr → 4 corner points (4,2)."""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = w / 2, h / 2
        corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return (corners @ R.T) + np.array([cx, cy])

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """Sort 4 pts → [top-left, top-right, bottom-right, bottom-left]."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        rect[0] = pts[np.argmin(s)]   # TL
        rect[2] = pts[np.argmax(s)]   # BR
        rect[1] = pts[np.argmin(d)]   # TR
        rect[3] = pts[np.argmax(d)]   # BL
        return rect

    @staticmethod
    def _compute_obb_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Approximate IoU between two OBBs using cv2.rotatedRectangleIntersection.
        box format: [cx, cy, w, h, angle_rad]."""
        # cv2.rotatedRectangleIntersection expects ((cx,cy), (w,h), angle_degrees)
        rect_a = (
            (float(box_a[0]), float(box_a[1])),
            (float(box_a[2]), float(box_a[3])),
            float(np.degrees(box_a[4])),
        )
        rect_b = (
            (float(box_b[0]), float(box_b[1])),
            (float(box_b[2]), float(box_b[3])),
            float(np.degrees(box_b[4])),
        )
        ret, region = cv2.rotatedRectangleIntersection(rect_a, rect_b)
        if ret == cv2.INTERSECT_NONE or region is None:
            return 0.0
        inter = cv2.contourArea(region)
        area_a = box_a[2] * box_a[3]
        area_b = box_b[2] * box_b[3]
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _rotated_nms(self, boxes: np.ndarray, scores: np.ndarray,
                     iou_threshold: float = 0.5) -> np.ndarray:
        """Simple greedy rotated NMS.
        boxes: (N, 5) [cx, cy, w, h, angle_rad]
        scores: (N,)
        Returns indices to keep."""
        order = np.argsort(scores)[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            ious = np.array([
                self._compute_obb_iou(boxes[i], boxes[j]) for j in rest
            ])
            order = rest[ious < iou_threshold]
        return np.array(keep, dtype=int)

    # -- public API --------------------------------------------------------

    def detect(self, img: np.ndarray):
        """
        Returns (corners_original | None, confidence | None).
        corners_original: (4,2) float32 in *original* image coordinates.
        """
        blob, ratio, (pad_w, pad_h) = self._letterbox(img)

        # Raw output: (1, 6, 8400) → transpose to (8400, 6)
        raw = self.session.run(None, {self.input_name: blob})[0]  # (1, 6, 8400)
        preds = raw[0].T  # (8400, 6) → [cx, cy, w, h, class_logit, angle]

        # Apply sigmoid to class logits (col 4) to get confidence
        confs = self._sigmoid(preds[:, 4])

        # Filter by confidence
        mask = confs >= self.conf
        if not np.any(mask):
            return None, None

        valid = preds[mask]
        valid_confs = confs[mask]

        # Rotated NMS — boxes need [cx, cy, w, h, angle]
        boxes = np.column_stack([valid[:, :4], valid[:, 5]])  # cx,cy,w,h + angle
        keep = self._rotated_nms(boxes, valid_confs, iou_threshold=0.5)

        # Take best detection after NMS
        best_idx = keep[0]
        best = valid[best_idx]
        best_conf = valid_confs[best_idx]

        cx, cy, w, h = best[:4]
        angle = best[5]
        corners = self._xywhr_to_corners(cx, cy, w, h, angle)

        # Map 640-space → original-space
        corners[:, 0] = (corners[:, 0] - pad_w) / ratio
        corners[:, 1] = (corners[:, 1] - pad_h) / ratio

        h_orig, w_orig = img.shape[:2]
        corners[:, 0] = np.clip(corners[:, 0], 0, w_orig - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h_orig - 1)

        return self._order_corners(corners.astype(np.float32)), float(best_conf)


# ---------------------------------------------------------------------------
#  Perspective Warp  (between Model 1 and Model 2)
# ---------------------------------------------------------------------------

def perspective_warp(image: np.ndarray, ordered_corners: np.ndarray) -> np.ndarray:
    """Warp a quadrilateral region into an upright rectangle."""
    tl, tr, br, bl = ordered_corners

    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    # Safety: prevent degenerate warps
    if out_w < 32 or out_h < 32:
        raise ValueError(f"Warped size too small: {out_w}×{out_h}")

    dst = np.array([
        [0, 0], [out_w - 1, 0],
        [out_w - 1, out_h - 1], [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered_corners, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))


# ---------------------------------------------------------------------------
#  CLAHE preprocessing  (between Warp and Model 2)
# ---------------------------------------------------------------------------

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    """Grayscale → CLAHE → back to BGR.  Identical to augment.py's
    ``preprocess_for_model2``."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
#  Model 2 — MobileNetV3-Small Readability Classifier
# ---------------------------------------------------------------------------

class ReadabilityClassifier:
    """
    MobileNetV3-Small binary classifier via ONNX Runtime.

    ONNX has sigmoid embedded → output is P(unreadable) ∈ [0, 1].
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: str, imgsz: int = 224, threshold: float = 0.5):
        self.imgsz = imgsz
        self.threshold = threshold

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            model_path, sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info("ReadabilityClassifier loaded  (%s)", model_path)

    def classify(self, image_bgr: np.ndarray) -> tuple[bool, float]:
        """
        Args:
            image_bgr: warped + CLAHE'd document (BGR, any resolution).
        Returns:
            (readable: bool, prob_unreadable: float)
        """
        resized = cv2.resize(image_bgr, (self.imgsz, self.imgsz),
                             interpolation=cv2.INTER_LINEAR)

        blob = resized[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB
        blob = (blob - self._MEAN) / self._STD                  # ImageNet norm
        blob = np.ascontiguousarray(
            np.transpose(blob, (2, 0, 1))[np.newaxis]           # (1,3,224,224)
        )

        prob = float(self.session.run(None, {self.input_name: blob})[0][0][0])
        return prob < self.threshold, prob


# ---------------------------------------------------------------------------
#  Orchestrator — chains everything together
# ---------------------------------------------------------------------------

class ScanPipeline:
    """
    End-to-end document scanning pipeline.

    Usage::

        pipeline = ScanPipeline("models/detector.onnx", "models/classifier.onnx")
        result   = pipeline.run(image_bgr)
    """

    def __init__(
        self,
        detector_path: str,
        classifier_path: str,
        det_imgsz: int = 640,
        clf_imgsz: int = 224,
        det_conf: float = 0.5,
        clf_threshold: float = 0.5,
    ):
        self.detector = DocumentDetector(detector_path, det_imgsz, det_conf)
        self.classifier = ReadabilityClassifier(
            classifier_path, clf_imgsz, clf_threshold,
        )
        logger.info("ScanPipeline ready.")

    def run(self, image_bgr: np.ndarray) -> PipelineResult:
        """
        Full chain:
          1. Detect document (YOLO OBB)
          2. Perspective-warp to rectangle
          3. CLAHE enhancement
          4. Readability check (MobileNetV3)
          5. Return result

        The returned ``image`` (if readable) is the warped + CLAHE'd BGR.
        """

        # ── Stage 1: Detection ──────────────────────────────────────────
        corners, det_conf = self.detector.detect(image_bgr)

        if corners is None:
            return PipelineResult(
                detected=False, stage_failed="detection",
            )

        # ── Stage 2: Perspective Warp ───────────────────────────────────
        try:
            warped = perspective_warp(image_bgr, corners)
        except ValueError as exc:
            logger.warning("Warp failed: %s", exc)
            return PipelineResult(
                detected=True,
                detection_confidence=det_conf,
                corners=corners,
                stage_failed="warp",
            )

        # ── Stage 3: CLAHE ──────────────────────────────────────────────
        enhanced = apply_clahe(warped)

        # ── Stage 4: Readability Classification ─────────────────────────
        readable, prob_unreadable = self.classifier.classify(enhanced)

        return PipelineResult(
            detected=True,
            readable=readable,
            image=enhanced if readable else None,
            detection_confidence=det_conf,
            readability_score=prob_unreadable,
            corners=corners,
            stage_failed=None if readable else "readability",
        )
