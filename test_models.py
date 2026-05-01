"""
Test Script — Validate both ONNX models after export.
Run from project root: python test_models.py

Tests:
  1. Model 1 (YOLO OBB): Output shape, sigmoid, detection on real images
  2. Model 2 (Classifier): Output range, class separation
  3. Full pipeline chain: Detection → Warp → CLAHE → Classification
"""

import os
import sys
import glob
import numpy as np
import cv2
import onnxruntime as ort

# ── Config ──────────────────────────────────────────────────────────────
DETECTOR_PATH = "models/document_detector.onnx"
CLASSIFIER_PATH = "models/readability_classifier.onnx"

YOLO_TEST_DIR = "dataset/final_split/model1_yolo/images/test"
CLF_READABLE_DIR = "dataset/final_split/model2_classification/train/0_readable"
CLF_UNREADABLE_DIR = "dataset/final_split/model2_classification/train/1_unreadable"
COCO_DIR = "dataset/sampling/model1_yolo/coco_negative"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

passed = 0
failed = 0


def status(test_name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ✅ {test_name}")
    else:
        failed += 1
        print(f"  ❌ {test_name}  — {detail}")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def find_images(folder, n=5):
    if not os.path.exists(folder):
        return []
    imgs = glob.glob(os.path.join(folder, "*.jpg")) + \
           glob.glob(os.path.join(folder, "*.jpeg")) + \
           glob.glob(os.path.join(folder, "*.png"))
    return imgs[:n]


def letterbox(img, imgsz=640):
    h, w = img.shape[:2]
    ratio = min(imgsz / h, imgsz / w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    pad_w = (imgsz - new_w) / 2
    pad_h = (imgsz - new_h) / 2

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))

    blob = padded[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[np.newaxis])
    return blob, ratio, (pad_w, pad_h)


def clf_preprocess(img_bgr):
    resized = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    blob = resized[:, :, ::-1].astype(np.float32) / 255.0
    blob = (blob - IMAGENET_MEAN) / IMAGENET_STD
    blob = np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[np.newaxis])
    return blob


# ════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  MODEL 1: YOLO OBB Document Detector")
print("=" * 60)

if not os.path.exists(DETECTOR_PATH):
    print(f"  ❌ File not found: {DETECTOR_PATH}")
    sys.exit(1)

det_sess = ort.InferenceSession(DETECTOR_PATH, providers=["CPUExecutionProvider"])
det_input = det_sess.get_inputs()[0]
det_output = det_sess.get_outputs()[0]

# Test 1.1: I/O shape
print(f"\n  Input:  {det_input.name} {det_input.shape}")
print(f"  Output: {det_output.name} {det_output.shape}")

expected_shape = [1, 6, 8400]  # nms=False
status("Output shape is (1, 6, 8400) [nms=False]",
       det_output.shape == expected_shape or det_output.shape == ['1', '6', '8400'],
       f"Got {det_output.shape}")

# Test 1.2: Dummy inference runs without error
dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
out = det_sess.run(None, {det_input.name: dummy})[0]
status("Dummy inference runs OK", out.shape == (1, 6, 8400), f"Shape: {out.shape}")

# Test 1.3: Output contains raw logits (not sigmoid) — values can be > 1
preds = out[0].T  # (8400, 6)
max_logit = preds[:, 4].max()
status("Column 4 contains raw logits (can be > 1 or < 0)",
       max_logit > 1.0 or preds[:, 4].min() < 0.0,
       f"max={max_logit:.4f} — might already be sigmoided?")

# Test 1.4: Detection on real document images
print(f"\n  Testing on real document images ({YOLO_TEST_DIR})...")
doc_images = find_images(YOLO_TEST_DIR, n=5)
if not doc_images:
    # Fallback: try any test image folder
    for alt in ["dataset/final_split/model1_yolo/images/val"]:
        doc_images = find_images(alt, n=5)
        if doc_images:
            break

if doc_images:
    det_count = 0
    for img_path in doc_images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        blob, _, _ = letterbox(img)
        raw = det_sess.run(None, {det_input.name: blob})[0]
        preds = raw[0].T
        confs = sigmoid(preds[:, 4])
        max_conf = confs.max()
        fname = os.path.basename(img_path)[:40]
        detected = max_conf >= 0.5
        if detected:
            det_count += 1
        print(f"    {fname:40s}  conf={max_conf:.4f}  {'DETECTED' if detected else 'missed'}")
    status(f"Detection on documents: {det_count}/{len(doc_images)} detected",
           det_count >= len(doc_images) * 0.6,
           f"Only {det_count}/{len(doc_images)}")
else:
    print("    ⚠️  No test images found, skipping")

# Test 1.5: COCO negative should NOT detect
print(f"\n  Testing on COCO negatives ({COCO_DIR})...")
coco_images = find_images(COCO_DIR, n=10)
if coco_images:
    false_pos = 0
    for img_path in coco_images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        blob, _, _ = letterbox(img)
        raw = det_sess.run(None, {det_input.name: blob})[0]
        preds = raw[0].T
        confs = sigmoid(preds[:, 4])
        max_conf = confs.max()
        fname = os.path.basename(img_path)[:40]
        detected = max_conf >= 0.55
        if detected:
            false_pos += 1
        print(f"    {fname:40s}  conf={max_conf:.4f}  {'FALSE POS!' if detected else 'ok'}")
    status(f"COCO false positives: {false_pos}/{len(coco_images)}",
           false_pos <= len(coco_images) * 0.3,
           f"{false_pos} false positives")
else:
    print("    ⚠️  No COCO images found, skipping")


# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  MODEL 2: MobileNetV3-Small Readability Classifier")
print("=" * 60)

if not os.path.exists(CLASSIFIER_PATH):
    print(f"  ❌ File not found: {CLASSIFIER_PATH}")
    sys.exit(1)

clf_sess = ort.InferenceSession(CLASSIFIER_PATH, providers=["CPUExecutionProvider"])
clf_input = clf_sess.get_inputs()[0]
clf_output = clf_sess.get_outputs()[0]

print(f"\n  Input:  {clf_input.name} {clf_input.shape}")
print(f"  Output: {clf_output.name} {clf_output.shape}")

# Test 2.1: Dummy inference
dummy_clf = np.random.rand(1, 3, 224, 224).astype(np.float32)
out_clf = clf_sess.run(None, {clf_input.name: dummy_clf})[0]
status("Dummy inference runs OK", out_clf.shape == (1, 1), f"Shape: {out_clf.shape}")

# Test 2.2: Check if sigmoid is embedded (output should be 0-1)
raw_val = float(out_clf[0][0])
is_sigmoid = 0.0 <= raw_val <= 1.0
print(f"    Dummy output value: {raw_val:.6f}")
status("Output is in [0, 1] range (sigmoid embedded)", is_sigmoid,
       f"Value={raw_val:.4f}, sigmoid might not be embedded")

# Test 2.3: Readable images → low scores
print(f"\n  Testing READABLE images...")
# Try multiple possible paths
for clf_r_dir in [CLF_READABLE_DIR,
                  "dataset/final_split/model2_clf/train/0_readable",
                  "dataset/final_split/model2_classification/train/0_readable"]:
    readable_imgs = find_images(clf_r_dir, n=5)
    if readable_imgs:
        break

readable_scores = []
if readable_imgs:
    for img_path in readable_imgs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        blob = clf_preprocess(img)
        val = float(clf_sess.run(None, {clf_input.name: blob})[0][0][0])
        readable_scores.append(val)
        fname = os.path.basename(img_path)[:50]
        print(f"    {fname:50s}  score={val:.4f}  {'✓ readable' if val < 0.5 else '✗ WRONG'}")
    avg_r = np.mean(readable_scores)
    status(f"Readable avg score: {avg_r:.4f} (should be < 0.3)",
           avg_r < 0.3, f"avg={avg_r:.4f}")
else:
    print("    ⚠️  No readable images found")

# Test 2.4: Unreadable images → high scores
print(f"\n  Testing UNREADABLE images...")
for clf_u_dir in [CLF_UNREADABLE_DIR,
                  "dataset/final_split/model2_clf/train/1_unreadable",
                  "dataset/final_split/model2_classification/train/1_unreadable"]:
    unreadable_imgs = find_images(clf_u_dir, n=5)
    if unreadable_imgs:
        break

unreadable_scores = []
if unreadable_imgs:
    for img_path in unreadable_imgs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        blob = clf_preprocess(img)
        val = float(clf_sess.run(None, {clf_input.name: blob})[0][0][0])
        unreadable_scores.append(val)
        fname = os.path.basename(img_path)[:50]
        print(f"    {fname:50s}  score={val:.4f}  {'✓ unreadable' if val >= 0.5 else '✗ WRONG'}")
    avg_u = np.mean(unreadable_scores)
    status(f"Unreadable avg score: {avg_u:.4f} (should be > 0.7)",
           avg_u > 0.7, f"avg={avg_u:.4f}")
else:
    print("    ⚠️  No unreadable images found")

# Test 2.5: Class separation
if readable_scores and unreadable_scores:
    gap = np.mean(unreadable_scores) - np.mean(readable_scores)
    max_r = max(readable_scores)
    min_u = min(unreadable_scores)
    margin = min_u - max_r
    print(f"\n    Class separation:")
    print(f"      Readable  range: [{min(readable_scores):.4f}, {max_r:.4f}]")
    print(f"      Unreadable range: [{min_u:.4f}, {max(unreadable_scores):.4f}]")
    print(f"      Gap (avg):  {gap:.4f}")
    print(f"      Margin (worst case): {margin:.4f}")
    status(f"Classes don't overlap (margin={margin:.4f})",
           margin > 0, f"OVERLAP! readable max={max_r:.4f} > unreadable min={min_u:.4f}")
    status(f"Mean gap > 0.5 (gap={gap:.4f})",
           gap > 0.5, f"Gap too small")


# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  FULL PIPELINE: Detection → Warp → CLAHE → Classification")
print("=" * 60)

doc_test_imgs = find_images(YOLO_TEST_DIR, n=3) or find_images(
    "dataset/final_split/model1_yolo/images/val", n=3)

if doc_test_imgs:
    for img_path in doc_test_imgs:
        fname = os.path.basename(img_path)[:50]
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Stage 1: Detection
        blob, ratio, (pad_w, pad_h) = letterbox(img)
        raw = det_sess.run(None, {det_input.name: blob})[0]
        preds = raw[0].T
        confs = sigmoid(preds[:, 4])
        best_idx = np.argmax(confs)
        best_conf = confs[best_idx]

        if best_conf < 0.5:
            print(f"\n  {fname}: NOT DETECTED (conf={best_conf:.4f})")
            continue

        # Get corners — format is [cx, cy, w, h, conf_logit, angle]
        cx, cy, w, h = preds[best_idx, :4]
        angle = preds[best_idx, 5]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = w / 2, h / 2
        corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = (corners @ R.T) + np.array([cx, cy])

        corners[:, 0] = (corners[:, 0] - pad_w) / ratio
        corners[:, 1] = (corners[:, 1] - pad_h) / ratio
        h_orig, w_orig = img.shape[:2]
        corners[:, 0] = np.clip(corners[:, 0], 0, w_orig - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h_orig - 1)

        # Order corners
        rect = np.zeros((4, 2), dtype=np.float32)
        s = corners.sum(axis=1)
        d = np.diff(corners, axis=1).ravel()
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]
        rect[1] = corners[np.argmin(d)]
        rect[3] = corners[np.argmax(d)]

        # Stage 2: Warp
        tl, tr, br, bl = rect
        out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

        if out_w < 32 or out_h < 32:
            print(f"\n  {fname}: WARP TOO SMALL ({out_w}x{out_h})")
            continue

        dst = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (out_w, out_h))

        # Stage 3: CLAHE
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

        # Stage 4: Classification
        blob_clf = clf_preprocess(enhanced)
        score = float(clf_sess.run(None, {clf_input.name: blob_clf})[0][0][0])
        readable = score < 0.5

        print(f"\n  {fname}:")
        print(f"    Detection: conf={best_conf:.4f}")
        print(f"    Warped size: {out_w}x{out_h}")
        print(f"    Readability: score={score:.4f} → {'READABLE ✅' if readable else 'UNREADABLE ❌'}")

    status("Full pipeline runs end-to-end without errors", True)
else:
    print("  ⚠️  No test images found for pipeline test")


# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  SUMMARY: {passed} passed, {failed} failed")
print("=" * 60)

if failed == 0:
    print("  🎉 All tests passed! Models are ready for deployment.")
else:
    print("  ⚠️  Some tests failed. Review output above before deploying.")

sys.exit(0 if failed == 0 else 1)