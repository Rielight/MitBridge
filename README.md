# DocScan API

Computer Vision API untuk memindai dokumen tugas mahasiswa dari foto kamera HP.  
Pipeline: **Detect → Warp → Enhance → Classify → Return**.

---

## Arsitektur Pipeline

```
Camera Photo (JPEG/PNG)
    │
    ▼
┌─────────────────────────────────┐
│  Model 1: YOLO11n-OBB           │   ← Detect document location
│  Input:  640×640 (letterbox)     │      via Oriented Bounding Box
│  Output: [cx,cy,w,h,angle,conf] │
└──────────────┬──────────────────┘
               │ detected? ──No──► return {detected: false}
               │ Yes
               ▼
┌─────────────────────────────────┐
│  Perspective Warp                │   ← 4 corner points → rectangle
│  cv2.warpPerspective             │      pada resolusi asli (bukan 640)
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  CLAHE Enhancement               │   ← Grayscale → CLAHE → BGR
│  clipLimit=2.0, grid=(8,8)       │      meratakan pencahayaan
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  Model 2: MobileNetV3-Small     │   ← Binary: readable vs unreadable
│  Input:  224×224 (ImageNet norm) │      sigmoid embedded in ONNX
│  Output: P(unreadable) 0.0~1.0  │
└──────────────┬──────────────────┘
               │ readable? ──No──► return {readable: false}
               │ Yes
               ▼
        Return processed JPEG
   (warped + CLAHE enhanced document)
```

---

## Struktur Project

```
docscan-api/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, routes, lifespan
│   ├── pipeline.py       # Inference chain (Model1 → Warp → CLAHE → Model2)
│   └── config.py         # Settings via environment variables
├── models/
│   ├── document_detector.onnx        # ← TARUH FILE ONNX MODEL 1 DI SINI
│   └── readability_classifier.onnx   # ← TARUH FILE ONNX MODEL 2 DI SINI
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Siapkan model weights

Salin file ONNX dari hasil training ke folder `models/`:

```bash
# Dari notebook 1 (YOLO OBB)
cp runs/model1_yolo/document_detector/weights/best.onnx models/document_detector.onnx

# Dari notebook 2 (MobileNetV3)
cp runs/classification/document_readability/readability_classifier.onnx models/readability_classifier.onnx
```

### 2. Jalankan Lokal (tanpa Docker)

```bash
# Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Jalankan server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Jalankan dengan Docker

```bash
# Build image
docker build -t docscan-api .

# Run container
docker run -p 8000:8000 docscan-api
```

---

## API Usage

### `POST /scan`

Upload foto kamera, terima hasil scan.

```bash
# Contoh dengan curl
curl -X POST http://localhost:8000/scan \
     -F "file=@photo.jpg" \
     -o result.jpg -D -
```

**Jika dokumen terdeteksi DAN readable:**
- HTTP 200, `Content-Type: image/jpeg`
- Body: gambar JPEG hasil warp + CLAHE
- Headers:
  - `X-Detected: true`
  - `X-Readable: true`
  - `X-Detection-Confidence: 0.9234`
  - `X-Readability-Score: 0.1023` (P(unreadable), makin rendah makin bagus)
  - `X-Inference-Ms: 342.1`

**Jika gagal (tidak terdeteksi / unreadable):**
- HTTP 200, `Content-Type: application/json`
- Body:
```json
{
    "detected": true,
    "readable": false,
    "detection_confidence": 0.8912,
    "readability_score": 0.7834,
    "stage_failed": "readability",
    "inference_ms": 287.3
}
```

### `GET /health`

```json
{"status": "ok", "models_loaded": true}
```

---

## Deployment ke Cloud Free Tier

### Railway / Render / Fly.io

Ketiga platform ini mendukung deploy dari Dockerfile secara langsung.

**Railway:**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login & deploy
railway login
railway init
railway up
```

**Render:**
1. Push repo ke GitHub
2. Di Render dashboard → New Web Service → Connect repo
3. Runtime: Docker
4. Render otomatis build dari Dockerfile

**Fly.io:**
```bash
fly launch          # Setup pertama kali
fly deploy          # Deploy / update
```

### Environment Variables (opsional)

Semua config bisa di-override via env var:

| Variable                 | Default                              | Keterangan                     |
|--------------------------|--------------------------------------|--------------------------------|
| `DETECTOR_MODEL_PATH`   | `models/document_detector.onnx`      | Path ke ONNX Model 1          |
| `CLASSIFIER_MODEL_PATH` | `models/readability_classifier.onnx` | Path ke ONNX Model 2          |
| `DET_CONF_THRESHOLD`    | `0.5`                                | Min confidence untuk deteksi   |
| `CLF_THRESHOLD`          | `0.5`                                | Threshold readable/unreadable  |

---

## Batasan & Constraint

- **1 vCPU, < 1GB RAM** — model nano + headless OpenCV, total RAM ~200-300 MB
- **Inference time** — target < 2 detik total pipeline
- **No PyTorch** — production hanya pakai ONNX Runtime
- **Single worker** — uvicorn dengan 1 worker (sesuai 1 vCPU)
- **Max file size** — 10 MB per upload
