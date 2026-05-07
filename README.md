---
title: MitBridge
emoji: 🌉
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# MitBridge
## Project Overview

**MitBridge API** adalah layanan Computer Vision berbasis API yang dirancang khusus untuk memindai, meluruskan, dan mengevaluasi keterbacaan dokumen (seperti tugas siswa) yang diambil melalui foto kamera HP. Sistem ini memecahkan masalah umum pada pengumpulan tugas digital: sudut pengambilan foto yang miring (skewed) dan kualitas gambar yang buruk (blur/gelap).

Sistem ini didesain untuk beroperasi pada lingkungan dengan sumber daya sangat terbatas (Cloud Free Tier) dengan menghilangkan dependensi berat seperti PyTorch dan sepenuhnya menggunakan **ONNX Runtime (CPU-Optimized)**.

Solusi ini menggunakan pipeline Machine Learning dua tahap:
1. **Model 1 (Document Localization):** Menggunakan **YOLO11n-OBB** (Oriented Bounding Box) varian Nano. Model ini mendeteksi 4 titik sudut dokumen terlepas dari rotasi atau orientasi kertas, yang kemudian digunakan untuk melakukan *perspective warping* (meluruskan dokumen menjadi persegi panjang sempurna).
2. **Model 2 (Readability Classification):** Menggunakan **MobileNetV3-Small**. Model klasifikasi biner ini mengevaluasi dokumen yang sudah diluruskan dan diberikan efek CLAHE (peningkatan kontras) untuk menentukan apakah tulisan pada dokumen layak dibaca atau pengguna perlu mengambil ulang foto.

---

## Arsitektur Pipeline

```text
Camera Photo (JPEG/PNG)
    │
    ▼
┌─────────────────────────────────┐
│  Model 1: YOLO11n-OBB           │   ← Detect document location
│  Input:  640×640 (letterbox)    │      via Oriented Bounding Box
│  Output: [cx,cy,w,h,angle,conf] │
└──────────────┬──────────────────┘
               │ detected? ──No──► return {detected: false}
               │ Yes
               ▼
┌─────────────────────────────────┐
│  Perspective Warp               │   ← 4 corner points → rectangle
│  cv2.warpPerspective            │      pada resolusi asli (bukan 640)
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  CLAHE Enhancement              │   ← Grayscale → CLAHE → BGR
│  clipLimit=2.0, grid=(8,8)      │      meratakan pencahayaan
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  Model 2: MobileNetV3-Small     │   ← Binary: readable vs unreadable
│  Input:  224×224 (ImageNet norm)│      sigmoid embedded in ONNX
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

```text
MitBridge/
├── app/
│   ├── main.py           # FastAPI app, routes, lifespan
│   ├── pipeline.py       # Inference chain (Model1 → Warp → CLAHE → Model2)
│   └── config.py         # Settings via environment variables
├── dataset/
│   └── final_split
│       ├── model1_yolo     # Train, test, val (containing images and labels) for YOLO OBB model
│       └── model2_clf      # Train, test, val (containing images and labels) for MobileNet Classification model
├── models/
│   ├── document_detector.onnx        # ONNX Model 1 (YOLO OBB)
│   └── readability_classifier.onnx   # ONNX Model 2 (MobileNetV3)
├── notebooks/
│   ├── classification_model.ipynb      # Training notebook for MobileNet Classification model
│   └── obb_model.ipynb                 # Training notebook for YOLO OBB model
├── src/
│   ├── augment.py          # Script for augmenting the dataset
│   ├── fix_labels.py       # Script for fixing YOLO labels
│   ├── sampling.py         # Script for sampling the raw data
│   ├── split.py            # Script for producing the train, test, and val set
│   └── test_models.py      # Script for testing the ONNX models on sample images
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## API Usage

### `POST /scan`

Endpoint utama untuk mengunggah foto kamera dan menerima hasil pemindaian.

```bash
# Contoh request dengan curl
curl -X POST https://gradienr-mitbridge.hf.space/scan
     -F "file=@photo.jpg"
     -o result.jpg -D -
```

**Jika dokumen terdeteksi DAN dapat dibaca (Readable):**
- **HTTP Status:** 200 OK
- **Content-Type:** `image/jpeg`
- **Body:** Berisi data biner gambar JPEG hasil *perspective warp* dan peningkatan kontras (CLAHE).
- **Custom Headers (Metadata):**
  - `X-Detected: true`
  - `X-Readable: true`
  - `X-Detection-Confidence: 0.9234`
  - `X-Readability-Score: 0.1023` (Nilai probabilitas *unreadable*, makin rendah makin baik)
  - `X-Inference-Ms: 342.1`

**Jika gagal (Tidak terdeteksi ATAU Unreadable):**
- **HTTP Status:** 200 OK
- **Content-Type:** `application/json`
- **Body:**
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

Endpoint untuk pengecekan *liveness* dan kesiapan model.

```bash
# Contoh request dengan curl
curl -X GET https://gradienr-mitbridge.hf.space/health
```

```json
{
    "status": "ok", 
    "models_loaded": true
}
```

---

## Environment Variables

Perilaku sistem dan threshold model dapat disesuaikan tanpa mengubah kode menggunakan *environment variables*:

| Variable                | Default                              | Keterangan                                                                 |
|-------------------------|--------------------------------------|----------------------------------------------------------------------------|
| `DETECTOR_MODEL_PATH`   | `models/document_detector.onnx`      | Path ke file ONNX Model 1.                                                |
| `CLASSIFIER_MODEL_PATH` | `models/readability_classifier.onnx` | Path ke file ONNX Model 2.                                                |
| `DET_CONF_THRESHOLD`    | `0.55`                               | Nilai batas minimal *confidence* bagi YOLO untuk menganggap dokumen valid. |
| `CLF_THRESHOLD`         | `0.5`                                | Threshold pemisah antara dokumen yang dianggap layak dibaca atau tidak.    |

---

## Batasan & Constraint Sistem

API ini dirancang dengan rekayasa sistem yang sangat ketat untuk memastikan keandalan di lingkungan produksi skala kecil:

- **Resource Limits (1 vCPU, < 1GB RAM):** Dengan kombinasi model varian Nano/Small, total konsumsi memori ditahan pada kisaran ~200-300 MB. Thread ONNX Runtime dibatasi secara eksplisit menjadi 1 thread untuk mencegah CPU *thrashing*.
- **Latency & Performance:** Target total waktu inferensi (*end-to-end pipeline*) berada di bawah 2 detik per gambar.
- **Zero PyTorch Dependency:** Pipeline produksi sama sekali tidak menyertakan PyTorch. Seluruh proses *inference* murni mengandalkan ONNX Runtime (CPUExecutionProvider).
- **Concurrency:** Uvicorn berjalan dengan 1 *worker* sinkron dengan spesifikasi sistem (1 vCPU).
- **Security:** Batasan ukuran unggahan file maksimal adalah 10 MB untuk mencegah *Out-Of-Memory* (OOM).