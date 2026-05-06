import os
import random
import shutil
import pandas as pd
import glob
import cv2
import numpy as np

def setup_directories(base_dir):
    """Membuat struktur direktori untuk identifikasi."""
    dirs = [
        f"{base_dir}/model1_yolo/roboflow/images",
        f"{base_dir}/model1_yolo/roboflow/labels",
        f"{base_dir}/model1_yolo/smartdoc2015/images",
        f"{base_dir}/model1_yolo/smartdoc2015/labels",
        f"{base_dir}/model1_yolo/coco_negative",
        f"{base_dir}/model2_clf/base_crops"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return base_dir

def sample_coco(raw_path, out_path, num_samples):
    print(f"Sampling COCO Negative: {num_samples} images...")
    images = glob.glob(os.path.join(raw_path, "*.jpg"))
    sampled = random.sample(images, min(num_samples, len(images)))

    for img in sampled:
        shutil.copy(img, os.path.join(out_path, os.path.basename(img)))

def sample_roboflow_model1(raw_path, out_path, num_samples):
    print(f"Sampling Roboflow (Model 1): {num_samples} images...")
    all_images = []
    for split in ['train', 'valid', 'test']:
        imgs = glob.glob(os.path.join(raw_path, split, 'images', '*.*'))
        all_images.extend(imgs)

    sampled = random.sample(all_images, min(num_samples, len(all_images)))

    for img_path in sampled:
        shutil.copy(img_path, os.path.join(out_path, 'images', os.path.basename(img_path)))
        label_path = img_path.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).rsplit('.', 1)[0] + '.txt'
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(out_path, 'labels', os.path.basename(label_path)))

def sample_smartdoc2015(raw_path, out_path, num_samples):
    print(f"Sampling & Generating Labels SmartDoc-2015: {num_samples} frames...")
    csv_path = os.path.join(raw_path, 'frames_metadata.csv')
    df = pd.read_csv(csv_path)

    unique_images = df['image_path'].unique().tolist()
    sampled_paths = random.sample(unique_images, min(num_samples, len(unique_images)))

    # Filter dataframe hanya untuk gambar yang tersample
    sampled_df = df[df['image_path'].isin(sampled_paths)]

    success_count = 0
    for index, row in sampled_df.iterrows():
        rel_path = row['image_path']
        src_img_path = os.path.join(raw_path, rel_path)

        if not os.path.exists(src_img_path):
            continue

        # Baca gambar untuk mendapatkan dimensi aslinya
        img = cv2.imread(src_img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Copy gambar ke folder output
        dst_name = rel_path.replace('/', '_') # Flatten nama agar tidak masuk subfolder
        dst_img_path = os.path.join(out_path, 'images', dst_name)
        shutil.copy(src_img_path, dst_img_path)

        # 3. Ekstrak koordinat (urutannya: Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        # Sesuai standar YOLO OBB.
        pts = [
            (row['tl_x'], row['tl_y']),
            (row['tr_x'], row['tr_y']),
            (row['br_x'], row['br_y']),
            (row['bl_x'], row['bl_y'])
        ]

        # 4. Normalisasi dan Clamping
        norm_pts = []
        for x, y in pts:
            # Membagi dengan resolusi lalu dijepit antara 0.0 sampai 1.0
            nx = max(0.0, min(1.0, float(x) / w))
            ny = max(0.0, min(1.0, float(y) / h))
            norm_pts.append(f"{nx:.6f} {ny:.6f}")

        # 5. Format Label YOLO (Class 0)
        label_line = f"0 {' '.join(norm_pts)}\n"

        # 6. Simpan ke .txt
        base_name = os.path.splitext(dst_name)[0]
        label_path = os.path.join(out_path, 'labels', f"{base_name}.txt")

        with open(label_path, 'w') as f:
            f.write(label_line)

        success_count += 1

    # Simpan CSV referensi
    sampled_df.to_csv(os.path.join(out_path, 'sampled_metadata.csv'), index=False)
    print(f"Berhasil memproses dan generate label untuk {success_count} gambar SmartDoc.")

def order_points(pts):
    """Mengurutkan 4 titik koordinat: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def is_readable_heuristic(img, min_size=150, blur_threshold=200.0, edge_density_threshold=0.03):
    """
    Screener OpenCV untuk memastikan gambar dokumen layak (readable).

    - blur_threshold: 200 (untuk memastikan gambar tidak blur)
    - edge_density_threshold: 0.03 (untuk memastikan kertas tidak kosong/minim teks)
    """
    h, w = img.shape[:2]

    # Filter Ukuran: Terlalu kecil = tidak terbaca
    if h < min_size or w < min_size:
        return False

    aspect_ratio = max(h, w) / min(h, w)
    if aspect_ratio > 3.0:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_brightness = np.mean(gray)
    if mean_brightness < 40 or mean_brightness > 230:
        return False

    # Filter Blur (Laplacian Variance)
    blurred_for_lap = cv2.GaussianBlur(gray, (3, 3), 0)
    lap_var = cv2.Laplacian(blurred_for_lap, cv2.CV_64F).var()
    if lap_var < blur_threshold:
        return False

    # Filter Kertas Kosong (Edge Density)
    edges = cv2.Canny(gray, 75, 150)
    edge_density = np.count_nonzero(edges) / (h * w)

    if edge_density < edge_density_threshold:
        return False

    return True

def sample_and_crop_roboflow_model2(raw_path, out_path, num_samples):
    print(f"Sampling & Cropping Roboflow (Model 2): {num_samples} images...")

    all_images = []
    for split in ['train', 'valid', 'test']:
        imgs = glob.glob(os.path.join(raw_path, split, 'images', '*.*'))
        all_images.extend(imgs)

    sampled = random.sample(all_images, min(num_samples, len(all_images)))

    crop_count = 0
    rejected_count = 0 # Tambahan counter untuk tracking

    for img_path in sampled:
        label_path = img_path.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).rsplit('.', 1)[0] + '.txt'

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5: continue

            coords = list(map(float, parts[1:]))
            pts = np.array(coords).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts = pts.astype(np.float32)

            if len(pts) != 4:
                rect_min = cv2.minAreaRect(pts)
                pts = cv2.boxPoints(rect_min)

            rect_pts = order_points(pts)
            (tl, tr, br, bl) = rect_pts

            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            dst_pts = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
            crop_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

            # Screening
            if is_readable_heuristic(crop_img):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                save_name = f"{base_name}_crop_{idx}.jpg"
                save_path = os.path.join(out_path, save_name)

                cv2.imwrite(save_path, crop_img)
                crop_count += 1
            else:
                rejected_count += 1

    print(f"Berhasil memotong {crop_count} gambar.")
    print(f"Ditolak oleh Screener: {rejected_count} gambar.")

def main():
    # Reproducibility Setup
    random.seed(42)

    RAW_DIR = "../dataset/raw"
    OUT_DIR = "../dataset/sampling"
    setup_directories(OUT_DIR)

    # Sampling
    try:
        # Model 1
        sample_roboflow_model1(f"{RAW_DIR}/roboflow_document_segmentation", f"{OUT_DIR}/model1_yolo/roboflow", num_samples=2000)
        sample_smartdoc2015(f"{RAW_DIR}/SmartDoc-2015", f"{OUT_DIR}/model1_yolo/smartdoc2015", num_samples=2500)
        sample_coco(f"{RAW_DIR}/COCO", f"{OUT_DIR}/model1_yolo/coco_negative", num_samples=600)

        # Model 2
        sample_and_crop_roboflow_model2(f"{RAW_DIR}/roboflow_document_segmentation", f"{OUT_DIR}/model2_clf/base_crops", num_samples=1500)

        print("\nSampling Selesai")
    except Exception as e:
        print(f"Terjadi error saat proses: {e}")

if __name__ == "__main__":
    main()
