import os
import cv2
import glob
import random
import albumentations as A
from tqdm import tqdm

def setup_aug_dirs(base_path):
    dirs = [
        os.path.join(base_path, "0_readable"),
        os.path.join(base_path, "1_unreadable")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs[0], dirs[1]

def preprocess_for_model2(img):
    """
    Simulasi preprocessing yang akan digunakan di endpoint API nanti.
    Grayscale -> CLAHE -> Convert balik ke BGR
    """
    # Jadikan Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplikasikan CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)

    # Kembalikan ke 3 channel (BGR)
    # Meskipun gambarnya hitam putih, format array-nya harus 3 channel.
    final_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

    return final_img

# PIPELINE 1: READABLE (Variasi sangat ringan, tanpa merusak fokus)
readable_pipeline = A.Compose([
    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
    A.Affine(
        p=0.3,
        scale=(0.98, 1.02),
        translate_percent=(-0.02, 0.02),
        rotate=(-2, 2)
    ),
])

# PIPELINE 2: UNREADABLE
unreadable_pipeline = A.Compose([
    A.SomeOf([
        A.MotionBlur(p=1, blur_limit=(7, 15)),          # Simulasi blur kamera bergerak
        A.GaussianBlur(p=1, blur_limit=(7, 15)),        # Simulasi kamera out-of-focus
        A.GaussNoise(p=1, var_limit=(30, 80)),           # Simulasi noise kamera low end
        A.ISONoise(p=1, intensity=(0.3, 0.6)),           # Simulasi ambil gambar dengan ISO jelek
        A.Downscale(p=1, scale_min=0.3, scale_max=0.5,  # Simulasi kamera dengan resolusi rendah
                    interpolation=cv2.INTER_LINEAR),
    ], n=2, p=1.0),  # n=2 untuk kombinasi 2 degradasi

    A.ImageCompression(p=0.5, quality_lower=20, quality_upper=50),  # Simulasi kompresi gambar
    A.RandomShadow(p=0.3),  # Simulasi shadow pada gambar
])

def run_augmentation_and_preprocess(input_dir, output_base):
    random.seed(42)
    read_dir, unread_dir = setup_aug_dirs(output_base)

    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    print(f"Memproses {len(image_paths)} gambar (Augmentasi -> Preprocess CLAHE)...")

    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        if img is None: continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 1. KELAS READABLE (Label 0)
        # Original tapi di Preprocess CLAHE
        clean_clahe = preprocess_for_model2(img)
        cv2.imwrite(os.path.join(read_dir, f"{base_name}_clean.jpg"), clean_clahe)

        # Variasi ringan + Preprocess CLAHE (1 variasi)
        aug_clean = readable_pipeline(image=img)["image"]
        aug_clean_clahe = preprocess_for_model2(aug_clean)
        cv2.imwrite(os.path.join(read_dir, f"{base_name}_var.jpg"), aug_clean_clahe)

        # 2. KELAS UNREADABLE (Label 1)
        # Bikin 2 variasi rusak untuk menyeimbangkan data
        for i in range(2):
            # Tahap A: Rusak gambarnya (Simulasi kamera jelek)
            bad_aug = unreadable_pipeline(image=img)["image"]

            # Tahap B: Preprocess (Sistem memproses gambar jelek tersebut)
            bad_clahe = preprocess_for_model2(bad_aug)

            # Simpan
            cv2.imwrite(os.path.join(unread_dir, f"{base_name}_bad_{i}.jpg"), bad_clahe)

    print(f"Kelas 0_readable: {len(os.listdir(read_dir))} images")
    print(f"Kelas 1_unreadable: {len(os.listdir(unread_dir))} images")

if __name__ == "__main__":
    BASE_CROP_DIR = "./dataset/sampling/model2_clf/base_crops"
    AUG_OUTPUT_DIR = "./dataset/sampling/model2_clf/augmented"

    if os.path.exists(BASE_CROP_DIR) and len(os.listdir(BASE_CROP_DIR)) > 0:
        run_augmentation_and_preprocess(BASE_CROP_DIR, AUG_OUTPUT_DIR)
    else:
        print("Error: Folder base_crops kosong.")
