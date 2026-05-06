import os
import glob
import random
import shutil
import yaml
from tqdm import tqdm

def get_exact_counts(total_items, ratios):
    """
    Largest Remainder Method.
    Memastikan dataset terbagi dengan persentase presisi tanpa kekurangan/kelebihan.
    """
    assert round(sum(ratios), 5) == 1.0, "Total rasio harus 1.0"

    exact_floats = [total_items * r for r in ratios]
    counts = [int(e) for e in exact_floats]

    remainders = [(i, exact_floats[i] - counts[i]) for i in range(len(ratios))]
    deficit = total_items - sum(counts)

    remainders.sort(key=lambda x: x[1], reverse=True)

    for i in range(deficit):
        idx = remainders[i][0]
        counts[idx] += 1

    return counts

def generate_yaml(yaml_path, yaml_data):
    """Menyimpan konfigurasi dictionary menjadi file .yaml"""
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

# Model 1
def split_yolo_dataset(input_base, out_base, ratios):
    print("\n" + "="*40)
    print("SPLIT MODEL 1 (YOLO OBB)")
    print("="*40)

    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(out_base, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_base, "labels", split), exist_ok=True)

    sources = {
        "Roboflow": {"img": os.path.join(input_base, "roboflow", "images"), "lbl": os.path.join(input_base, "roboflow", "labels"), "type": "normal"},

        # PERUBAHAN DI SINI: SmartDoc sekarang bertipe "normal" dan path-nya spesifik ke images/labels
        "SmartDoc": {"img": os.path.join(input_base, "smartdoc2015", "images"), "lbl": os.path.join(input_base, "smartdoc2015", "labels"), "type": "normal"},

        "COCO": {"img": os.path.join(input_base, "coco_negative"), "lbl": None, "type": "background"}
    }

    total_yolo = 0
    for src_name, paths in sources.items():
        if not os.path.exists(paths["img"]):
            print(f"Skipping {src_name} (Folder tidak ditemukan)")
            continue

        images = glob.glob(os.path.join(paths["img"], "*.*"))
        # Filter hanya format gambar
        images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        total_src = len(images)
        if total_src == 0: continue

        random.shuffle(images)
        counts = get_exact_counts(total_src, ratios)
        print(f"\n[{src_name}] Total: {total_src} -> Train: {counts[0]} | Val: {counts[1]} | Test: {counts[2]}")

        tasks = [("train", images[:counts[0]]),
                 ("val", images[counts[0]:counts[0]+counts[1]]),
                 ("test", images[counts[0]+counts[1]:])]

        for split_name, img_list in tasks:
            for img_path in tqdm(img_list, desc=f"Copying {src_name} -> {split_name}", leave=False):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                img_dst = os.path.join(out_base, "images", split_name, os.path.basename(img_path))
                lbl_dst = os.path.join(out_base, "labels", split_name, f"{base_name}.txt")

                # Copy Gambar
                shutil.copy(img_path, img_dst)

                # Copy / Generate Label
                if paths["type"] == "normal" and paths["lbl"]:
                    lbl_src = os.path.join(paths["lbl"], f"{base_name}.txt")
                    if os.path.exists(lbl_src):
                        shutil.copy(lbl_src, lbl_dst)
                elif paths["type"] == "background":
                    # COCO Negative: Buat file .txt kosong agar YOLO paham ini background
                    open(lbl_dst, 'w').close()
                elif paths["type"] == "image_only":
                    pass

        total_yolo += total_src

    # Copy CSV SmartDoc jika ada agar referensinya tidak hilang
    csv_src = os.path.join(input_base, "smartdoc2015", "sampled_metadata.csv")
    if os.path.exists(csv_src):
        shutil.copy(csv_src, os.path.join(out_base, "smartdoc_metadata.csv"))

    # Generate YOLO YAML
    yolo_yaml = {
        "path": os.path.abspath(out_base),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "document"}
    }
    generate_yaml(os.path.join(out_base, "model1_yolo.yaml"), yolo_yaml)
    print(f"\n-> YOLO Split Selesai! ({total_yolo} gambar)")

# Model 2
def split_clf_dataset(input_base, out_base, ratios):
    print("\n" + "="*40)
    print("SPLIT MODEL 2 (CLASSIFICATION)")
    print("="*40)

    splits = ["train", "val", "test"]
    classes = [d for d in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, d))]
    classes.sort()

    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(out_base, split, cls), exist_ok=True)

    total_clf = 0
    for cls in classes:
        images = glob.glob(os.path.join(input_base, cls, "*.jpg"))
        total_cls = len(images)
        if total_cls == 0: continue

        random.shuffle(images)
        counts = get_exact_counts(total_cls, ratios)
        print(f"\n[Kelas '{cls}'] Total: {total_cls} -> Train: {counts[0]} | Val: {counts[1]} | Test: {counts[2]}")

        tasks = [("train", images[:counts[0]]),
                 ("val", images[counts[0]:counts[0]+counts[1]]),
                 ("test", images[counts[0]+counts[1]:])]

        for split_name, img_list in tasks:
            for img_path in tqdm(img_list, desc=f"Copying {cls} -> {split_name}", leave=False):
                dst = os.path.join(out_base, split_name, cls, os.path.basename(img_path))
                shutil.copy(img_path, dst)

        total_clf += total_cls

    # Generate CLF YAML
    clf_yaml = {
        "path": os.path.abspath(out_base),
        "train": "train",
        "val": "val",
        "test": "test",
        "nc": len(classes),
        "names": {i: cls_name for i, cls_name in enumerate(classes)}
    }
    generate_yaml(os.path.join(out_base, "model2_clf.yaml"), clf_yaml)
    print(f"\n-> Classification Split Selesai! ({total_clf} gambar)")

# MAIN
if __name__ == "__main__":
    # Reproducibility Setup
    random.seed(42)

    BASE_SAMPLED_DIR = "./dataset/sampling"
    YOLO_RAW_DIR = os.path.join(BASE_SAMPLED_DIR, "model1_yolo")
    CLF_RAW_DIR = os.path.join(BASE_SAMPLED_DIR, "model2_clf", "augmented")

    FINAL_DIR = "./dataset/final_split"
    YOLO_OUT_DIR = os.path.join(FINAL_DIR, "model1_yolo")
    CLF_OUT_DIR = os.path.join(FINAL_DIR, "model2_clf")

    # Rasio Splitting
    RATIOS = (0.70, 0.15, 0.15)

    # Eksekusi
    if os.path.exists(YOLO_RAW_DIR):
        split_yolo_dataset(YOLO_RAW_DIR, YOLO_OUT_DIR, RATIOS)
    else:
        print(f"Error: Folder input YOLO tidak ditemukan di {YOLO_RAW_DIR}")

    if os.path.exists(CLF_RAW_DIR):
        split_clf_dataset(CLF_RAW_DIR, CLF_OUT_DIR, RATIOS)
    else:
        print(f"Error: Folder input Classification tidak ditemukan di {CLF_RAW_DIR}")

    print("\n" + "#"*50)
    print(f"Folder Output Final: {os.path.abspath(FINAL_DIR)}")
    print("#"*50)