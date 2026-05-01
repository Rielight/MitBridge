import os
import glob

def fix_yolo_labels(labels_base_dir):
    print("Mengecek dan memperbaiki Class ID serta Koordinat Out-of-Bounds...")

    # Ambil semua file txt di dalam folder train, val, test
    txt_files = glob.glob(os.path.join(labels_base_dir, "**", "*.txt"), recursive=True)

    modified_count = 0
    for txt_path in txt_files:
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        needs_modification = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                # 1. Paksa Class ID menjadi 0
                if parts[0] != '0':
                    parts[0] = '0'
                    needs_modification = True

                new_parts = [parts[0]]

                # 2. Clamping koordinat: Jika > 1.0 jadi 1.0, jika < 0.0 jadi 0.0
                for val_str in parts[1:]:
                    val = float(val_str)

                    if val < 0.0:
                        val = 0.0
                        needs_modification = True
                    elif val > 1.0:
                        val = 1.0
                        needs_modification = True

                    # Simpan kembali dengan 6 angka di belakang koma
                    new_parts.append(f"{val:.6f}")

                # Gabungkan kembali menjadi satu baris
                new_lines.append(" ".join(new_parts) + "\n")
            else:
                new_lines.append(line)

        # Tulis ulang file jika ada perubahan (Class ID salah atau Out-of-Bounds)
        if needs_modification:
            with open(txt_path, 'w') as f:
                f.writelines(new_lines)
            modified_count += 1

    print(f"✅ Selesai! Berhasil memperbaiki (Class & Clamping) pada {modified_count} file label.")

if __name__ == "__main__":
    FINAL_LABELS_DIR = "./dataset/final_split/model1_yolo/labels"

    if os.path.exists(FINAL_LABELS_DIR):
        fix_yolo_labels(FINAL_LABELS_DIR)
    else:
        print(f"Folder tidak ditemukan: {FINAL_LABELS_DIR}")