import os
from PIL import Image
from collections import Counter


def analyze_folder(folder_path):
    print(f"\n📂 分析資料夾: {folder_path}")
    sizes = []
    error_files = []
    image_count = 0

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(".png"):
            continue
        full_path = os.path.join(folder_path, fname)
        try:
            with Image.open(full_path) as img:
                img = img.convert("RGB")
                sizes.append(img.size)
                image_count += 1
        except Exception as e:
            print(f"  ⚠️ 錯誤讀取 {fname}: {e}")
            error_files.append(fname)

    print(f"✅ 成功讀取 {image_count} 張圖片")
    if error_files:
        print(f"❌ 有 {len(error_files)} 張讀取失敗：{error_files}")

    size_count = Counter(sizes)
    print("📊 圖片尺寸統計（W×H）:")
    for size, count in size_count.items():
        print(f"  {size}: {count} 張")
    return sizes


def main(root_dir):
    train_clean = os.path.join(root_dir, "train", "clean")
    train_degraded = os.path.join(root_dir, "train", "degraded")
    test_degraded = os.path.join(root_dir, "test", "degraded")

    analyze_folder(train_clean)
    analyze_folder(train_degraded)
    analyze_folder(test_degraded)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./hw4_data", help="根目錄")
    args = parser.parse_args()

    main(args.root)
