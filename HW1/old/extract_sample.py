import os
import shutil
import random


def extract_val_samples(val_dir, output_dir):
    """從 val 目錄的每個類別中隨機複製一張圖片到新的資料夾"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cls in sorted(
        os.listdir(val_dir), key=lambda x: int(x)
    ):  # 確保類別是 0~99 的順序
        cls_path = os.path.join(val_dir, cls)
        if not os.path.isdir(cls_path):
            continue  # 確保是資料夾

        images = [
            f
            for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(images) == 0:
            print(f"⚠️ 類別 {cls} 沒有圖片，跳過")
            continue

        selected_img = random.choice(images)  # 隨機選一張
        src_path = os.path.join(cls_path, selected_img)
        dst_path = os.path.join(output_dir, f"{cls}_{selected_img}")  # 確保檔名不重複

        shutil.copy(src_path, dst_path)
        print(f"✅ 已複製 {src_path} 到 {dst_path}")


# 設定資料夾
val_path = "hw1-data/data/val"
output_path = "sampled_val"

extract_val_samples(val_path, output_path)
