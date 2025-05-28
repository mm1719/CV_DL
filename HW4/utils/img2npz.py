import os
import numpy as np
from PIL import Image


def convert_folder_to_npz(folder_path, output_npz="pred.npz"):
    """
    將資料夾內所有圖片轉成 (3, H, W) 的 dict，儲存為 .npz 檔。
    """
    images_dict = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(folder_path, filename)
            image = Image.open(file_path).convert("RGB")
            img_array = np.array(image)
            img_array = np.transpose(img_array, (2, 0, 1))  # (H, W, 3) -> (3, H, W)
            images_dict[filename] = img_array

    np.savez(output_npz, **images_dict)
    print(f"✅ Saved {len(images_dict)} images to {output_npz}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="要轉換的資料夾路徑"
    )
    parser.add_argument(
        "--output_npz", type=str, default="pred.npz", help="輸出的 .npz 檔名"
    )
    args = parser.parse_args()

    convert_folder_to_npz(args.input_dir, args.output_npz)
