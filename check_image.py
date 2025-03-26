import os
from PIL import Image, UnidentifiedImageError


def is_image_corrupt(image_path):
    try:
        with Image.open(image_path) as img:
            img.load()  # 嘗試實際解碼像素內容
        return False
    except (IOError, UnidentifiedImageError, SyntaxError, OSError):
        return True


def check_and_delete_corrupt_images(root_dir):
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    corrupt_list = []

    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(supported_ext):
                fpath = os.path.join(root, fname)
                if is_image_corrupt(fpath):
                    corrupt_list.append(fpath)
                    print(f"[損壞] {fpath}")
                    os.remove(fpath)
                    print(f"[已刪除] {fpath}")

    print("\n✅ 檢查完成，共刪除損壞圖片：", len(corrupt_list))


if __name__ == "__main__":
    check_and_delete_corrupt_images("./hw1-data/")
