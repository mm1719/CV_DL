import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from config import cfg


def extract_wh_from_json(json_path, label):
    with open(json_path) as f:
        data = json.load(f)
    widths, heights, tags = [], [], []
    for img in data["images"]:
        widths.append(img["width"])
        heights.append(img["height"])
        tags.append(label)
    return widths, heights, tags


def extract_wh_from_folder(image_dir, label):
    widths, heights, tags = [], [], []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                tags.append(label)
    return widths, heights, tags


def main():
    # 收集所有寬高
    train_w, train_h, train_tag = extract_wh_from_json(cfg.train_json, "Train")
    val_w, val_h, val_tag = extract_wh_from_json(cfg.valid_json, "Val")
    test_w, test_h, test_tag = extract_wh_from_folder(cfg.test_dir, "Test")

    all_w = train_w + val_w + test_w
    all_h = train_h + val_h + test_h
    all_tag = train_tag + val_tag + test_tag

    # 畫出散佈圖
    colors = {"Train": "skyblue", "Val": "orange", "Test": "green"}
    plt.figure(figsize=(8, 6))
    for tag in ["Train", "Val", "Test"]:
        xs = [w for w, t in zip(all_w, all_tag) if t == tag]
        ys = [h for h, t in zip(all_h, all_tag) if t == tag]
        plt.scatter(xs, ys, label=tag, alpha=0.5, s=10, color=colors[tag])

    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.title("Image Resolution Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("image_resolution_scatter.png")
    print("✅ 散佈圖已儲存為 image_resolution_scatter.png")


if __name__ == "__main__":
    main()
