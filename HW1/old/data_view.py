import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def count_images_in_folders(root_dir):
    counts = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            continue
        class_counts = {
            cls: len(os.listdir(os.path.join(split_dir, cls)))
            for cls in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, cls))  # **確保是資料夾**
        }
        counts[split] = class_counts
    return counts

# 繪製類別分布長條圖
def plot_class_distribution(counts, split_name):
    sorted_classes = sorted(counts.keys(), key=lambda x: int(x))  # 按類別 ID 排序
    num_images = [counts[cls] for cls in sorted_classes]

    plt.figure(figsize=(12, 5))
    plt.bar(sorted_classes, num_images)
    plt.xlabel("Class ID")
    plt.ylabel("Number of Images")
    plt.title(f"Class Distribution in {split_name} Set")
    plt.xticks(rotation=90)
    plt.show()


# 設定數據目錄
data_path = "hw1-data/data"
image_counts = count_images_in_folders(data_path)

# 顯示結果
for split, counts in image_counts.items():
    print(f"\n{split.upper()} SET:")
    for cls, num in sorted(counts.items(), key=lambda x: int(x[0])):
        print(f"Class {cls}: {num} images")

# 為 train, val, test 繪製分布圖
if 'train' in image_counts:
    plot_class_distribution(image_counts['train'], 'train')