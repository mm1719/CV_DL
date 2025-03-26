import os
from PIL import Image
import matplotlib.pyplot as plt

data_root = "hw1-data/data/train"

# 收集圖片尺寸
image_sizes = []
for root, dirs, files in os.walk(data_root):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    image_sizes.append((width, height))
            except Exception as e:
                print(f"無法開啟圖片: {path}，錯誤: {e}")
                continue

# 若沒找到任何圖檔，提醒用戶
if len(image_sizes) == 0:
    print(f"在資料夾 {data_root} 中未找到圖片，請檢查路徑是否正確！")
    exit()

# 拆分寬與高
widths, heights = zip(*image_sizes)

# -------- Plot 1: 散佈圖 (width vs height) --------
plt.figure(figsize=(6, 6))
plt.scatter(widths, heights, alpha=0.3, s=5)
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Image Size Distribution (W vs H)")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_wh.png")
plt.show()

# -------- Plot 2: 寬度直方圖 --------
plt.figure(figsize=(6, 4))
plt.hist(widths, bins=30, color="steelblue", edgecolor="black")
plt.xlabel("Width")
plt.ylabel("Count")
plt.title("Histogram of Image Widths")
plt.grid(True)
plt.tight_layout()
plt.savefig("hist_width.png")
plt.show()

# -------- Plot 3: 高度直方圖 --------
plt.figure(figsize=(6, 4))
plt.hist(heights, bins=30, color="orange", edgecolor="black")
plt.xlabel("Height")
plt.ylabel("Count")
plt.title("Histogram of Image Heights")
plt.grid(True)
plt.tight_layout()
plt.savefig("hist_height.png")
plt.show()

# -------- Plot 4: 2D Hexbin Heatmap --------
plt.figure(figsize=(6, 6))
plt.hexbin(widths, heights, gridsize=40, cmap="viridis")
plt.colorbar(label="Image Count")
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("2D Heatmap of Image Size Density")
plt.grid(True)
plt.tight_layout()
plt.savefig("heatmap_wh.png")
plt.show()
