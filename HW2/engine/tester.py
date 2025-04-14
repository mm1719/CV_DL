import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from config import cfg
from torchvision.transforms import functional as F
import pandas as pd
from collections import defaultdict
from utils.image_transform import (
    ResizeWithPaddingAndBox,
    restore_boxes,
    ResizeWithPaddingForTraining,
    AugmentWithoutResize,
)


def load_test_images(test_dir):
    files = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])
    return [(int(os.path.splitext(f)[0]), os.path.join(test_dir, f)) for f in files]


@torch.no_grad()
def run_inference(
    model,
    test_dir,
    save_json_path="pred.json",
    save_csv_path="pred.csv",
    score_thresh=0.5,
):
    model.eval()
    model.to(cfg.device)

    test_images = load_test_images(test_dir)
    results = []

    print(f"📦 共偵測 {len(test_images)} 張圖片")
    for image_id, path in tqdm(test_images, desc="[Test]"):
        image = Image.open(path).convert("RGB")

        # ✅ 根據 backbone 自動選擇 transform
        if cfg.backbone == "resnet50":
            transform = AugmentWithoutResize()
        else:
            transform = ResizeWithPaddingAndBox(target_size=256)

        image_tensor, dummy_target = transform(image, {"boxes": torch.zeros((0, 4))})
        resize_info = dummy_target.get("resize_info", None)

        image_tensor = image_tensor.to(cfg.device)
        outputs = model([image_tensor])[0]

        boxes = outputs["boxes"].cpu()
        if resize_info is not None:
            boxes = restore_boxes(boxes, resize_info)  # ✅ 還原 bbox
        labels = outputs["labels"].cpu()
        scores = outputs["scores"].cpu()

        for box, label, score in zip(boxes, labels, scores):
            if score < score_thresh:
                continue

            x_min, y_min, x_max, y_max = box.tolist()
            x = x_min
            y = y_min
            w = x_max - x_min
            h = y_max - y_min

            result = {
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x, y, w, h],
                "score": float(score),
            }
            results.append(result)

    # 儲存 pred.json
    with open(save_json_path, "w") as f:
        json.dump(results, f)
    print(f"[完成] 預測結果存為: {save_json_path}")

    # === 轉換成 pred.csv 格式（image_id, pred_label）===
    image_to_preds = defaultdict(list)
    for pred in results:
        image_to_preds[pred["image_id"]].append(pred)

    pred_label_rows = []
    all_ids = [i for i, _ in test_images]
    for image_id in all_ids:
        preds = image_to_preds.get(image_id, [])
        if len(preds) == 0:
            pred_label = -1
        else:
            preds = sorted(preds, key=lambda x: x["bbox"][0])  # 按照 x 座標排序
            digits = [
                str(int(p["category_id"]) - 1) for p in preds
            ]  # category_id 是 1-indexed
            pred_label = int("".join(digits)) if digits else -1

        pred_label_rows.append({"image_id": image_id, "pred_label": pred_label})

    # 儲存 pred.csv
    df = pd.DataFrame(sorted(pred_label_rows, key=lambda x: x["image_id"]))
    df.to_csv(save_csv_path, index=False)
    print(f"[完成] 整體數字預測存為: {save_csv_path}")
