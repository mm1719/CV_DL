import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os


def compute_iou(box1, box2):
    """
    box: [x_min, y_min, x_max, y_max]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(x2 - x1, 0) * max(y2 - y1, 0)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


def evaluate_predictions(preds, gts, iou_threshold=0.5):
    """
    評估預測結果的準確率、召回率和 F1 分數。
    Args:
        preds: 預測結果，格式為 [{"image_id": int, "bbox": [x, y, w, h], "category_id": int}, ...]
        gts: 標註結果，格式為 [{"image_id": int, "bbox": [x, y, w, h], "category_id": int}, ...]
        iou_threshold: IoU 閾值
    Returns:
        dict: 包含 precision、recall 和 f1 的評估結果
    """
    tp = 0
    fp = 0
    fn = 0

    # 建立 GT 對應表
    from collections import defaultdict

    gt_dict = defaultdict(list)
    for gt in gts:
        gt_dict[gt["image_id"]].append(gt)

    for pred in preds:
        pred_box = xywh_to_xyxy(pred["bbox"])
        pred_img = pred["image_id"]
        pred_cat = pred["category_id"]

        matched = False
        for gt in gt_dict[pred_img]:
            if gt["category_id"] != pred_cat:
                continue
            iou = compute_iou(pred_box, xywh_to_xyxy(gt["bbox"]))
            if iou >= iou_threshold:
                matched = True
                gt_dict[pred_img].remove(gt)  # 避免重複配對
                break

        if matched:
            tp += 1
        else:
            fp += 1

    # 剩下沒被配對到的 GT 都是 FN
    fn = sum(len(v) for v in gt_dict.values())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def coco_evaluation(pred_json_path, gt_json_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # ➕ 只取 accuracy
    with open(pred_json_path) as f:
        preds = json.load(f)
    gts = []
    for ann in coco_gt.dataset["annotations"]:
        gts.append(
            {
                "image_id": ann["image_id"],
                "bbox": ann["bbox"],
                "category_id": ann["category_id"],
            }
        )
    acc_result = evaluate_predictions(preds, gts)
    accuracy = acc_result["precision"]  # 因為 accuracy 就是 precision 的簡化

    return {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
        "AR_1": coco_eval.stats[6],
        "AR_10": coco_eval.stats[7],
        "AR_100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
        "ACC": accuracy,
    }
