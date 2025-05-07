# dataloader/coco_mil_dataset.py

import re
from pathlib import Path
from typing import List, Dict
import numpy as np
import cv2
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from config import Config

cfg_obj = Config()


def _load_mil_sample(dir_path: Path, img_id: int) -> Dict:
    img_path = dir_path / "image.tif"
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    record = {
        "file_name": str(img_path),
        "image_id": img_id,
        "height": h,
        "width": w,
        "annotations": [],
        "label_vec": [0] * cfg_obj.num_classes,
    }

    for mask_path in dir_path.glob("class*.tif"):
        m = re.search(r"class(\d+)\.tif", mask_path.name)
        if not m:
            continue
        class_idx = int(m.group(1))
        if class_idx > cfg_obj.num_classes:
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask.ndim > 2:
            mask = mask[..., 0]
        for inst_id in np.unique(mask):
            if inst_id == 0:
                continue
            bin_mask = (mask == inst_id).astype(np.uint8)
            ys, xs = np.where(bin_mask)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            record["annotations"].append(
                {
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": class_idx - 1,
                    "iscrowd": 0,
                }
            )
            record["label_vec"][class_idx - 1] = 1
    return record


def _build_mil_dicts(split: str) -> List[Dict]:
    dirs = sorted((cfg_obj.data_root / "train").iterdir())
    rng = np.random.RandomState(cfg_obj.random_seed)
    rng.shuffle(dirs)
    if split == "train":
        cut = int(len(dirs) * (1 - cfg_obj.val_ratio))
        chosen = dirs[:cut]
    else:
        cut = int(len(dirs) * (1 - cfg_obj.val_ratio))
        chosen = dirs[cut:]
    return [_load_mil_sample(d, i) for i, d in enumerate(chosen)]


def register_mil_dataset():
    _register("cells_train_mil", "train")
    _register("cells_val_mil", "val")


def _register(name: str, split: str):
    if name in DatasetCatalog.list():
        return
    DatasetCatalog.register(name, lambda s=split: _build_mil_dicts(s))
    MetadataCatalog.get(name).set(
        thing_classes=[
            cfg_obj.class_name_map[i + 1] for i in range(cfg_obj.num_classes)
        ],
        mask_format="bitmask",
    )
