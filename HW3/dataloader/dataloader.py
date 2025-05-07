# dataloader/dataset.py

import random
import re
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from pycocotools import mask as mask_utils

from config import Config

cfg_obj = Config()


def _annos_from_mask(mask_path: Path, cat_id: int) -> List[Dict]:
    """
    Convert a mask image to COCO-style annotations.
    Args:
        mask_path (Path): Path to the mask image.
        cat_id (int): Category ID for the mask.
    Returns:
        List[Dict]: List of annotations in COCO format.
    """

    img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if img.ndim > 2:
        img = img[..., 0]
    annos = []
    for inst_id in np.unique(img):
        if inst_id == 0:
            continue
        bin_mask = (img == inst_id).astype(np.uint8).copy()
        ys, xs = np.where(bin_mask)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        rle = mask_utils.encode(np.asfortranarray(bin_mask))
        rle["counts"] = rle["counts"].decode("ascii")
        annos.append(
            {
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": rle,
                "category_id": cat_id,
                "iscrowd": 0,
            }
        )
    return annos


def _load_sample(dir_path: Path, img_id: int) -> Dict:
    """
    Load a sample from the given directory.
    Args:
        dir_path (Path): Path to the sample directory.
        img_id (int): Image ID for the sample.
    Returns:
        Dict: Dictionary containing image and annotation information.
    """
    img_path = dir_path / "image.tif"
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    record = {
        "file_name": str(img_path),
        "image_id": img_id,
        "height": h,
        "width": w,
        "annotations": [],
    }
    for mask_path in dir_path.glob("class*.tif"):
        m = re.search(r"class(\d+)\.tif", mask_path.name)
        if not m:
            continue
        class_idx = int(m.group(1))
        if class_idx > cfg_obj.num_classes:
            continue
        record["annotations"] += _annos_from_mask(mask_path, class_idx - 1)
    return record


def _build_dicts(split: str) -> List[Dict]:
    """
    Build a list of dictionaries for the dataset.
    Args:
        split (str): Split type (train, val, trainval).
    Returns:
        List[Dict]: List of dictionaries containing image and annotation information.
    """
    dirs = sorted((cfg_obj.data_root / "train").iterdir())
    rnd = random.Random(cfg_obj.random_seed)
    rnd.shuffle(dirs)
    if split == "trainval":
        chosen = dirs
    else:
        cut = int(len(dirs) * (1 - cfg_obj.val_ratio))
        chosen = dirs[:cut] if split == "train" else dirs[cut:]
    return [_load_sample(d, i) for i, d in enumerate(chosen)]


def _register(name: str, split: str):
    """
    Register the dataset with Detectron2.
    Args:
        name (str): Name of the dataset.
        split (str): Split type (train, val, trainval).
    """

    if name in DatasetCatalog.list():
        return
    DatasetCatalog.register(name, lambda s=split: _build_dicts(s))
    MetadataCatalog.get(name).set(
        thing_classes=[
            cfg_obj.class_name_map[i + 1] for i in range(cfg_obj.num_classes)
        ],
        mask_format="bitmask",
    )


def register_cell_dataset():
    """
    Register the cell dataset with Detectron2.
    """
    _register("cells_train", "train")
    _register("cells_val", "val")
    _register("cells_trainval", "trainval")
