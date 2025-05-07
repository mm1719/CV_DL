# engine/tester.py

import json
import zipfile
import cv2
from pathlib import Path
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from config import Config
from dataloader.dataloader import register_cell_dataset
from utils.encode import encode_binary_mask


def _build_cfg(cfg_obj: Config) -> get_cfg:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_obj.yaml))
    cfg.MODEL.WEIGHTS = str(cfg_obj.pretrained_weights)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg_obj.num_classes
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return cfg


def _load_id_map(cfg_obj: Config):
    with open(cfg_obj.data_root / "test_image_name_to_ids.json") as f:
        return {d["file_name"]: d["id"] for d in json.load(f)}


def _get_test_images(cfg_obj: Config):
    return sorted((cfg_obj.data_root / "test_release").glob("*.tif"))


def do_test(cfg_obj: Config):
    register_cell_dataset()
    cfg = _build_cfg(cfg_obj)
    predictor = DefaultPredictor(cfg)
    id_map = _load_id_map(cfg_obj)

    results = []

    for pth in tqdm(_get_test_images(cfg_obj), desc="[Testing]"):
        img = cv2.imread(str(pth))
        h, w = img.shape[:2]
        inst = predictor(img)["instances"].to("cpu")

        for box, mask, cls, score in zip(
            inst.pred_boxes.tensor.numpy(),
            inst.pred_masks.numpy(),
            inst.pred_classes.numpy(),
            inst.scores.numpy(),
        ):
            x1, y1, x2, y2 = box.tolist()
            rle = encode_binary_mask(mask)
            results.append(
                {
                    "image_id": int(id_map[pth.name]),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(score),
                    "category_id": int(cls) + 1,
                    "segmentation": {"size": [int(h), int(w)], "counts": rle["counts"]},
                }
            )

    # 儲存為 test-results.json 並壓縮為 submission.zip
    with open(cfg_obj.result_json, "w") as f:
        json.dump(results, f)

    with zipfile.ZipFile(cfg_obj.result_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(cfg_obj.result_json, arcname="test-results.json")

    print(f"[✓] Result written to: {cfg_obj.result_zip}")
