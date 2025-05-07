# config.py

import os
import pathlib
from datetime import datetime


class Config:
    def __init__(self, mode="train", timestamp=None, variant="default"):
        self.mode = mode
        self.variant = variant  # <- 加上這一行
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")

        # 根目錄
        self.root = pathlib.Path(__file__).resolve().parent

        # 資料集根目錄
        self.data_root = self.root / "hw3-data"

        # 模型結構選擇
        if self.variant == "weakly":
            self.meta_arch = "WeaklySupervisedMaskRCNN"
            self.yaml = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        else:
            self.meta_arch = "GeneralizedRCNN"
            self.yaml = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"

        self.pretrained_weights = None  # 若測試時要指定，可以在 main 傳進來

        # 分類設定
        self.num_classes = 4
        self.class_name_map = {i: f"class{i}" for i in range(1, self.num_classes + 1)}

        # 分割訓練與驗證集的比例
        self.val_ratio = 0.2
        self.random_seed = 42

        # 輸出相關路徑
        self.output_dir = self.root / "outputs" / self.timestamp
        self.log_path = self.output_dir / "log.txt"
        self.model_path = self.output_dir / "best_model.pth"
        self.result_json = self.output_dir / "test-results.json"
        self.result_zip = self.output_dir / "submission.zip"

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        return f"[Config] Mode: {self.mode}, Variant: {self.variant}, Timestamp: {self.timestamp}, Output: {self.output_dir}"
