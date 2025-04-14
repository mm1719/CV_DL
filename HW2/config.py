import os
import torch
from datetime import datetime


class Config:
    def __init__(self):
        # === 資料路徑設定 ===
        self.data_root = "nycu-hw2-data"
        self.train_dir = os.path.join(self.data_root, "train")
        self.valid_dir = os.path.join(self.data_root, "valid")
        self.test_dir = os.path.join(self.data_root, "test")
        self.train_json = os.path.join(self.data_root, "train.json")
        self.valid_json = os.path.join(self.data_root, "valid.json")

        # === 模型選擇與訓練變體（會自動選擇超參數與 Optimizer/Scheduler） ===
        self.backbone = "resnet50"  # "resnet50" 或 "swin"
        self.variant = "adamw_step"  # 可選："adamw_step", "adamw_cosine"
        self.swin_model_name = "swinv2_base_window8_256"

        # === 類別與硬體設定 ===
        self.num_classes = 11  # 數字 0~9 加上背景
        self.num_workers = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # === 各種變體對應的超參數 ===
        self.variants = {
            "adamw_step": {
                "batch_size": 4,
                "lr": 0.005,
                "lr_step_size": 10,
                "lr_gamma": 0.1,
                "weight_decay": 0.0005,
                "num_epochs": 30,
                "patience": 6,
                "optimizer": "SGD",
                "scheduler": "StepLR",
            },
            "adamw_cosine": {
                "batch_size": 4,
                "lr": 0.005,  # ⭐️ 調高學習率
                "lr_step_size": 6,  # 保留（但 cosine 不會用到）
                "lr_gamma": 0.5,
                "weight_decay": 0.0005,  # ⭐️ 降低 weight decay
                "num_epochs": 30,
                "patience": 6,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
            },
        }

        # === 使用當前 variant 的參數 ===
        hp = self.variants[self.variant]
        self.batch_size = hp["batch_size"]
        self.lr = hp["lr"]
        self.lr_step_size = hp["lr_step_size"]
        self.lr_gamma = hp["lr_gamma"]
        self.weight_decay = hp["weight_decay"]
        self.num_epochs = hp["num_epochs"]
        self.early_stopping_patience = hp["patience"]
        self.optimizer_name = hp["optimizer"]
        self.scheduler_name = hp["scheduler"]

        # === 執行階段產生的路徑（main.py 裡會設定） ===
        self.output_dir = None
        self.checkpoint_path = None
        self.tensorboard_logdir = None
        self.log_file = None


# 全域設定實例
cfg = Config()
