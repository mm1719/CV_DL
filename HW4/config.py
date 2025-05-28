from datetime import datetime
import os


class Config:
    def __init__(self, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.timestamp = timestamp
        self.output_dir = os.path.join("outputs", self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        # Training hyperparameters
        self.batch_size = 64  # 可以視 GPU 調整為 8 或 32
        self.lr = 1e-4  # 同學也使用這個，搭配 AdamW 效果好
        self.epochs = 400  # 支援較長訓練
        self.crop_size = 64  # 建議保持，不建議用 64
        self.val_ratio = 0.2  # 同學使用 train_ratio=0.8

        # Model
        self.in_channels = 3
        self.out_channels = 3
        self.decoder = True

        # Paths
        self.data_root = "hw4_realse_dataset"
        self.model_path = os.path.join(self.output_dir, "best_model.pt")
        self.output_npz = os.path.join(self.output_dir, "pred.npz")
