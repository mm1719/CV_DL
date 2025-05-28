import argparse
import os
import torch
from sklearn.model_selection import train_test_split

from config import Config
from model.promptir import PromptIR
from dataloader.hw4_dataset import HW4ImageDataset
from engine.trainer import train
from engine.tester import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument(
        "--timestamp", type=str, default=None, help="指定輸出資料夾 timestamp（測試用）"
    )
    args = parser.parse_args()

    config = Config(timestamp=args.timestamp)

    if args.mode == "train":
        print("🚀 進行訓練...")
        full_dataset = HW4ImageDataset(
            config.data_root, mode="train", crop_size=config.crop_size
        )
        indices = list(range(len(full_dataset)))
        train_idx, val_idx = train_test_split(
            indices, test_size=config.val_ratio, random_state=42
        )

        train_set = HW4ImageDataset(
            config.data_root,
            mode="train",
            indices=train_idx,
            crop_size=config.crop_size,
        )
        val_set = HW4ImageDataset(
            config.data_root, mode="val", indices=val_idx, crop_size=config.crop_size
        )

        model = PromptIR(decoder=config.decoder)
        train(model, train_set, val_set, config)

    elif args.mode == "test":
        print(f"🔍 進行測試預測於 timestamp: {config.timestamp}")
        test_set = HW4ImageDataset(config.data_root, mode="test")
        model = PromptIR(decoder=config.decoder)
        predict(model, test_set, config)


if __name__ == "__main__":
    main()
