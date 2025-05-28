import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torch
import random


class HW4ImageDataset(Dataset):
    def __init__(self, root_dir, mode="train", crop_size=128, shuffle_list=None):
        self.mode = mode
        self.crop_size = crop_size
        self.shuffle_list = shuffle_list

        if self.mode in ["train", "val"]:
            self.degraded_dir = os.path.join(root_dir, "train/degraded")
            self.clean_dir = os.path.join(root_dir, "train/clean")
        else:
            self.degraded_dir = os.path.join(root_dir, "test/degraded")
            self.clean_dir = None

        self.degraded_filenames = sorted(
            [f for f in os.listdir(self.degraded_dir) if f.endswith(".png")]
        )

        if self.shuffle_list is not None:
            if self.mode == "train":
                self.degraded_filenames = [
                    f
                    for f, keep in zip(self.degraded_filenames, self.shuffle_list)
                    if keep
                ]
            elif self.mode == "val":
                self.degraded_filenames = [
                    f
                    for f, keep in zip(self.degraded_filenames, self.shuffle_list)
                    if not keep
                ]

        self.transform = self.get_transform()

    def get_transform(self):
        if self.mode == "train":
            return T.Compose(
                [
                    T.RandomCrop(self.crop_size),
                    T.ToTensor(),
                ]
            )
        else:
            return T.Compose(
                [
                    T.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.degraded_filenames)

    def __getitem__(self, idx):
        seed = np.random.randint(2147483647)
        filename = self.degraded_filenames[idx]
        degraded_path = os.path.join(self.degraded_dir, filename)
        degraded_img = Image.open(degraded_path).convert("RGB")

        if self.mode == "test":
            degraded_tensor = self.transform(degraded_img)
            return filename, degraded_tensor

        prefix = "rain_clean-" if "rain" in filename else "snow_clean-"
        index = filename.split("-")[-1].replace(".png", "")
        clean_filename = f"{prefix}{index}.png"
        clean_path = os.path.join(self.clean_dir, clean_filename)
        clean_img = Image.open(clean_path).convert("RGB")

        random.seed(seed)
        torch.manual_seed(seed)
        degraded_tensor = self.transform(degraded_img)

        random.seed(seed)
        torch.manual_seed(seed)
        clean_tensor = self.transform(clean_img)

        return degraded_tensor, clean_tensor
