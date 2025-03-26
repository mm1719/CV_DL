import os
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets


class TwoCropTransform:
    """
    給定一個 base_transform，回傳兩個隨機增強版本（for SupCon）
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


class MultiCropTransform:
    """
    給定 global_crop 與 local_crop 的 transform，以及各自張數，回傳多張增強圖片（for Multi-Crop SupCon）
    """

    def __init__(
        self, global_transform, local_transform, num_global_crops=2, num_local_crops=4
    ):
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops

    def __call__(self, x):
        crops = [self.global_transform(x) for _ in range(self.num_global_crops)]
        crops += [self.local_transform(x) for _ in range(self.num_local_crops)]
        return crops


class ImageFolderWithPath(datasets.ImageFolder):
    """
    ImageFolder 的子類別，在 __getitem__ 額外回傳圖片路徑
    用於 test 階段記錄 image_id
    """

    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)  # (image, label)
        path, _ = self.samples[index]
        return original_tuple[0], path  # image tensor, path string


def get_transform(cfg, mode="train", two_crop=False, multi_crop=False):
    # 通用 augment 定義
    global_transform = transforms.Compose(
        [
            transforms.Resize(cfg["resize_size"]),
            transforms.RandomResizedCrop(cfg["image_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ]
    )
    local_transform = transforms.Compose(
        [
            transforms.Resize(cfg["resize_size"]),
            transforms.RandomResizedCrop(cfg["local_crop_size"], scale=(0.05, 0.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ]
    )
    base_val_test = transforms.Compose(
        [
            transforms.Resize(cfg["resize_size"]),
            transforms.CenterCrop(cfg["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ]
    )

    if mode == "train":
        if multi_crop:
            return MultiCropTransform(
                global_transform,
                local_transform,
                cfg.get("num_global_crops", 2),
                cfg.get("num_local_crops", 4),
            )
        elif two_crop:
            return TwoCropTransform(global_transform)
        else:
            return global_transform
    else:
        return base_val_test


def get_tta_transform(cfg):
    return transforms.Compose(
        [
            transforms.Resize(cfg["resize_size"]),
            transforms.CenterCrop(cfg["resize_size"]),
            transforms.RandomCrop(cfg["image_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ]
    )


def get_datasets(cfg, mode="supcon"):
    data_dir = cfg["data_root"]
    multi_crop = cfg.get("use_multi_crop", False)

    train_transform = get_transform(
        cfg,
        mode="train",
        two_crop=(mode == "supcon" and not multi_crop),
        multi_crop=(mode == "supcon" and multi_crop),
    )
    val_transform = get_transform(cfg, mode="val")
    test_transform = get_transform(cfg, mode="test")

    train_set = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"), transform=train_transform
    )
    val_set = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"), transform=val_transform
    )
    test_set = ImageFolderWithPath(
        root=os.path.join(data_dir, "test"), transform=test_transform
    )

    if mode == "test":
        return None, val_set, test_set
    elif mode == "tsne_plt":
        return train_set, val_set, test_set
    else:
        return train_set, val_set, test_set


def compute_class_weights(dataset):
    """根據 dataset 中的類別分佈計算權重"""
    targets = [label for _, label in dataset]
    counter = Counter(targets)
    total = sum(counter.values())
    weights = [total / counter[i] for i in range(len(counter))]
    return torch.tensor(weights, dtype=torch.float)
