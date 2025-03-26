import os
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset
from config import config
from PIL import Image
import torch


class MultiAugDataset(Dataset):
    """
    Multiple augmentations dataset.
    """

    def __init__(self, root, transform, n_views=1):
        self.base_dataset = datasets.ImageFolder(root)
        self.transform = transform
        self.n_views = n_views

    def __getitem__(self, index):
        path, label = self.base_dataset.samples[index]
        image = Image.open(path).convert("RGB")
        views = [self.transform(image) for _ in range(self.n_views)]
        return views, label

    def __len__(self):
        return len(self.base_dataset)


class TestImageDataset(Dataset):
    """
    Dataset for test images.
    """

    def __init__(self, image_dir, transform):
        self.image_paths = sorted(
            [
                os.path.join(image_dir, fname)
                for fname in os.listdir(image_dir)
                if fname.lower().endswith(".jpg")
            ]
        )
        self.transform = transform
        self.loader = default_loader

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = self.loader(path)
        image = self.transform(image)
        return image, path

    def __len__(self):
        return len(self.image_paths)


def get_transforms(config):
    """
    Returns a list of transforms for training and testing.
    Args:
        config: Configuration object.
    Returns:
        train_transform: Transform for training images.
        test_transform: Transform for test images.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, test_transform


def get_dataloaders(config):
    """
    Returns the DataLoader objects for training and validation.
    Args:
        config: Configuration object.
    Returns:
        train_loader: DataLoader for training images.
        val_loader: DataLoader for validation images.
    """
    train_transform, test_transform = get_transforms(config)

    if getattr(config, "num_augmentations", 1) > 1:
        train_dataset = MultiAugDataset(
            config.train_dir,
            transform=train_transform,
            n_views=config.num_augmentations,
        )
    else:
        train_dataset = datasets.ImageFolder(
            config.train_dir, transform=train_transform
        )

    val_dataset = datasets.ImageFolder(config.val_dir, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader


def get_test_loader(config):
    """
    Returns the DataLoader object for test images.
    Args:
        config: Configuration object.
    Returns:
        test_loader: DataLoader for test images.
    """
    _, test_transform = get_transforms(config)

    if config.use_tta:

        def tta_transform():
            return transforms.Compose(
                [
                    transforms.Resize(int(config.image_size * 1.14)),
                    transforms.TenCrop(config.image_size),
                    transforms.Lambda(
                        lambda crops: torch.stack(
                            [
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                )(transforms.ToTensor()(crop))
                                for crop in crops
                            ]
                        )
                    ),
                ]
            )

        test_dataset = TestImageDataset(
            image_dir=config.test_dir, transform=tta_transform()
        )
    else:
        test_dataset = TestImageDataset(
            image_dir=config.test_dir, transform=test_transform
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return test_loader
