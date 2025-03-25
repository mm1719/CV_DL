import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import config

def get_transforms(config):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, test_transform

def get_dataloaders(config):
    train_transform, test_transform = get_transforms(config)

    train_dataset = datasets.ImageFolder(config.train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(config.val_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers,
                              pin_memory=config.pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers,
                            pin_memory=config.pin_memory)

    return train_loader, val_loader

def get_test_loader(config):
    _, test_transform = get_transforms(config)

    test_dataset = datasets.ImageFolder(
        root=os.path.dirname(config.test_dir),
        transform=test_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=config.num_workers,
                             pin_memory=config.pin_memory)
    
    return test_loader
