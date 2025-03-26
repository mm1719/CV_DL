import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from model import SupConResNet
from dataset import get_transform
from utils.logger import setup_logger


def extract_features(encoder, dataloader, device):
    encoder.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            feats = encoder(imgs)
            features.append(F.normalize(feats, dim=1).cpu())
            labels.append(lbls)
    return torch.cat(features), torch.cat(labels)


def evaluate_knn(cfg):
    print("[kNN] 使用 SupCon encoder 進行 kNN 分類...")

    log_dir = os.path.join(cfg["output_dir"], cfg["timestamp"])
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    # 資料集與 dataloader
    transform = get_transform(cfg, mode="val")
    train_set = ImageFolder(
        root=os.path.join(cfg["data_root"], "train"), transform=transform
    )
    val_set = ImageFolder(
        root=os.path.join(cfg["data_root"], "val"), transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # 載入 encoder
    encoder = SupConResNet(name=cfg["model_name"], pretrained=False).encoder
    encoder_path = os.path.join(
        cfg["output_dir"], cfg["timestamp"], f"supcon_encoder_{cfg['timestamp']}.pth"
    )
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    encoder.to(cfg["device"])

    # 抽特徵
    train_feats, train_labels = extract_features(encoder, train_loader, cfg["device"])
    val_feats, val_labels = extract_features(encoder, val_loader, cfg["device"])

    # 計算相似度（cosine）與預測
    sim_matrix = torch.mm(val_feats, train_feats.t())  # [N_val, N_train]
    pred_indices = sim_matrix.argmax(dim=1)
    pred_labels = train_labels[pred_indices]

    acc = accuracy_score(val_labels, pred_labels)
    logger.info(f"[kNN Accuracy] Top-1 Accuracy: {acc:.4f}")
    print(f"[kNN Accuracy] Top-1 Accuracy: {acc:.4f}")

    # 混淆矩陣
    cm = confusion_matrix(val_labels, pred_labels, labels=range(len(train_set.classes)))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=False, yticklabels=False)
    ax.set_title("kNN Confusion Matrix")
    cm_path = os.path.join(log_dir, "knn_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"[kNN] 混淆矩陣儲存於: {cm_path}")
