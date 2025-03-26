import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from dataset import get_datasets, compute_class_weights
from losses import WeightedCrossEntropy
from model import SupConResNet, LinearClassifier, MLPClassifier
from utils.logger import setup_logger
from utils.metrics import accuracy
from utils.visualizer import plot_confusion_matrix
from tqdm import tqdm


def train_classifier(cfg):
    print("[Classifier] 啟動分類器微調階段...")

    train_set, val_set, _ = get_datasets(cfg, mode="classifier")
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
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

    # Logger
    log_dir = os.path.join(cfg["output_dir"], cfg["timestamp"])
    logger = setup_logger(log_dir)

    # 建立 encoder 並載入預訓練權重
    encoder = SupConResNet(name=cfg["model_name"], pretrained=cfg["pretrained"]).encoder
    encoder_path = os.path.join(
        cfg["output_dir"], cfg["timestamp"], f"supcon_encoder_{cfg['timestamp']}.pth"
    )
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))

    if cfg["freeze_backbone"]:
        for param in encoder.parameters():
            param.requires_grad = False
        logger.info("[Info] Encoder 已凍結，不參與分類器訓練")

    # 初始化分類器
    if cfg.get("classifier_type", "linear") == "mlp":
        classifier = MLPClassifier(in_dim=2048, num_classes=len(train_set.classes))
        logger.info("[Info] 使用 MLPClassifier")
    else:
        classifier = LinearClassifier(in_dim=2048, num_classes=len(train_set.classes))
        logger.info("[Info] 使用 LinearClassifier")

    # 支援多 GPU
    encoder.to(cfg["device"])
    classifier.to(cfg["device"])
    if torch.cuda.device_count() > 1:
        print(f"[Multi-GPU] 使用 {torch.cuda.device_count()} 顆 GPU")
        encoder = nn.DataParallel(encoder)
        classifier = nn.DataParallel(classifier)

    # 損失與 optimizer
    class_weights = compute_class_weights(train_set).to(cfg["device"])
    criterion = WeightedCrossEntropy(class_weights)
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    # TensorBoard
    writer = SummaryWriter(os.path.join(log_dir, "logs"))

    best_acc = 0.0
    for epoch in range(cfg["epochs_cls"]):
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            images, labels = images.to(cfg["device"]), labels.to(cfg["device"])
            with torch.no_grad():
                features = encoder(images)
            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/classifier_train", avg_loss, epoch)
        writer.add_scalar("Acc/classifier_train", train_acc, epoch)

        # Validation
        val_acc, val_preds, val_targets = evaluate(encoder, classifier, val_loader, cfg)
        writer.add_scalar("Acc/classifier_val", val_acc, epoch)

        logger.info(
            f"[Epoch {epoch+1}/{cfg['epochs_cls']}] Train Loss: {avg_loss:.4f} Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f} Time: {time.time() - start_time:.1f}s"
        )

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(
                cfg["output_dir"],
                cfg["timestamp"],
                f"classifier_best_{cfg['timestamp']}.pth",
            )
            torch.save(classifier.state_dict(), save_path)
            logger.info(f"[模型儲存] 儲存最佳分類器至: {save_path}")

            # 畫 confusion matrix
            cm_path = os.path.join(
                cfg["output_dir"],
                cfg["timestamp"],
                f"confusion_matrix_epoch{epoch+1}.png",
            )
            plot_confusion_matrix(val_targets, val_preds, train_set.classes, cm_path)

    writer.close()


def evaluate(encoder, classifier, val_loader, cfg):
    encoder.eval()
    classifier.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Evaluating"):
            images, labels = images.to(cfg["device"]), labels.to(cfg["device"])
            features = encoder(images)
            logits = classifier(features)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return correct / total, all_preds, all_labels
