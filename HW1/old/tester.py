import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from dataset import get_datasets, get_tta_transform
from model import SupConResNet, LinearClassifier, MLPClassifier
from utils.logger import setup_logger
from utils.visualizer import plot_tsne, plot_knn_confusion_matrix, plot_class_variance
from utils.metrics import accuracy
from tqdm import tqdm


def test_model(cfg):
    print("[Test] 開始測試與輸出 prediction.csv...")
    _, val_set, test_set = get_datasets(cfg, mode="test")
    test_loader = DataLoader(
        test_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    encoder = SupConResNet(name=cfg["model_name"], pretrained=False).encoder
    if cfg.get("classifier_type", "linear") == "mlp":
        classifier = MLPClassifier(in_dim=2048, num_classes=cfg["num_classes"])
    else:
        classifier = LinearClassifier(in_dim=2048, num_classes=cfg["num_classes"])

    encoder_path = os.path.join(
        cfg["output_dir"], cfg["timestamp"], f"supcon_encoder_{cfg['timestamp']}.pth"
    )
    classifier_path = os.path.join(
        cfg["output_dir"], cfg["timestamp"], f"classifier_best_{cfg['timestamp']}.pth"
    )
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))

    encoder.to(cfg["device"]).eval()
    classifier.to(cfg["device"]).eval()

    # 多 GPU 支援
    if torch.cuda.device_count() > 1:
        print(f"[Info] 使用 {torch.cuda.device_count()} 顆 GPU 進行測試")
        encoder = nn.DataParallel(encoder)
        classifier = nn.DataParallel(classifier)

    log_dir = os.path.join(cfg["output_dir"], cfg["timestamp"])
    logger = setup_logger(log_dir)

    tta_times = cfg.get("tta_times", 1)
    tta_transform = get_tta_transform(cfg)

    image_ids, predictions, features = [], [], []
    with torch.no_grad():
        for _, paths in tqdm(test_loader, desc="Testing"):
            # 用原始檔案路徑進行多次 TTA 推論
            batch_preds = []
            batch_feats = []

            for _ in range(tta_times):
                augmented_images = torch.stack(
                    [tta_transform(Image.open(p).convert("RGB")) for p in paths]
                )
                augmented_images = augmented_images.to(cfg["device"])

                feats = encoder(augmented_images)
                logits = classifier(feats)

                batch_feats.append(feats.cpu())
                batch_preds.append(logits.softmax(dim=1).cpu())

            # 平均多次 logits 後 argmax
            avg_logits = torch.stack(batch_preds).mean(dim=0)
            final_preds = torch.argmax(avg_logits, dim=1)

            # 平均特徵用於 t-SNE
            avg_feats = torch.stack(batch_feats).mean(dim=0)
            features.append(avg_feats)

            predictions.extend(final_preds.numpy())
            image_ids.extend([os.path.splitext(os.path.basename(p))[0] for p in paths])

    print(f"[Debug] len(image_ids) = {len(image_ids)}")
    print(f"[Debug] len(predictions) = {len(predictions)}")

    df = pd.DataFrame({"image_name": image_ids, "pred_label": predictions})
    csv_path = os.path.join(cfg["output_dir"], cfg["timestamp"], "prediction.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"[輸出完成] 預測結果儲存至 {csv_path}")

    # t-SNE 可視化
    if cfg.get("tsne", True):
        tsne_path = os.path.join(cfg["output_dir"], cfg["timestamp"], "tsne_plot.png")
        plot_tsne(torch.cat(features).numpy(), predictions, tsne_path)
        logger.info(f"[t-SNE] 可視化儲存於 {tsne_path}")

    # 類別內變異分析（optional）
    if cfg.get("tsne", True):
        var_path = os.path.join(
            cfg["output_dir"], cfg["timestamp"], "intra_class_variance.png"
        )
        plot_class_variance(torch.cat(features).numpy(), predictions, var_path)
        logger.info(f"[Variance] 類別內變異圖儲存於 {var_path}")

        # kNN Confusion Matrix
        if cfg.get("tsne", True):  # 使用相同的 flag 控制
            knn_cm_path = os.path.join(
                cfg["output_dir"], cfg["timestamp"], "knn_confusion_matrix.png"
            )
            plot_knn_confusion_matrix(
                torch.cat(features).numpy(), predictions, knn_cm_path, k=5
            )
            logger.info(f"[kNN Confusion] 已儲存至 {knn_cm_path}")
