import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from tqdm import tqdm

from dataset import get_datasets, get_tta_transform
from model import SupConResNet
from utils.logger import setup_logger


def extract_features(model, dataloader, device, tta_transform=None, tta_times=1, sample_paths=None):
    features, paths = [], []
    index = 0
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting features"):
            batch_size = images.size(0)
            batch_feats = []
            for _ in range(tta_times):
                augmented = torch.stack([tta_transform(transforms.ToPILImage()(img.cpu())) for img in images])
                augmented = augmented.to(device)
                feats = model(augmented)
                batch_feats.append(feats.cpu())
            avg_feats = torch.stack(batch_feats).mean(dim=0)
            features.append(avg_feats)
            paths.extend(sample_paths[index:index + batch_size])
            index += batch_size
    return torch.cat(features).numpy(), paths


def main(cfg, timestamp):
    print("[kNN Test] 使用 kNN 分類器進行測試...")

    cfg['timestamp'] = timestamp
    log_dir = os.path.join(cfg['output_dir'], timestamp)
    logger = setup_logger(log_dir)

    # GPU 設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg['device'] = device

    # 資料載入
    train_set, val_set, test_set = get_datasets(cfg, mode='test')
    val_sample_paths = [p for p, _ in val_set.samples]
    test_sample_paths = [p for p, _ in test_set.samples]

    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False,
                             num_workers=cfg['num_workers'], pin_memory=True)

    # Encoder 載入
    encoder = SupConResNet(name=cfg['model_name'], pretrained=False).encoder
    encoder_path = os.path.join(cfg['output_dir'], timestamp, f"supcon_encoder_{timestamp}.pth")
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    encoder.to(cfg['device']).eval()

    # TTA 設定
    tta_transform = get_tta_transform(cfg)
    tta_times = cfg.get("tta_times", 1)

    print("[Step 1] 提取 Validation Set 特徵 (作為 support set)...")
    val_features, _ = extract_features(encoder, val_loader, cfg['device'], tta_transform, tta_times, val_sample_paths)
    val_labels = [label for _, label in val_set.samples]

    print("[Step 2] 提取 Test Set 特徵 (作為 query)...")
    test_features, test_paths = extract_features(encoder, test_loader, cfg['device'], tta_transform, tta_times, test_sample_paths)

    # kNN
    k = cfg.get('knn_k', 5)
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(val_features, val_labels)
    test_preds = knn.predict(test_features)

    # 輸出 CSV
    image_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]
    df = pd.DataFrame({'image_name': image_ids, 'pred_label': test_preds})
    save_path = os.path.join(cfg['output_dir'], timestamp, f'knn_prediction.csv')
    df.to_csv(save_path, index=False)
    logger.info(f"[輸出完成] kNN 預測結果儲存至 {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', type=str, required=True, help='對應的訓練結果資料夾 timestamp')
    args = parser.parse_args()

    from config import get_config
    cfg = get_config()

    main(cfg, args.timestamp)
