import os
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import get_datasets, get_transform
from model import SupConResNet

def extract_features(encoder, dataloader, device):
    encoder.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feats = encoder(images)
            features.append(feats.cpu())
            labels.extend(lbls)
    return torch.cat(features).numpy(), labels

def plot_tsne(features, labels, save_path=None):
    print("[t-SNE] 降維中...")
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=10, alpha=0.6)
    plt.colorbar(scatter)
    plt.title("t-SNE of SupCon Encoder Features")
    if save_path:
        plt.savefig(save_path)
        print(f"[t-SNE] 圖片儲存至 {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', required=True, help="訓練結果資料夾 timestamp")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'], help="選擇資料集")
    args = parser.parse_args()

    # 載入設定
    from config import get_config
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型
    encoder = SupConResNet(name=cfg['model_name'], pretrained=False).encoder
    encoder_path = os.path.join(cfg['output_dir'], args.timestamp, f"supcon_encoder_{args.timestamp}.pth")
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    encoder.to(device)

    # 資料集
    if args.mode == 'train':
        dataset, _, _ = get_datasets(cfg, mode='tsne_plt')
    elif args.mode == 'val':
        _, dataset, _ = get_datasets(cfg, mode='tsne_plt')
    else:  # test 沒 label，跳過
        print("⚠️ test set 沒有 label，無法畫 t-SNE。")
        return

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=True)

    # 特徵提取
    features, labels = extract_features(encoder, dataloader, device)

    # 繪圖
    save_path = os.path.join(cfg['output_dir'], args.timestamp, f"tsne_supcon_{args.mode}.png")
    plot_tsne(features, labels, save_path)

if __name__ == '__main__':
    main()
