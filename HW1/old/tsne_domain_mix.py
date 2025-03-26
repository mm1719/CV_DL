import os
import argparse
from PIL import Image
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import get_datasets, get_tta_transform
from model import SupConResNet


@torch.no_grad()
def extract_features(encoder, dataloader, device, tta_transform, tta_times):
    features = []
    for _, paths in tqdm(dataloader):
        batch_feats = []
        for _ in range(tta_times):
            augmented_images = torch.stack(
                [tta_transform(Image.open(p).convert("RGB")) for p in paths]
            )
            augmented_images = augmented_images.to(device)
            feats = encoder(augmented_images).cpu()
            batch_feats.append(feats)
        features.append(torch.stack(batch_feats).mean(dim=0))
    return torch.cat(features)


def plot_tsne_all(features, domains, save_path):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(features)

    colors = {"train": "blue", "val": "green", "test": "red"}
    plt.figure(figsize=(10, 8))
    for domain in ["train", "val", "test"]:
        idxs = [i for i, d in enumerate(domains) if d == domain]
        plt.scatter(
            tsne_result[idxs, 0],
            tsne_result[idxs, 1],
            s=10,
            alpha=0.6,
            label=domain,
            color=colors[domain],
        )

    plt.legend()
    plt.title("t-SNE of Train / Val / Test Features")
    plt.savefig(save_path)
    print(f"[t-SNE] 儲存至：{save_path}")


def main(cfg, timestamp):
    cfg["timestamp"] = timestamp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["device"] = device

    tta_transform = get_tta_transform(cfg)
    tta_times = cfg.get("tta_times", 1)

    train_set, val_set, test_set = get_datasets(cfg, mode="tsne_plt")
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=False)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False)

    encoder = SupConResNet(name=cfg["model_name"], pretrained=False).encoder
    encoder_path = os.path.join(
        cfg["output_dir"], timestamp, f"supcon_encoder_{timestamp}.pth"
    )
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    encoder.to(device).eval()

    print("[Step 1] 提取 Train 特徵...")
    train_features = extract_features(
        encoder, train_loader, device, tta_transform, tta_times
    )
    train_domains = ["train"] * train_features.size(0)

    print("[Step 2] 提取 Val 特徵...")
    val_features = extract_features(
        encoder, val_loader, device, tta_transform, tta_times
    )
    val_domains = ["val"] * val_features.size(0)

    print("[Step 3] 提取 Test 特徵...")
    test_features = extract_features(
        encoder, test_loader, device, tta_transform, tta_times
    )
    test_domains = ["test"] * test_features.size(0)

    # 合併
    all_features = torch.cat([train_features, val_features, test_features]).numpy()
    all_domains = train_domains + val_domains + test_domains

    # 畫圖
    save_path = os.path.join(cfg["output_dir"], timestamp, "tsne_domain_mix.png")
    plot_tsne_all(all_features, all_domains, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()

    from config import get_config

    cfg = get_config()

    main(cfg, args.timestamp)
