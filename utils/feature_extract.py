import torch
from tqdm import tqdm

def extract_features_for_tsne(model, dataloader, device):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            outputs = model.forward_features(images)  # TIMM 模型內建
            features.append(outputs.cpu())
            labels.extend(targets)

    return torch.cat(features).numpy(), labels
