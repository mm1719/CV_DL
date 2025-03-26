import torch
from tqdm import tqdm

def extract_features_for_tsne(model, dataloader, device):
    """
    Extract features from a model for t-SNE visualization.
    Args:
        model: A pre-trained model.
        dataloader: PyTorch dataloader.
        device: Device to run the model on.
    Returns:
        features: Extracted features.
        labels: Labels for each feature.
    """
    model.eval()
    features, all_labels = [], []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            if isinstance(images, list):
                images = torch.cat(images, dim=0)
                targets = targets.repeat_interleave(len(images) // len(targets))
            images = images.to(device)
            outputs = model.forward_features(images)  # TIMM 模型內建

            if outputs.ndim == 4:
                outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)

            features.append(outputs.cpu())
            all_labels.append(targets.cpu())

    features = torch.cat(features)
    labels = torch.cat(all_labels)

    return features.numpy(), labels.numpy()
