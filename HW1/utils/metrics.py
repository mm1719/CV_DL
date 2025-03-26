import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import os


def plot_confusion_matrix(y_true, y_pred, save_path=None, labels=None):
    """
    Plot the confusion matrix.
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        save_path: Path to save the plot.
        labels: List of class labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=labels if labels is not None else np.arange(cm.shape[0]),
        yticklabels=labels if labels is not None else np.arange(cm.shape[0]),
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_tsne(features, labels, save_path="tsne.png"):
    """
    Plot t-SNE visualization.
    Args:
        features: Features to visualize.
        labels: Labels for each feature.
        save_path: Path to save the plot.
    """
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1], c=labels, cmap="tab20", alpha=0.7, s=10
    )
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization")
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE saved to {save_path}")
