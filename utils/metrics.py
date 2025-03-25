import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")

def plot_tsne(features, labels, save_path="tsne.png"):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="tab20", alpha=0.7, s=10)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ t-SNE saved to {save_path}")
