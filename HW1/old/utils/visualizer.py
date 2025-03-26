import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns


def plot_tsne(features, labels, save_path=None):
    tsne = TSNE(n_components=2, init="pca", random_state=0, perplexity=30)
    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings[:, 0], embeddings[:, 1], c=labels, cmap="tab20", s=5, alpha=0.6
    )
    plt.colorbar(scatter, ticks=range(0, 100))
    plt.title("t-SNE")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        fmt=".2f",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_class_variance(features, labels, save_path=None):
    # 計算每個類別的 centroid 距離與變異
    features = np.array(features)
    labels = np.array(labels)
    num_classes = len(np.unique(labels))

    intra_class_var = []
    for cls in range(num_classes):
        cls_features = features[labels == cls]
        if len(cls_features) == 0:
            intra_class_var.append(0)
        else:
            centroid = np.mean(cls_features, axis=0)
            var = np.mean(np.linalg.norm(cls_features - centroid, axis=1))
            intra_class_var.append(var)

    plt.figure(figsize=(12, 5))
    plt.bar(range(num_classes), intra_class_var)
    plt.xlabel("Class ID")
    plt.ylabel("Intra-class Variance")
    plt.title("Intra-class Feature Variance")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_knn_confusion_matrix(features, labels, save_path, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels)
    preds = knn.predict(features)

    cm = confusion_matrix(labels, preds, normalize="true")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", square=True, cbar=True)
    plt.title(f"kNN Confusion Matrix (k={k})")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
