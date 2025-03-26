import torch
import torch.nn.functional as F


def accuracy(output, target, topk=(1,)):
    """
    計算 Top-k Accuracy
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k / batch_size).item())
        return res


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, temperature=0.1):
    """
    計算單筆特徵的 kNN 預測結果（用於嵌入表示）
    """
    sim_matrix = torch.mm(feature, feature_bank.t())
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = feature_labels[sim_indices]
    sim_weight = (sim_weight / temperature).exp()

    one_hot = torch.zeros(feature.size(0) * knn_k, classes, device=feature.device)
    one_hot.scatter_(1, sim_labels.view(-1, 1), 1)
    weighted_one_hot = one_hot.view(feature.size(0), knn_k, -1) * sim_weight.unsqueeze(-1)
    probs = weighted_one_hot.sum(1)
    pred_labels = probs.argsort(dim=-1, descending=True)
    return pred_labels
