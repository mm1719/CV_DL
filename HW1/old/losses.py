import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    來自原始 Supervised Contrastive Learning 論文的實作
    https://arxiv.org/pdf/2004.11362.pdf
    """

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        features = F.normalize(features, dim=2)

        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)

        contrast_count = features.shape[1]  # e.g., 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [2N, D]

        anchor_feature = contrast_feature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # remove self-contrast
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = logits_mask.repeat(contrast_count, contrast_count)
        mask = mask * logits_mask

        # 計算 log-softmax 並取 positive 部分的平均
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(
            exp_logits.sum(1, keepdim=True) + 1e-12
        )

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()

        return loss


class WeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights):
        super(WeightedCrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, labels):
        return self.criterion(logits, labels)
