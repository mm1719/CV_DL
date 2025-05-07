# utils/mil_loss.py

import torch
import torch.nn.functional as F


def mil_pooling(pred_mask_logits, mode="max"):
    """
    Args:
        pred_mask_logits: Tensor of shape (N, C)
                          N = number of proposals, C = number of classes
        mode: str, pooling type: "max", "mean", or "lse"
    Returns:
        pooled_logits: (C,) pooled score for each class
    """
    if mode == "max":
        pooled_logits = pred_mask_logits.max(dim=0)[0]  # shape: (C,)
    elif mode == "mean":
        pooled_logits = pred_mask_logits.mean(dim=0)
    elif mode == "lse":
        # log-sum-exp: log(Σ exp(x_i)) ≈ smooth-max
        pooled_logits = torch.logsumexp(pred_mask_logits, dim=0)
    else:
        raise ValueError(f"Unknown MIL pooling mode: {mode}")
    return pooled_logits


def compute_mil_loss(pred_mask_logits, label_vec=None, mode="max"):
    """
    Compute MIL loss from predicted mask logits and image-level label vector.

    Args:
        pred_mask_logits: Tensor of shape (N, C), from all proposals in a batch
        label_vec: Tensor of shape (C,), binary vector indicating if class c exists in the image
                   If None, default to all-ones (assume all classes present)
        mode: str, pooling type

    Returns:
        loss: scalar, BCE loss from MIL aggregation
    """
    if pred_mask_logits.dim() != 2:
        raise ValueError("Expected 2D tensor (N, C) for pred_mask_logits")

    C = pred_mask_logits.size(1)
    if label_vec is None:
        label_vec = torch.ones(C, dtype=torch.float32, device=pred_mask_logits.device)

    # Apply sigmoid before pooling
    probs = torch.sigmoid(pred_mask_logits)  # shape: (N, C)
    pooled = mil_pooling(probs, mode=mode)  # shape: (C,)

    # Compute BCE loss against image-level label vector
    loss = F.binary_cross_entropy(pooled, label_vec)
    return loss
