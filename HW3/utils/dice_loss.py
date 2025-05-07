# utils/dice_loss.py

import torch
import torch.nn.functional as F


def dice_loss(pred_logits, target_mask, smooth=1e-5):
    """
    Args:
        pred_logits: shape (N, 1, H, W) or (N, H, W)
        target_mask: same shape, float in {0,1}
    Returns:
        Dice loss: scalar
    """
    pred_probs = torch.sigmoid(pred_logits)
    if pred_probs.dim() == 4:
        pred_probs = pred_probs[:, 0, :, :]
    if target_mask.dim() == 4:
        target_mask = target_mask[:, 0, :, :]

    intersection = (pred_probs * target_mask).sum(dim=(1, 2))
    union = pred_probs.sum(dim=(1, 2)) + target_mask.sum(dim=(1, 2))

    dice = (2 * intersection + smooth) / (union + smooth)
    loss = 1 - dice.mean()
    return loss
