# model/weakly_maskrcnn.py

import torch
from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from utils.mil_loss import compute_mil_loss


@META_ARCH_REGISTRY.register()
class WeaklySupervisedMaskRCNN(GeneralizedRCNN):
    """
    Weakly Supervised Mask R-CNN using MIL loss on mask head.
    """

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # Proposals (from RPN, no GT box)
        proposals, proposal_losses = self.proposal_generator(images, features, None)

        # ROI heads — classification and box regression (no GT)
        _, detector_losses = self.roi_heads(images, features, proposals, None)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # Prepare image-level label vector (MIL target)
        label_vec_batch = []
        for x in batched_inputs:
            # shape: (C,), float32
            label_vec = torch.tensor(
                x["label_vec"], dtype=torch.float32, device=images.tensor.device
            )
            label_vec_batch.append(label_vec)

        # Stack into shape: (B, C)
        label_vec_batch = torch.stack(label_vec_batch, dim=0)  # (B, C)

        if self.roi_heads.mask_on:
            # Shared ROI feature for masks
            mask_features = self.roi_heads._shared_roi_transform(
                features, [x.proposal_boxes for x in proposals]
            )
            pred_mask_logits = self.roi_heads.mask_head(mask_features)  # (N, C, H, W)

            # Global average pool over (H, W), flatten to (N, C)
            pred_mask_logits = (
                torch.nn.functional.adaptive_avg_pool2d(pred_mask_logits, 1)
                .squeeze(-1)
                .squeeze(-1)
            )

            # MIL loss: input shape (N, C), target shape (C,)
            mil_loss = compute_mil_loss(
                pred_mask_logits,
                label_vec=label_vec_batch.sum(dim=0).clamp_max(1.0),
                mode="max",
            )
            losses["loss_mask"] = mil_loss

        return losses
