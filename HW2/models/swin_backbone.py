import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import timm
from config import cfg


class SwinBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.swin_model_name,
            pretrained=pretrained,
            features_only=True,
        )

        # 🔥 手動覆寫 patch_embed 層的 img_size
        if hasattr(self.backbone, "patch_embed"):
            self.backbone.patch_embed.img_size = (256, 256)

            ps = self.backbone.patch_embed.patch_size
            if isinstance(ps, tuple):
                ps = ps[0]

            self.backbone.patch_embed.grid_size = (256 // ps, 256 // ps)
            self.backbone.patch_embed.num_patches = (
                self.backbone.patch_embed.grid_size[0]
                * self.backbone.patch_embed.grid_size[1]
            )

        # ✅ print 確認
        print("🎯 強制設定 patch_embed.img_size =", self.backbone.patch_embed.img_size)

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        patch_embed_size = (
            self.backbone.patch_embed.img_size
            if hasattr(self.backbone, "patch_embed")
            else "N/A"
        )
        print(f"🔥 Input size: ({H}, {W}) | Model expects: {patch_embed_size}")
        return self.backbone(x)


def build_swin_frcnn(num_classes=11, pretrained=True):
    # 建 backbone（含特徵抽取）
    swin = SwinBackbone(pretrained=pretrained)

    # Swin Tiny 輸出的是 C2~C4 對應 channel 數量如下
    #   C2: stage 2 output (1/8), channels=96
    #   C3: stage 3 output (1/16), channels=192
    #   C4: stage 4 output (1/32), channels=384
    #   C5: stage 5 output (1/32), channels=768

    # 選用其中 C2~C4 作為 FPN 輸入
    in_channels_list = [96, 192, 384]
    out_channels = 256  # FPN 統一輸出通道數

    # FPN 組成
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list, out_channels=out_channels
    )

    # 建 BackboneWithFPN（會接上 FPN 並只回傳 mapping）
    class SwinWithFPN(nn.Module):
        def __init__(self, backbone, fpn):
            super().__init__()
            self.body = backbone
            self.fpn = fpn
            self.out_channels = 256  # FPN 輸出 channel 數

        def forward(self, x):
            # e.g. [C2, C3, C4] → dict: {"0":..., "1":..., "2":...}
            feats = self.body(x)
            feat_dict = {str(i): f for i, f in enumerate(feats[:3])}
            print(f"🔥 Input size: ({H}, {W}) | Model expects: {patch_embed_size}")
            return self.fpn(feat_dict)

    backbone_with_fpn = SwinWithFPN(swin, fpn)

    # 建 anchor generator（可依據輸入調整）
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # 自訂 RCNN Transform（避免預設 resize 成 800x1333）
    transform = GeneralizedRCNNTransform(
        min_size=256,
        max_size=256,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    # 組成 FasterRCNN 模型
    model = FasterRCNN(
        backbone=backbone_with_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=None,  # 預設就會使用 RoIAlign
        transform=transform,
    )

    return model
