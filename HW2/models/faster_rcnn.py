import torch
from config import cfg
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes=10, pretrained=True):
    if cfg.backbone == "resnet50":
        # 使用 PyTorch 官方提供的 v2 版本
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model.to(cfg.device)

    elif cfg.backbone == "swin":
        # 若使用 Swin 則使用自定義的實作
        from models.swin_backbone import build_swin_frcnn

        return build_swin_frcnn(num_classes).to(cfg.device)

    else:
        raise ValueError(f"不支援的 backbone: {cfg.backbone}")
