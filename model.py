import torch
import torch.nn as nn
import timm

def build_model(config):
    # 建立預訓練模型
    model = timm.create_model(
        config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes  # 直接改成 100 類
    )

    # 是否使用多 GPU
    if config.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model
