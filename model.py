import torch
import torch.nn as nn
import timm

def build_model(config):
    # 建立預訓練模型
    model = timm.create_model(
        config.model_name,
        pretrained=config.pretrained,
        num_classes=0  # 先去掉預設分類頭
    )

    in_features = model.num_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, config.num_classes)
    )

    # 是否使用多 GPU
    if config.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model
