
import torch
import torch.nn as nn
from torchvision import models

class SupConResNet(nn.Module):
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, pretrained=True):
        super(SupConResNet, self).__init__()
        self.encoder = self._get_backbone(name, pretrained)
        dim_in = self.encoder.fc.in_features

        if head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.BatchNorm1d(dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_in),
                nn.BatchNorm1d(dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            self.head = nn.Linear(dim_in, feat_dim)

        self.encoder.fc = nn.Identity()

    def _get_backbone(self, name, pretrained):
        if name == 'resnet50':
            return models.resnet50(pretrained=pretrained)
        elif name == 'resnet18':
            return models.resnet18(pretrained=pretrained)
        elif name == 'resnext101_32x8d':
            return models.resnext101_32x8d(pretrained=pretrained)
        elif name == 'resnext50_32x4d':
            return models.resnext50_32x4d(pretrained=pretrained)
        else:
            raise ValueError(f"不支援的模型架構: {name}")

    def forward(self, x):
        feat = self.encoder(x)
        out = nn.functional.normalize(self.head(feat), dim=1)
        return out

class LinearClassifier(nn.Module):
    def __init__(self, in_dim=2048, num_classes=100):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=1024, num_classes=100):
        super(MLPClassifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)