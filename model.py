import torch.nn as nn
from torchvision import models

# 定义模型架构
class OpenSetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(OpenSetModel, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(resnet.fc.in_features, num_classes)
        self.open_set_layer = nn.Linear(resnet.fc.in_features, 1)  # For open set detection

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_outputs = self.classifier(features)
        open_set_output = self.open_set_layer(features)
        return class_outputs, open_set_output