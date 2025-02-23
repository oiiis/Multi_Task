import torch
import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.upsample(x)

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class RiskPredictionHead(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(RiskPredictionHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
