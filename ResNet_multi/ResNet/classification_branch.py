import torch
import torch.nn as nn

class ClassificationBranch3D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ClassificationBranch3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(input_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


