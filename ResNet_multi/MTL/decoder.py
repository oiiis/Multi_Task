import torch
import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_channels=2048, out_channels=1):
        super(SegmentationHead, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(128, out_channels, kernel_size=1)  # Output channel = 1 (binary segmentation)

    def forward(self, x):
        print(f"SegmentationHead input shape: {x.shape}")  # Debugging print
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)

        # Upsample to match label size
        x = torch.nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        print(f"SegmentationHead output shape AFTER upsampling: {x.shape}")  # Debugging print

        return x

class ClassificationHead(nn.Module):
    def __init__(self, in_channels=2048, num_classes=2):  # Fix input features to 2048
        super(ClassificationHead, self).__init__()
        
        self.fc = nn.Linear(in_channels, num_classes)  # Ensure matching feature size

    def forward(self, x):
        print(f"ClassificationHead input shape BEFORE flattening: {x.shape}")  # Debugging print

        x = x.view(x.shape[0], -1)  # Flatten
        print(f"ClassificationHead input shape AFTER flattening: {x.shape}")  # Debugging print
        
        return self.fc(x)


class RiskPredictionHead(nn.Module):
    def __init__(self, in_channels=2048, num_classes=2):  # Fix input channels
        super(RiskPredictionHead, self).__init__()
        
        self.fc = nn.Linear(in_channels, num_classes)  # Match 2048 input features

    def forward(self, x):
        print(f"RiskPredictionHead input shape BEFORE flattening: {x.shape}")  # Debugging print

        x = x.view(x.shape[0], -1)  # Flatten before FC layer
        print(f"RiskPredictionHead input shape AFTER flattening: {x.shape}")  # Debugging print

        return self.fc(x)

