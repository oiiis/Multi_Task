import torch
import torch.nn as nn

class SegmentationHead(nn.Module):
    """Segmentation head for multi-task learning."""
    def __init__(self, in_channels=2048, out_channels=1):
        super(SegmentationHead, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(128, out_channels, kernel_size=1)  # Output: 1-channel binary segmentation mask

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
    """Classification head for OA prediction."""
    def __init__(self, in_channels=2048, num_classes=2):
        super(ClassificationHead, self).__init__()

        # Ensure features are projected correctly
        if in_channels != 2048:
            self.feature_projection = nn.Linear(in_channels, 2048)
        else:
            self.feature_projection = nn.Identity()  # No projection needed if already 2048

        # Global Average Pooling (only for 4D tensors)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        print(f"ClassificationHead input shape BEFORE processing: {x.shape}")

        # If input is already 2D (e.g., [batch, 512]), apply linear projection
        if x.dim() == 2:
            if x.shape[1] < 512:
                print(f"Warning: Unexpected feature size {x.shape}, expanding to 512.")
                x = torch.cat([x, torch.zeros(x.shape[0], 512 - x.shape[1]).to(x.device)], dim=1)

            if x.shape[1] != 2048:  # Only project if needed
                x = self.feature_projection(x)

        elif x.dim() == 4:
            x = self.global_avg_pool(x)  # Reduce to [batch, in_channels, 1, 1]
            x = torch.flatten(x, 1)  # Flatten to [batch, in_channels]

        print(f"ClassificationHead input shape BEFORE FC: {x.shape}")
        return self.fc(x)  # Output: [batch, num_classes]
    
    

class RiskPredictionHead(nn.Module):
    """Risk Prediction Head"""
    def __init__(self, in_channels=2048, num_classes=2):
        super(RiskPredictionHead, self).__init__()

        if in_channels != 2048:
            self.feature_projection = nn.Linear(in_channels, 2048)
        else:
            self.feature_projection = nn.Identity()  # No projection needed if already 2048

        # Global Average Pooling (for 4D inputs)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        print(f"RiskPredictionHead input shape BEFORE processing: {x.shape}")

        # If input is 2D (batch, features), apply feature projection
        if x.dim() == 2:
            if x.shape[1] < 512:
                print(f"Warning: Expanding features for RiskPredictionHead. Current shape: {x.shape}")
                x = torch.cat([x, torch.zeros(x.shape[0], 512 - x.shape[1]).to(x.device)], dim=1)

            if x.shape[1] != 2048:
                x = self.feature_projection(x)

        elif x.dim() == 4:
            x = self.global_avg_pool(x)
            x = torch.flatten(x, 1)

        print(f"RiskPredictionHead input shape BEFORE FC: {x.shape}")
        return self.fc(x)