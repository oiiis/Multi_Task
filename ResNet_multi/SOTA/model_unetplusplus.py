import torch
import torch.nn as nn

class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes=1, feature_channels=512):
        super(UNetPlusPlus, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, feature_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(feature_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )

        # Feature projection for classification
        self.feature_projection = nn.Linear(feature_channels, 512)  # Ensure compatibility with classification head

    def forward(self, x, _):  # Ignore X-ray input
        if len(x.shape) == 5:
            x = torch.mean(x, dim=2, keepdim=False)  # Convert [B, C, D, H, W] -> [B, C, H, W]
            print(f"Adjusted input shape before encoder: {x.shape}")

        x = self.encoder(x)
        segmentation_features = x  # Ensure it outputs the correct feature map
        segmentation_output = self.decoder(x)

        # Flatten feature map for classification task
        class_features = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        class_features = self.feature_projection(class_features)  # Map to 512 features

        return segmentation_features, class_features