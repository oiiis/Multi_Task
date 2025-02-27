import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    """3D Residual Block for MRI processing."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Handle input channels dynamically
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

class ResidualBlock2D(nn.Module):
    """2D Residual Block for X-ray processing."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

class FusionLayer(nn.Module):
    """Fusion Layer for combining 3D MRI and 2D X-ray features."""
    def __init__(self, fused_channels, fusion_alpha=0.5):
        super(FusionLayer, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(fusion_alpha))  
        self.fusion_conv = nn.Conv3d(fused_channels, fused_channels, kernel_size=1)

    def forward(self, mri_features, xray_features):
        # Ensure X-ray features match MRI spatial dimensions
        xray_features = F.interpolate(xray_features, size=mri_features.shape[-3:], mode='trilinear', align_corners=False)
        fused_features = self.alpha * mri_features + (1 - self.alpha) * xray_features
        return self.fusion_conv(fused_features)

# the proposed OANet model
class OANet(nn.Module):
    """OANet with configurable fusion strategy & pooling type."""
    def __init__(self, fusion_strategy="late", pooling_type="adaptive", fusion_alpha=0.5):
        super(OANet, self).__init__()

        self.fusion_strategy = fusion_strategy

        # **Handle input channels for early fusion**
        input_channels = 1 if fusion_strategy == "late" else 2  

        # 3D ResNet for MRI
        self.mri_resnet = nn.Sequential(
            ResidualBlock3D(input_channels, 32, kernel_size=7, stride=2),
            ResidualBlock3D(32, 64, kernel_size=3, stride=1),
            ResidualBlock3D(64, 128, kernel_size=3, stride=1),
            ResidualBlock3D(128, 256, kernel_size=3, stride=1),
        )

        # 2D ResNet for X-ray
        self.xray_resnet = nn.Sequential(
            ResidualBlock2D(1, 32, kernel_size=7, stride=2),
            ResidualBlock2D(32, 64, kernel_size=3, stride=1),
            ResidualBlock2D(64, 128, kernel_size=3, stride=1),
            ResidualBlock2D(128, 256, kernel_size=3, stride=1),
        )

        # Adaptive Pooling
        self.agp_mri = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.agp_xray = nn.AdaptiveAvgPool2d((1, 1))

        # Fusion Layer
        self.fusion_layer = FusionLayer(fused_channels=256, fusion_alpha=fusion_alpha)

        # Fully connected output layer
        self.fc = nn.Linear(256, 1)

    def forward(self, mri, xray):
        print(f"Input MRI shape: {mri.shape}")  # Debugging
        print(f"Input X-ray shape: {xray.shape}")

        if self.fusion_strategy == "early":
            # Early Fusion: Resize & concatenate MRI & X-ray before feature extraction
            xray_resized = F.interpolate(xray.unsqueeze(1), size=mri.shape[-3:], mode='trilinear', align_corners=False)
            fused_input = torch.cat((mri, xray_resized), dim=1)  
            print(f"Early Fusion - Fused input shape: {fused_input.shape}")  # Debugging
            features = self.mri_resnet(fused_input)

        else:
            # Late Fusion: Extract features separately, then fuse
            mri_features = self.mri_resnet(mri)
            print(f"Late Fusion - MRI features shape: {mri_features.shape}")  # Debugging

            xray_features = self.xray_resnet(xray)
            print(f"Late Fusion - X-ray features shape: {xray_features.shape}")  # Debugging

            xray_resized = F.interpolate(xray_features.unsqueeze(2), size=mri_features.shape[-3:], mode='trilinear', align_corners=False)
            fused_features = self.fusion_layer(mri_features, xray_resized)

            features = fused_features.view(fused_features.size(0), -1)

        return self.fc(features)

# **Test the model before running training**
if __name__ == "__main__":
    for fusion in ["early", "late"]:
        model = OANet(fusion_strategy=fusion, fusion_alpha=0.5)

        mri_input = torch.randn(2, 1, 160, 384, 384)  # (Batch, Channels, Depth, Height, Width)
        xray_input = torch.randn(2, 1, 384, 384)  # (Batch, Channels, Height, Width)

        output = model(mri_input, xray_input)
        print(f"Config: Fusion={fusion} → Output shape: {output.shape}")
