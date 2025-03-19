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
        xray_features = F.interpolate(xray_features, size=mri_features.shape[-3:], mode='trilinear', align_corners=False)
        fused_features = self.alpha * mri_features + (1 - self.alpha) * xray_features
        return self.fusion_conv(fused_features)

class OANet(nn.Module):
    """OANet with configurable fusion strategy and adaptive pooling."""
    def __init__(self, fusion_strategy="late", fusion_alpha=0.5, pooling_type="traditional"):
        super(OANet, self).__init__()

        self.fusion_strategy = fusion_strategy
        self.pooling_type = pooling_type
        input_channels = 1 if fusion_strategy == "late" else 2  

        self.mri_resnet = nn.Sequential(
            ResidualBlock3D(input_channels, 32, kernel_size=7, stride=2),
            ResidualBlock3D(32, 64, kernel_size=3, stride=1),
            ResidualBlock3D(64, 128, kernel_size=3, stride=1),
            ResidualBlock3D(128, 256, kernel_size=3, stride=1),
        )

        self.xray_resnet = nn.Sequential(
            ResidualBlock2D(1, 32, kernel_size=7, stride=2),
            ResidualBlock2D(32, 64, kernel_size=3, stride=1),
            ResidualBlock2D(64, 128, kernel_size=3, stride=1),
            ResidualBlock2D(128, 256, kernel_size=3, stride=1),
        )

        self.fusion_layer = FusionLayer(fused_channels=256, fusion_alpha=fusion_alpha)
        self.fc = nn.Linear(256, 1)

        # Set pooling type
        if pooling_type == "traditional":
            self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif pooling_type == "adaptive":
            self.pooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        elif pooling_type == "equal":
            self.pooling = nn.Identity()  # No pooling, keeping all dimensions
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

    def forward(self, mri, xray):
        print(f"Input MRI shape: {mri.shape}")
        print(f"Input X-ray shape: {xray.shape}")

        if self.fusion_strategy == "early":
            xray_resized = F.interpolate(xray.unsqueeze(1), size=mri.shape[-3:], mode='trilinear', align_corners=False)
            fused_input = torch.cat((mri, xray_resized), dim=1)  
            print(f"Early Fusion - Fused input shape: {fused_input.shape}")

            fused_features = self.mri_resnet(fused_input)

        else:
            mri_features = self.mri_resnet(mri)
            print(f"Late Fusion - MRI features shape: {mri_features.shape}")

            xray_features = self.xray_resnet(xray)
            print(f"Late Fusion - X-ray features shape: {xray_features.shape}")

            xray_resized = F.interpolate(xray_features.unsqueeze(2), size=mri_features.shape[-3:], mode='trilinear', align_corners=False)
            fused_features = self.fusion_layer(mri_features, xray_resized)

        pooled_features = self.pooling(fused_features)  
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  

        return self.fc(pooled_features)

if __name__ == "__main__":
    for fusion in ["early", "late"]:
        for pooling in ["traditional", "adaptive", "equal"]:
            print(f"Testing OANet with Fusion={fusion}, Pooling={pooling}")
            model = OANet(fusion_strategy=fusion, fusion_alpha=0.5, pooling_type=pooling)

            mri_input = torch.randn(2, 1, 160, 384, 384)  
            xray_input = torch.randn(2, 1, 384, 384)  

            output = model(mri_input, xray_input)
            print(f"Config: Fusion={fusion}, Pooling={pooling} → Output shape: {output.shape}")