import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    """
    3D Residual Block for MRI processing.
    """
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
    """
    2D Residual Block for X-ray processing.
    """
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


class AdaptiveGlobalPooling(nn.Module):
    """
    Adaptive Global Pooling Layer.
    """
    def __init__(self, output_size=(1,1,1)):
        super(AdaptiveGlobalPooling, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(output_size)
    
    def forward(self, x):
        return self.global_pool(x)


class FusionLayer(nn.Module):
    """
    Fusion Layer for combining 3D MRI and 2D X-ray features.
    """
    def __init__(self, in_channels_3D, in_channels_2D, fused_channels):
        super(FusionLayer, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable weighting factor
        self.fusion_conv = nn.Conv2d(fused_channels, fused_channels, kernel_size=1)

    def forward(self, mri_features, xray_features):
        # Ensure X-ray features match MRI spatial dimensions
        xray_features = F.interpolate(xray_features, size=mri_features.shape[-2:], mode='bilinear', align_corners=False)

        # Weighted feature fusion
        fused_features = self.alpha * mri_features + (1 - self.alpha) * xray_features

        return self.fusion_conv(fused_features)


class OANet(nn.Module):
    """
    OANet architecture with Dual-stream ResNet, Adaptive Global Pooling, and Fusion Layers.
    """
    def __init__(self):
        super(OANet, self).__init__()
        
        # 3D ResNet for MRI
        self.mri_resnet = nn.Sequential(
            ResidualBlock3D(1, 32, kernel_size=7, stride=2),
            ResidualBlock3D(32, 64, kernel_size=3, stride=1),
            ResidualBlock3D(64, 128, kernel_size=3, stride=1),
            ResidualBlock3D(128, 256, kernel_size=3, stride=1),
        )
        
        # 2D ResNet for X-ray (Fixed channel mismatch)
        self.xray_resnet = nn.Sequential(
            ResidualBlock2D(1, 32, kernel_size=7, stride=2),  # Fix: Input is 1 channel
            ResidualBlock2D(32, 64, kernel_size=3, stride=1),
            ResidualBlock2D(64, 128, kernel_size=3, stride=1),
            ResidualBlock2D(128, 256, kernel_size=3, stride=1),
        )
        
        # Adaptive Global Pooling
        self.agp_mri = AdaptiveGlobalPooling(output_size=(1,1,1))
        self.agp_xray = nn.AdaptiveAvgPool2d((1,1))
        
        # Fusion Layer
        self.fusion_layer = FusionLayer(256, 256, fused_channels=256)
        
        # Fully connected output layer
        self.fc = nn.Linear(256, 1)  # Adjust output size based on task (classification/segmentation)
        
    def forward(self, mri, xray):
        # Process MRI
        mri_features = self.mri_resnet(mri)
        mri_features = self.agp_mri(mri_features).squeeze()

        # Process X-ray
        xray_features = self.xray_resnet(xray)
        xray_features = self.agp_xray(xray_features).squeeze()

        # Fix batch size issues
        if mri_features.dim() == 1:
            mri_features = mri_features.unsqueeze(0)
        if xray_features.dim() == 1:
            xray_features = xray_features.unsqueeze(0)

        # Fusion
        fused_features = self.fusion_layer(mri_features.unsqueeze(-1).unsqueeze(-1), xray_features.unsqueeze(-1).unsqueeze(-1))

        # Flatten and pass to classifier
        fused_features = fused_features.view(fused_features.size(0), -1)
        output = self.fc(fused_features)

        return output


# **Test the model before running training**
if __name__ == "__main__":
    model = OANet()
    mri_input = torch.randn(2, 1, 160, 256, 256)  # (Batch, Channels, Depth, Height, Width)
    xray_input = torch.randn(2, 1, 256, 256)  # (Batch, Channels, Height, Width)
    output = model(mri_input, xray_input)
    print("Output shape:", output.shape)  # Expected: (2, 1)
