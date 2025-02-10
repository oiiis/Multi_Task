import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive_global_pooling import AdaptiveGlobalPoolingLayer

class ResidualBlock3D(nn.Module):
    """3D Residual Block for MRI Processing"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ResidualBlock2D(nn.Module):
    """2D Residual Block for X-ray Processing"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class DualResNet(nn.Module):
    """Dual-stream ResNet for MRI and X-ray Processing, now with AGPL."""
    def __init__(self, num_blocks):
        super(DualResNet, self).__init__()
        self.mri_stream = self._make_layer3D(1, 64, num_blocks)
        self.xray_stream = self._make_layer2D(1, 64, num_blocks)
        self.agpl_mri = AdaptiveGlobalPoolingLayer(num_blocks, 64)
        self.agpl_xray = AdaptiveGlobalPoolingLayer(num_blocks, 64)

    def forward(self, mri, xray):
        mri_features = [layer(mri) for layer in self.mri_stream]  # Extract features per block
        xray_features = [layer(xray) for layer in self.xray_stream]
        
        # Apply AGPL to both streams
        mri_pooled = self.agpl_mri(mri_features)
        xray_pooled = self.agpl_xray(xray_features)
        
        return mri_pooled, xray_pooled
