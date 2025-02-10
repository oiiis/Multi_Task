import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionLayer(nn.Module):
    """Fusion Layer for Combining 3D MRI and 2D X-ray Features"""
    def __init__(self, channels):
        super(FusionLayer, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable weight for fusion
        self.conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, mri_features, xray_features):
        # Ensure dimensions match (convert X-ray to match MRI features)
        xray_features = xray_features.unsqueeze(2)  # Add depth dimension
        fused = self.alpha * mri_features + (1 - self.alpha) * xray_features
        fused = self.relu(self.bn(self.conv(fused)))
        return fused
