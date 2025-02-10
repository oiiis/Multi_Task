import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGlobalPoolingLayer(nn.Module):
    """Adaptive Global Pooling Layer (AGPL) with learnable weights."""
    def __init__(self, num_blocks, feature_dim):
        super(AdaptiveGlobalPoolingLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(num_blocks))  # Learnable weights for each block
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # Applies GAP across depth, height, and width

    def forward(self, feature_maps):
        """
        Args:
            feature_maps (list of tensors): A list of feature maps from different residual blocks.
        Returns:
            torch.Tensor: Aggregated feature representation.
        """
        pooled_features = [self.global_avg_pool(f) for f in feature_maps]  # Apply GAP to each block
        pooled_features = torch.stack(pooled_features, dim=-1)  # Stack across a new dimension

        # Weight each feature map dynamically
        weighted_features = self.alpha.view(1, 1, 1, 1, -1) * pooled_features
        output = weighted_features.sum(dim=-1)  # Sum across blocks
        
        return output.squeeze(-1)  # Remove unnecessary dimensions
