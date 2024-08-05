import torch
import torch.nn as nn
from ResNet_18 import ResNet18Encoder3D
from segmentation_branch import SegmentationBranch3D

class SegmentationModel(nn.Module):
    def __init__(self, input_channels, num_segmentation_classes):
        super(SegmentationModel, self).__init__()
        self.shared_encoder = ResNet18Encoder3D(input_channels=input_channels)
        self.segmentation_branch = SegmentationBranch3D(512, num_segmentation_classes)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Invalid values found in input images")

        features, encoder_outputs = self.shared_encoder(x)
        
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError("Invalid values found in encoder output")

        seg_output = self.segmentation_branch(features, encoder_outputs)
        
        if torch.isnan(seg_output).any() or torch.isinf(seg_output).any():
            raise ValueError("Invalid values found in segmentation branch output")

        return seg_output
