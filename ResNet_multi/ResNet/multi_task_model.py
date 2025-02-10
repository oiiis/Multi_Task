import torch
import torch.nn as nn
from ResNet_18 import ResNet18Encoder3D
from segmentation_branch import SegmentationBranch3D
from classification_branch import ClassificationBranch3D

def clip_values(tensor, min_value=-1e5, max_value=1e5):
    tensor = torch.clamp(tensor, min=min_value, max=max_value)
    return tensor

class MultiTaskModel(nn.Module):
    def __init__(self, input_channels, num_segmentation_classes, num_classification_classes):
        super(MultiTaskModel, self).__init__()
        self.shared_encoder = ResNet18Encoder3D(input_channels=input_channels)
        self.segmentation_branch = SegmentationBranch3D(512, num_segmentation_classes)
        self.classification_branch = ClassificationBranch3D(512, num_classification_classes)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Invalid values found in input images")
        
        print("Input images passed initial check.")

        x = clip_values(x)
        features, encoder_outputs = self.shared_encoder(x)
        
        features = clip_values(features)
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"Encoder output has NaN: {torch.isnan(features).any()}, Inf: {torch.isinf(features).any()}")
            print(f"Encoder output min: {features.min()}, max: {features.max()}, mean: {features.mean()}")
            raise ValueError("Invalid values found in encoder output")
        
        print("Encoder outputs passed check.")

        seg_output = self.segmentation_branch(features, encoder_outputs)
        
        seg_output = clip_values(seg_output)
        if torch.isnan(seg_output).any() or torch.isinf(seg_output).any():
            print(f"Segmentation branch output has NaN: {torch.isnan(seg_output).any()}, Inf: {torch.isinf(seg_output).any()}")
            print(f"Segmentation branch output min: {seg_output.min()}, max: {seg_output.max()}, mean: {seg_output.mean()}")
            raise ValueError("Invalid values found in segmentation branch output")
        
        print("Segmentation branch output passed check.")

        class_output = self.classification_branch(features)
        
        class_output = clip_values(class_output)
        if torch.isnan(class_output).any() or torch.isinf(class_output).any():
            print(f"Classification branch output has NaN: {torch.isnan(class_output).any()}, Inf: {torch.isinf(class_output).any()}")
            print(f"Classification branch output min: {class_output.min()}, max: {class_output.max()}, mean: {class_output.mean()}")
            raise ValueError("Invalid values found in classification branch output")

        print("Classification branch output passed check.")

        return seg_output, class_output
