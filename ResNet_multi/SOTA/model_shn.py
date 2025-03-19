import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SHN(nn.Module):
    """
    SHN Model:
    - Uses ResNet18 for feature extraction.
    - Applies a Transformer Encoder.
    - Outputs features for segmentation, classification, and risk prediction.
    """

    def __init__(self, num_classes=2, reduce_method="mean"):
        super(SHN, self).__init__()

        self.reduce_method = reduce_method  # Choose depth reduction method

        # Depth reduction if needed (for MRI 3D input)
        if self.reduce_method == "conv":
            self.depth_reducer = nn.Conv3d(1, 1, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        elif self.reduce_method == "mean":
            self.depth_reducer = lambda x: x.mean(dim=2, keepdim=True)
        elif self.reduce_method == "slice":
            self.depth_reducer = lambda x: x[:, :, x.shape[2] // 2, :, :]

        # CNN-based feature extractor (ResNet18)
        self.cnn_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.cnn_extractor.fc.in_features
        self.cnn_extractor.fc = nn.Identity()  # Remove FC layer

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Fully connected output for classification and risk prediction
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, _):
        """
        Forward pass:
        - x: MRI input (B, C, D, H, W) → Needs to be converted to (B, C, H, W)
        - _: Placeholder for compatibility
        """
        print(f"Original input shape: {x.shape}")  # Debugging print

        # Reduce depth dimension to 1 for CNN input
        x = self.depth_reducer(x)  # Use selected reduction method
        x = x.squeeze(2)  # Remove depth dimension

        print(f"Input shape before CNN: {x.shape}")  # Debugging print

        x = self.cnn_extractor(x)  # CNN feature extraction

        print(f"Feature shape after CNN: {x.shape}")  # Debugging print

        # Ensure features for segmentation retain spatial dimensions
        x_seg = x.unsqueeze(-1).unsqueeze(-1)  # Expand to (B, C, 1, 1) for segmentation

        x = x.unsqueeze(0)  # Add sequence dimension for Transformer
        x = self.transformer(x)  # Transformer encoding
        x = x.mean(dim=0)  # Reduce sequence dimension

        print(f"Feature shape before FC: {x.shape}")  # Debugging print

        return x_seg, self.fc(x)  # Return separate tensors for segmentation & classification