import torch
import torch.nn as nn
import torchvision.models as models

class ImageNetModel(nn.Module):
    """ Pretrained ImageNet model (ResNet50) for feature extraction. """
    def __init__(self, model_name="resnet50", pretrained=True):
        super(ImageNetModel, self).__init__()

        # Select ResNet variant
        if model_name == "resnet18":
            self.model = models.resnet18(weights="IMAGENET1K_V1")
        elif model_name == "resnet34":
            self.model = models.resnet34(weights="IMAGENET1K_V1")
        elif model_name == "resnet50":
            self.model = models.resnet50(weights="IMAGENET1K_V1")
        else:
            raise ValueError("Unsupported ResNet model!")

        # Modify first conv layer to accept 1-channel input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove fully connected layers (for feature extraction)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, mri_input, xray_input):
        """
        Process MRI and X-ray inputs through ResNet.
        """
        # Convert MRI from 3D to 2D
        mri_slices = torch.mean(mri_input, dim=2)  # Average along the depth dimension

        # Ensure correct shape
        if mri_slices.shape[1] != 1:
            raise ValueError(f"Unexpected MRI shape: {mri_slices.shape}")

        # Pass through ImageNet model
        mri_features = self.model(mri_slices)
        xray_features = self.model(xray_input)

        return (mri_features + xray_features) / 2  # Simple feature fusion
