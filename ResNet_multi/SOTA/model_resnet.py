import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, version=18, num_classes=1):
        super(ResNetModel, self).__init__()
        
        if version == 10:
            self.model = models.resnet18()
            self.model.layer4 = nn.Identity()  # Remove deeper layers
        elif version == 18:
            self.model = models.resnet18()
        elif version == 34:
            self.model = models.resnet34()
        elif version == 50:
            self.model = models.resnet50()
        else:
            raise ValueError("Unsupported ResNet version")

        # Modify first layer for medical imaging
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, _):  # Ignore xray input
        return self.model(x.squeeze(1))  # Remove single depth dim
