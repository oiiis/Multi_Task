import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Encoder3D(nn.Module):
    def __init__(self, input_channels=3):
        super(ResNet18Encoder3D, self).__init__()
        resnet2d = resnet18(pretrained=True)

        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer3d(resnet2d.layer1)
        self.layer2 = self._make_layer3d(resnet2d.layer2, stride=2)
        self.layer3 = self._make_layer3d(resnet2d.layer3, stride=2)
        self.layer4 = self._make_layer3d(resnet2d.layer4, stride=2)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer3d(self, layer2d, stride=1):
        layers = []
        for block in layer2d:
            layers.append(self._make_block3d(block, stride))
        return nn.Sequential(*layers)

    def _make_block3d(self, block, stride):
        conv1 = nn.Conv3d(block.conv1.in_channels, block.conv1.out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        bn1 = nn.BatchNorm3d(block.bn1.num_features)
        conv2 = nn.Conv3d(block.conv2.in_channels, block.conv2.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm3d(block.bn2.num_features)
        return nn.Sequential(conv1, bn1, nn.ReLU(inplace=True), conv2, bn2, nn.ReLU(inplace=True))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 添加裁剪和异常检查
        x = torch.clamp(x, min=-1e5, max=1e5)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Invalid values after conv1/bn1/relu/maxpool: NaN: {torch.isnan(x).any()}, Inf: {torch.isinf(x).any()}")
            raise ValueError("Invalid values found after initial convolution block")

        encoder_outputs = []
        x = self.layer1(x)
        encoder_outputs.append(x)
        print(f"Output after layer1: {x.shape}, NaN: {torch.isnan(x).any()}, Inf: {torch.isinf(x).any()}")
        x = torch.clamp(x, min=-1e5, max=1e5)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Invalid values after layer1: NaN: {torch.isnan(x).any()}, Inf: {torch.isinf(x).any()}")
            raise ValueError("Invalid values found after layer1")

        x = self.layer2(x)
        encoder_outputs.append(x)
        print(f"Output after layer2: {x.shape}, NaN: {torch.isnan(x).any()}, Inf: {torch.isinf(x).any()}")
        x = torch.clamp(x, min=-1e5, max=1e5)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Invalid values after layer2: NaN: {torch.isnan(x).any()}, Inf: {torch.isinf(x).any()}")
            raise ValueError("Invalid values found after layer2")

        x = self.layer3(x)
        encoder_outputs.append(x)
        print(f"Output after layer3: {x.shape}, NaN: {torch.isnan(x).any()}, Inf: {torch.isinf(x).any()}")
        x = torch.clamp(x, min=-1e5, max=1e5)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Invalid values after layer3: NaN: {torch.isnan(x).any()}, Inf: {torch.isinf(x).any()}")
            raise ValueError("Invalid values found after layer3")

        x = self.layer4(x)
        encoder_outputs.append(x)
        print(f"Output after layer4: {x.shape}, NaN: {torch.isnan(x).any()}, Inf: {torch.isinf(x).any()}")
        x = torch.clamp(x, min=-1e5, max=1e5)
        if torch.isnan(x).any() or torch.isinf(x).any(): 
            print(f"Invalid values after layer4: NaN: {torch.isnan(x).any()}, Inf: {torch.isinf(x).any()}")
            raise ValueError("Invalid values found after layer4")

        return x, encoder_outputs
