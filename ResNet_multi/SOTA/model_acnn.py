import torch
import torch.nn as nn

class ACNN(nn.Module):
    def __init__(self, num_classes=1):
        super(ACNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.attention = nn.Conv2d(64, 64, kernel_size=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, _):
        x = self.conv1(x.squeeze(1))
        x = torch.relu(self.conv2(x))
        x = torch.sigmoid(self.attention(x)) * x
        return self.fc(x.mean(dim=[2, 3]))  # Global average pooling
