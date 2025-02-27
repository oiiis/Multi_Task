import torch
import torch.nn as nn

class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes=1):
        super(UNetPlusPlus, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 3, padding=1)
        )

    def forward(self, x, _):  # Ignore xray input
        x = self.encoder(x.squeeze(1))
        return self.decoder(x)
