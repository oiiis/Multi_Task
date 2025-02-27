import torch
import torch.nn as nn

class SRF(nn.Module):
    def __init__(self, num_classes=1):
        super(SRF, self).__init__()

        # Super-resolution layer
        self.upscale = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, _):
        x = self.upscale(x.squeeze(1))  # Super-resolution
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return self.fc(x.mean(dim=[2, 3]))  # Global pooling

# Test
if __name__ == "__main__":
    model = SRF()
    mri_input = torch.randn(2, 1, 128, 128)
    output = model(mri_input, None)
    print("SRF Model Output Shape:", output.shape)
