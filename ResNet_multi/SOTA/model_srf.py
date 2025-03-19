import torch
import torch.nn as nn

class SRF(nn.Module):
    def __init__(self):
        super(SRF, self).__init__()

        # Reduce input to a single-channel
        self.conv1 = nn.Conv2d(1, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1)  # Match segmentation head

        self.final_conv = nn.Conv2d(2048, 2048, kernel_size=1)  # Ensure compatibility

    def forward(self, x, _):
        x = torch.mean(x, dim=1, keepdim=True)  # Convert 160 channels to single grayscale
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = self.final_conv(x)  # Ensure 2048 channels

        return x

# Test
if __name__ == "__main__":
    model = SRF()
    mri_input = torch.randn(2, 160, 256, 256)  # Simulated MRI batch
    output = model(mri_input, None)
    print("SRF Output Shape:", output.shape)  # Expected [2, 2048, 256, 256]