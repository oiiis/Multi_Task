import torch
import torch.nn as nn

class BEASM(nn.Module):
    def __init__(self, num_classes=1, mode="fully"):
        super(BEASM, self).__init__()
        self.mode = mode

        # Ensure correct feature depth for segmentation head
        self.conv1 = nn.Conv2d(160, 512, kernel_size=3, padding=1)  # Increased to 512
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1)  # Ensure it matches segmentation head

        # 1x1 convolution to match the expected feature depth
        self.final_conv = nn.Conv2d(2048, 2048, kernel_size=1)

        self.dropout = nn.Dropout(p=0.5) if mode == "fully" else nn.Dropout(p=0.2)

        self.fc = nn.Linear(2048, num_classes)  # Keep classification layer

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = self.final_conv(x)  # Ensure the output has 2048 channels
        x = self.dropout(x)

        return x  # Now compatible with segmentation head

# Test
if __name__ == "__main__":
    model = BEASM(mode="fully")
    mri_input = torch.randn(2, 160, 256, 256)  # Simulated MRI batch
    output = model(mri_input)
    print("BEASM Output Shape:", output.shape)  # Expected [2, 2048, 256, 256]