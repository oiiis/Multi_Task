import torch
import torch.nn as nn

class BEASM(nn.Module):
    def __init__(self, num_classes=1, mode="fully"):
        super(BEASM, self).__init__()
        self.mode = mode

        # Base convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Bayesian Dropout for uncertainty estimation
        self.dropout = nn.Dropout(p=0.5) if mode == "fully" else nn.Dropout(p=0.2)

        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, _):
        x = torch.relu(self.conv1(x.squeeze(1)))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.dropout(x.mean(dim=[2, 3]))  # Global pooling + dropout
        return self.fc(x)

# Test
if __name__ == "__main__":
    model = BEASM(mode="fully")
    mri_input = torch.randn(2, 1, 256, 256)
    output = model(mri_input, None)
    print("BEASM Model Output Shape:", output.shape)
