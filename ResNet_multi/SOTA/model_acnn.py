import torch
import torch.nn as nn

class ACNN(nn.Module):
    def __init__(self, num_classes=1):
        super(ACNN, self).__init__()
        self.conv1 = nn.Conv2d(160, 32, 3, padding=1)  # Expecting 160 input channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.attention = nn.Conv2d(64, 64, kernel_size=1)

        # **Fix: Projection layer to match expected feature size (2048 channels)**
        self.projection = nn.Conv2d(64, 2048, kernel_size=1)  # Project 64 -> 2048

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, _):
        if x.dim() == 5:  
            x = x.squeeze(1)  
            print(f"Adjusted ACNN input shape: {x.shape}")

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.sigmoid(self.attention(x)) * x  

        x = self.projection(x)  # **Project to 2048 channels for compatibility**

        return x  

# Test
if __name__ == "__main__":
    model = ACNN()
    mri_input = torch.randn(2, 1, 160, 384, 384)
    output = model(mri_input, None)
    print("ACNN Model Output Shape:", output.shape)  # Should be [batch_size, 2048, H, W]