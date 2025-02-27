import torch
import torch.nn as nn
import torchvision.models as models

class SHN(nn.Module):
    def __init__(self, num_classes=1):
        super(SHN, self).__init__()

        # CNN-based feature extractor
        self.cnn_extractor = models.resnet18(pretrained=True)
        self.cnn_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.cnn_extractor.fc.in_features
        self.cnn_extractor.fc = nn.Identity()  # Remove FC layer

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Fully connected output
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, _):
        x = self.cnn_extractor(x.squeeze(1))  # Extract CNN features
        x = x.unsqueeze(0)  # Add sequence dimension for Transformer
        x = self.transformer(x)
        return self.fc(x.mean(dim=0))

# Test
if __name__ == "__main__":
    model = SHN()
    mri_input = torch.randn(2, 1, 256, 256)
    output = model(mri_input, None)
    print("SHN Model Output Shape:", output.shape)
