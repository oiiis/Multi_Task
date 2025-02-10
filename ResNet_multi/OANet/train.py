import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dual_sharedlayer import DualResNet
from fusion_layer import FusionLayer

# Dummy Data
mri_data = torch.randn(10, 1, 160, 256, 256)  # 3D MRI Data
xray_data = torch.randn(10, 1, 256, 256)  # 2D X-ray Data
labels = torch.randint(0, 2, (10,))  # Binary classification labels

# DataLoader
dataset = TensorDataset(mri_data, xray_data, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model
num_blocks = 3  # Number of residual blocks
dual_resnet = DualResNet(num_blocks)
fusion_layer = FusionLayer(64)

# Optimizer & Loss
optimizer = optim.Adam(list(dual_resnet.parameters()) + list(fusion_layer.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(5):
    for mri, xray, label in dataloader:
        optimizer.zero_grad()
        mri_features, xray_features = dual_resnet(mri, xray)
        fused_features = fusion_layer(mri_features, xray_features)
        
        # Simple classification head (for demonstration)
        output = fused_features.mean(dim=[2, 3, 4])  # Global pooling
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
