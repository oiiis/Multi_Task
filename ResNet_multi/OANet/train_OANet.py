import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to sys.path

from data.data_process import MedicalImageDataset
from model import OANet  # Import your OANet model

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths
mri_paths = [
    "/Users/jiangwengyao/Desktop/ResNet_multi/data/MRI_Xray/MOST_MRI",
    "/Users/jiangwengyao/Desktop/ResNet_multi/data/MRI_Xray/OAI_MRI",
    "/Users/jiangwengyao/Desktop/ResNet_multi/data/MRI_Xray/SkI1Data"
]

xray_paths = [
    "/Users/jiangwengyao/Desktop/ResNet_multi/data/MRI_Xray/MOST_Xray",
    "/Users/jiangwengyao/Desktop/ResNet_multi/data/MRI_Xray/Digital_Xray",
    "/Users/jiangwengyao/Desktop/ResNet_multi/data/MRI_Xray/KOSGD"
]

# Training loop wrapped inside "if __name__ == '__main__'" to avoid multiprocessing issues on MacOS
if __name__ == "__main__":
    # Load dataset
    batch_size = 2
    dataset = MedicalImageDataset(mri_paths, xray_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Set num_workers=0

    # Initialize model
    model = OANet().to(device)

    # Define loss function & optimizer
    criterion = nn.BCEWithLogitsLoss()  # Use for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training settings
    num_epochs = 1
    save_path = "OANet_trained.pth"

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            if batch is None:  # Skip empty batches
                continue

            mri_input, xray_input = batch["mri"].to(device), batch["xray"].to(device)
            labels = torch.randint(0, 2, (mri_input.size(0), 1), dtype=torch.float).to(device)  # Dummy labels

            optimizer.zero_grad()
            outputs = model(mri_input, xray_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), save_path)
    print(f"Training completed! Model saved as {save_path}")
