import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter  # TensorBoard logging

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to sys.path

from data.data_process import MedicalImageDataset
from model import OANet  # Import your OANet model

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths
# Change according to real terminal path
mri_paths = [
    "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/MOST_MRI",
    "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/OAI_MRI",
    "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/SkI1Data"
]

xray_paths = [
    "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/MOST_Xray",
    "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/Digital_Xray",
    "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/KOSGD"
]

# Load dataset
batch_size = 2
dataset = MedicalImageDataset(mri_paths, xray_paths)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = OANet().to(device)

# Define loss function & optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TensorBoard SummaryWriter
writer = SummaryWriter(log_dir="runs/OANet_training")

# Training settings
num_epochs = 5
save_path = "OANet_trained.pth"

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

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

        # Convert logits to binary predictions
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        labels_np = labels.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels_np)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {running_loss:.4f} | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")

    # Log metrics to TensorBoard
    writer.add_scalar("Loss/train", running_loss, epoch)
    writer.add_scalar("Accuracy/train", acc, epoch)
    writer.add_scalar("Precision/train", precision, epoch)
    writer.add_scalar("Recall/train", recall, epoch)
    writer.add_scalar("F1-score/train", f1, epoch)

# Save trained model
torch.save(model.state_dict(), save_path)
print(f"Training completed! Model saved as {save_path}")

# Close TensorBoard writer
writer.close()
