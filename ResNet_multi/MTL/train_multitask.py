import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os

# Add parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from OANet.model import OANet
from MTL.decoder import SegmentationHead, ClassificationHead, RiskPredictionHead
from data.data_process import MedicalImageDatasetWithLabel

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset paths
mri_paths = ["/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/OAI_MRI"]
xray_paths = ["/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/Digital_Xray"]

mri_label_path = "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/label/OAI_MRI.xlsx"
xray_label_path = "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/label/OAI_Xray.xlsx"

# Load dataset
batch_size = 4
dataset = MedicalImageDatasetWithLabel(mri_paths, xray_paths, mri_label_path, xray_label_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
encoder = OANet().to(device)
seg_head = SegmentationHead(256).to(device)
class_head = ClassificationHead(256).to(device)
risk_head = RiskPredictionHead(256).to(device)

# Freeze encoder for initial training (transfer learning)
for param in encoder.parameters():
    param.requires_grad = False

# Loss functions
seg_criterion = nn.BCEWithLogitsLoss()
class_criterion = nn.CrossEntropyLoss()
risk_criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    list(encoder.parameters()) +
    list(seg_head.parameters()) +
    list(class_head.parameters()) +
    list(risk_head.parameters()), lr=0.001
)

# Training settings
num_epochs = 10
save_path = "MultiTask_OANet.pth"

# Training loop
for epoch in range(num_epochs):
    encoder.train()
    seg_head.train()
    class_head.train()
    risk_head.train()

    running_loss = 0.0
    total_correct_oa = 0
    total_correct_risk = 0
    total_samples = 0

    # Segmentation evaluation metrics
    total_intersection = 0
    total_union = 0
    total_dice_score = 0
    num_batches = 0

    for batch in dataloader:
        if batch is None:
            continue

        mri_input, xray_input = batch["mri"].to(device), batch["xray"].to(device)
        seg_labels = torch.randint(0, 2, (mri_input.size(0), 1, 256, 256), dtype=torch.float).to(device)
        class_labels = batch["oa_label"].to(device)
        risk_labels = batch["risk_label"].to(device)

        optimizer.zero_grad()
        features = encoder(mri_input, xray_input)

        # Ensure the correct shape for segmentation head
        if features.dim() == 2:
            features = features.unsqueeze(-1).unsqueeze(-1)  # Add height and width as (1,1)

        # Expand channels if needed
        if features.shape[1] == 1:
            features = features.expand(-1, 256, -1, -1)  # Expand to (B, 256, 1, 1)

        # Forward pass through task heads
        seg_output = seg_head(features)
        class_output = class_head(features.view(features.size(0), -1))  # Flatten for FC layer
        risk_output = risk_head(features.view(features.size(0), -1))

        # Compute loss for each task
        seg_loss = seg_criterion(seg_output, seg_labels)
        class_loss = class_criterion(class_output, class_labels)
        risk_loss = risk_criterion(risk_output, risk_labels)

        total_loss = seg_loss + class_loss + risk_loss
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

        # Compute accuracy for classification tasks
        _, predicted_oa = torch.max(class_output, 1)
        _, predicted_risk = torch.max(risk_output, 1)

        total_correct_oa += (predicted_oa == class_labels).sum().item()
        total_correct_risk += (predicted_risk == risk_labels).sum().item()
        total_samples += class_labels.size(0)

        # Compute Segmentation IoU and Dice Score
        predicted_seg = (torch.sigmoid(seg_output) > 0.5).float()  # Binarization
        intersection = (predicted_seg * seg_labels).sum()
        union = predicted_seg.sum() + seg_labels.sum() - intersection
        dice_score = (2. * intersection) / (predicted_seg.sum() + seg_labels.sum() + 1e-6)  # Avoid zero division

        total_intersection += intersection.item()
        total_union += union.item()
        total_dice_score += dice_score.item()
        num_batches += 1

    # Compute final metrics
    accuracy_oa = total_correct_oa / total_samples
    accuracy_risk = total_correct_risk / total_samples
    iou_seg = total_intersection / (total_union + 1e-6)  # Avoid zero division
    avg_dice_score = total_dice_score / num_batches

    # Print results
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, "
          f"OA Accuracy: {accuracy_oa:.2f}, Risk Accuracy: {accuracy_risk:.2f}, "
          f"Segmentation IoU: {iou_seg:.4f}, Dice Score: {avg_dice_score:.4f}")

    # Unfreeze encoder for fine-tuning after 5 epochs
    if epoch == 5:
        for param in encoder.parameters():
            param.requires_grad = True

# Save trained model
torch.save({
    "encoder": encoder.state_dict(),
    "seg_head": seg_head.state_dict(),
    "class_head": class_head.state_dict(),
    "risk_head": risk_head.state_dict()
}, save_path)

print(f"Training completed! Model saved as {save_path}")