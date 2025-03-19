import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import time
import sys
import os
import warnings

# Suppress enable_nested_tensor warning
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

# Add necessary directories to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../OANet')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MTL')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../SOTA')))

from OANet.model import OANet
from MTL.decoder import SegmentationHead, ClassificationHead, RiskPredictionHead
from data.data_process import MedicalImageDatasetWithLabel

from SOTA.model_imagenet import ImageNetModel
from SOTA.model_beasm import BEASM
from SOTA.model_srf import SRF
from SOTA.model_shn import SHN
from SOTA.model_resnet import ResNet
from SOTA.model_acnn import ACNN
from SOTA.model_unetplusplus import UNetPlusPlus

# Define models dictionary
models = {
    "ImageNet": ImageNetModel(),
    "BEASM-Fully": BEASM(mode="fully"),
    "BEASM-Semi": BEASM(mode="semi"),
    "SRF": SRF(),
    "SHN": SHN(),
    "ACNN": ACNN(),
    "U-Net++": UNetPlusPlus(),
    "OANet": OANet()
}

# Allow different ResNet versions
resnet_versions = [10, 18, 34, 50, 101, 152]
for version in resnet_versions:
    models[f"ResNet{version}"] = ResNet(version=version)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset paths
mri_dirs = ["/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/OAI_MRI"]
xray_dirs = ["/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/Digital_Xray"]

mri_label_path = "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/label/OAI_MRI.xlsx"
xray_label_path = "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/label/OAI_Xray.xlsx"

# Load dataset
batch_size = 4
dataset = MedicalImageDatasetWithLabel(mri_dirs, xray_dirs, mri_label_path, xray_label_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Get model selection from user input
if len(sys.argv) < 2:
    print("\nAvailable models:")
    for model_name in models.keys():
        print(f" - {model_name}")
    print("\nUsage: python train_compare.py <ModelName>")
    sys.exit(1)

selected_model_name = sys.argv[1]

if selected_model_name not in models:
    print(f"Error: Model '{selected_model_name}' not found.")
    print("Available models:", list(models.keys()))
    sys.exit(1)

# Set up results directory
results_dir = os.path.join(os.path.dirname(__file__), "results", selected_model_name)
os.makedirs(results_dir, exist_ok=True)

# Train the selected model
model = models[selected_model_name].to(device)
print(f"\n=== Training {selected_model_name} on MTL tasks ===")

# Initialize task heads
seg_head = SegmentationHead(512).to(device)
class_head = ClassificationHead(512, num_classes=2).to(device)
risk_head = RiskPredictionHead(512, num_classes=2).to(device)

# Freeze encoder initially for transfer learning
for param in model.parameters():
    param.requires_grad = False

# Define losses
seg_criterion = nn.BCEWithLogitsLoss()
class_criterion = nn.CrossEntropyLoss()
risk_criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(list(model.parameters()) +
                       list(seg_head.parameters()) +
                       list(class_head.parameters()) +
                       list(risk_head.parameters()), lr=0.001)

num_epochs = 10
save_path = os.path.join(results_dir, f"{selected_model_name}.pth")

# Store loss & accuracy history
loss_history = []
accuracy_results = []

# Train the model
for epoch in range(num_epochs):
    model.train()
    seg_head.train()
    class_head.train()
    risk_head.train()

    running_loss = 0.0
    total_correct_oa = 0
    total_correct_risk = 0
    total_correct_seg = 0
    total_samples = 0

    start_time = time.time()

    for batch in dataloader:
        if batch is None:
            continue

        mri_input, xray_input = batch["mri"].to(device), batch["xray"].to(device)

        # Handle ResNet input shape issue
        if selected_model_name.startswith("ResNet"):
            if mri_input.shape[2] != 1:  # If MRI has multiple slices (e.g., 160), reduce to 1
                mri_input = torch.mean(mri_input, dim=2, keepdim=True)  # Average across depth dim

        # Get model output
        model_output = model(mri_input, xray_input)

        if isinstance(model_output, tuple) and len(model_output) == 2:
            features_seg, features_class = model_output
        else:
            features_seg, features_class = None, model_output  # ResNet only outputs classification features

        seg_labels = torch.randint(0, 2, (mri_input.size(0), 1, 256, 256), dtype=torch.float).to(device)
        class_labels = batch["oa_label"].to(device)
        risk_labels = batch["risk_label"].to(device)

        seg_output = seg_head(features_seg) if features_seg is not None else torch.zeros_like(seg_labels)
        class_output = class_head(features_class.view(features_class.size(0), -1))
        risk_output = risk_head(features_class.view(features_class.size(0), -1))

        # Compute loss
        seg_loss = seg_criterion(seg_output, seg_labels)
        class_loss = class_criterion(class_output, class_labels)
        risk_loss = risk_criterion(risk_output, risk_labels)

        total_loss = seg_loss + class_loss + risk_loss
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

        # Compute accuracy
        _, predicted_oa = torch.max(class_output, 1)
        _, predicted_risk = torch.max(risk_output, 1)

        total_correct_oa += (predicted_oa == class_labels).sum().item()
        total_correct_risk += (predicted_risk == risk_labels).sum().item()
        total_correct_seg += (torch.round(seg_output) == seg_labels).sum().item()
        total_samples += class_labels.size(0)

    accuracy_oa = total_correct_oa / total_samples
    accuracy_risk = total_correct_risk / total_samples
    accuracy_seg = total_correct_seg / (total_samples * 256 * 256)

    epoch_time = time.time() - start_time
    loss_history.append(running_loss)
    accuracy_results.append([epoch+1, accuracy_oa, accuracy_risk, accuracy_seg])

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, OA Accuracy: {accuracy_oa:.2f}, Risk Accuracy: {accuracy_risk:.2f}, Segmentation Accuracy: {accuracy_seg:.4f}, Time: {epoch_time:.2f}s")

# Save accuracy results
pd.DataFrame(accuracy_results, columns=["Epoch", "OA_Accuracy", "Risk_Accuracy", "Segmentation_Accuracy"]).to_csv(os.path.join(results_dir, "accuracy_results.csv"), index=False)

# Save loss history
loss_df = pd.DataFrame(loss_history, columns=["Loss"])
loss_csv_path = os.path.join(results_dir, "loss_history.csv")
loss_df.to_csv(loss_csv_path, index=False)
print(f"Loss history saved for {selected_model_name} at {loss_csv_path}")

torch.save(model.state_dict(), save_path)
print(f"Training completed for {selected_model_name}! Model saved as {save_path}")