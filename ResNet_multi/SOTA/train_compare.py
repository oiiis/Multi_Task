import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import time

import sys
import os


# Import models
# from model_imagenet import ImageNetModel
# from model_beasm import BEASM
# from model_srf import SRF
# from model_shn import SHN
# from model_resnet import ResNet
# from model_acnn import ACNN
# from model_unetplusplus import UNetPlusPlus

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to sys.path
print("PYTHONPATH Directories:\n", "\n".join(sys.path)) 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../OANet')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MTL')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../SOTA')))

from OANet.model import OANet
from MTL.decoder import SegmentationHead, ClassificationHead, RiskPredictionHead
from data.data_process import  MedicalImageDatasetWithLabel

from OANet.model import OANet  

from MTL.decoder import SegmentationHead, ClassificationHead, RiskPredictionHead
# from data_process import MedicalImageDatasetWithLabel

from SOTA.model_imagenet import ImageNetModel
from SOTA.model_beasm import BEASM
from SOTA.model_srf import SRF
from SOTA.model_shn import SHN
from SOTA.model_resnet import ResNet
from SOTA.model_acnn import ACNN
from SOTA.model_unetplusplus import UNetPlusPlus




# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset paths
mri_dirs = [
    "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/OAI_MRI",
]
xray_dirs = [
    "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/Digital_Xray"
]

mri_label_path = "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/label/OAI_MRI.xlsx"
xray_label_path = "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/label/OAI_Xray.xlsx"

# Load dataset
batch_size = 4
dataset = MedicalImageDatasetWithLabel(mri_dirs, xray_dirs, mri_label_path, xray_label_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define models to compare
models = {
    "ImageNet": ImageNetModel(),
    "BEASM-Fully": BEASM(mode="fully"),
    "BEASM-Semi": BEASM(mode="semi"),
    "SRF": SRF(),
    "SHN": SHN(),
    "ACNN": ACNN(),
    "U-Net++": UNetPlusPlus(),
    "OANet (ours)": OANet()
}

# Load pre-trained weights if available
for model_name, model in models.items():
    model.to(device)
    if os.path.exists(f"{model_name}_trained.pth"):
        model.load_state_dict(torch.load(f"{model_name}_trained.pth"))
        print(f"Loaded {model_name} pre-trained weights.")

# Train and evaluate each model
results = {}

for model_name, model in models.items():
    print(f"\n=== Training {model_name} on MTL tasks ===")
    
    # Initialize task heads
    seg_head = SegmentationHead(2048).to(device)
    class_head = ClassificationHead(2048, num_classes=2).to(device)  # Adjusted to 2048
    risk_head = RiskPredictionHead(2048, num_classes=2).to(device)


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
    save_path = f"{model_name}_MTL.pth"

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        seg_head.train()
        class_head.train()
        risk_head.train()

        running_loss = 0.0
        total_correct_oa = 0
        total_correct_risk = 0
        total_samples = 0

        start_time = time.time()

        for batch in dataloader:
            if batch is None:
                continue

            mri_input, xray_input = batch["mri"].to(device), batch["xray"].to(device)
            seg_labels = torch.randint(0, 2, (mri_input.size(0), 1, 256, 256), dtype=torch.float).to(device)
            class_labels = batch["oa_label"].to(device)
            risk_labels = batch["risk_label"].to(device)

            optimizer.zero_grad()
            features = model(mri_input, xray_input)

            seg_output = seg_head(features)
            class_output = class_head(features)
            risk_output = risk_head(features)

            # Compute loss for each task
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
            total_samples += class_labels.size(0)

        accuracy_oa = total_correct_oa / total_samples
        accuracy_risk = total_correct_risk / total_samples
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, OA Accuracy: {accuracy_oa:.2f}, Risk Accuracy: {accuracy_risk:.2f}, Time: {epoch_time:.2f}s")

        # Unfreeze encoder after 5 epochs for fine-tuning
        if epoch == 5:
            for param in model.parameters():
                param.requires_grad = True

    # Save trained model
    torch.save({
        "encoder": model.state_dict(),
        "seg_head": seg_head.state_dict(),
        "class_head": class_head.state_dict(),
        "risk_head": risk_head.state_dict()
    }, save_path)

    print(f"Training completed for {model_name}! Model saved as {save_path}")

    # Store evaluation results
    results[model_name] = {
        "OA Accuracy": accuracy_oa,
        "Risk Accuracy": accuracy_risk,
        "Final Loss": running_loss / num_epochs
    }

# Save results to CSV
df_results = pd.DataFrame.from_dict(results, orient="index")
df_results.to_csv("MTL_SOTA_comparison.csv")
print("Comparison results saved to MTL_SOTA_comparison.csv")
