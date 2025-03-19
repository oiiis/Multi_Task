import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import itertools
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to sys.path

from data.data_process import MedicalImageDatasetWithoutLabel
from OANet.model import OANet  

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths
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
dataset = MedicalImageDatasetWithoutLabel(mri_paths, xray_paths)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Experimental configurations
fusion_strategies = ["early", "late"]
fusion_weights = [0.3, 0.5, 0.7]  # Alpha values for weighted fusion
pooling_types = ["traditional", "equal", "adaptive"]

# Training settings
num_epochs = 1  # Short training per experiment
results_dir = "results/OANet"
os.makedirs(results_dir, exist_ok=True)

# Store experiment results
experiment_results = []
log_file = open(os.path.join(results_dir, "debug_log.txt"), "w")

# Iterate through all combinations of settings
for fusion, alpha, pooling in itertools.product(fusion_strategies, fusion_weights, pooling_types):
    print(f"\n=== Running Experiment: Fusion={fusion}, Fusion α={alpha}, Pooling={pooling} ===\n")
    log_file.write(f"\n=== Running Experiment: Fusion={fusion}, Fusion α={alpha}, Pooling={pooling} ===\n")

    # Initialize model
    model = OANet(fusion_strategy=fusion, fusion_alpha=alpha, pooling_type=pooling).to(device)

    # Define loss function & optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=f"runs/OANet_{fusion}_alpha{alpha}_pooling{pooling}")

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

            print(f"Input MRI shape: {mri_input.shape}")
            print(f"Input X-ray shape: {xray_input.shape}")
            log_file.write(f"Input MRI shape: {mri_input.shape}\n")
            log_file.write(f"Input X-ray shape: {xray_input.shape}\n")

            labels = torch.randint(0, 2, (mri_input.size(0), 1), dtype=torch.float).to(device)  # Dummy labels

            optimizer.zero_grad()
            print(f"Before model call - MRI input: {mri_input.shape}, X-ray input: {xray_input.shape}")
            outputs = model(mri_input, xray_input)
            print(f"Model output shape: {outputs.shape}")

            print(f"Output shape before FC: {outputs.shape}")
            log_file.write(f"Output shape before FC: {outputs.shape}\n")

            # Ensure correct shape before fc layer
            if len(outputs.shape) > 2:
                outputs = outputs.view(outputs.size(0), -1)  # Flatten feature map

            features = outputs.view(outputs.size(0), -1)  # Reshape features before fc layer
            print(f"Features shape before FC: {features.shape}")  
            log_file.write(f"Features shape before FC: {features.shape}\n")

            # Dynamically adjust FC layer input size
            expected_fc_input_size = model.fc.in_features
            actual_fc_input_size = features.shape[1]

            if actual_fc_input_size != expected_fc_input_size:
                print(f"Shape Mismatch! Expected {expected_fc_input_size}, but got {actual_fc_input_size}")
                log_file.write(f"Shape Mismatch! Expected {expected_fc_input_size}, but got {actual_fc_input_size}\n")

                # Fix the shape mismatch
                model.fc = nn.Linear(actual_fc_input_size, 1).to(device)

            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Convert logits to binary predictions
            preds = (torch.sigmoid(features) > 0.5).cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels_np)

        # Compute metrics
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {running_loss:.4f} | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f} | Pooling: {pooling}")
        log_file.write(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {running_loss:.4f} | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f} | Pooling: {pooling}\n")

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", running_loss, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)
        writer.add_scalar("Precision/train", precision, epoch)
        writer.add_scalar("Recall/train", recall, epoch)
        writer.add_scalar("F1-score/train", f1, epoch)

    # Save trained model
    model_save_path = os.path.join(results_dir, f"OANet_{fusion}_alpha{alpha}_pooling{pooling}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Experiment completed! Model saved as {model_save_path}")
    log_file.write(f"Experiment completed! Model saved as {model_save_path}\n")

    # Store results
    experiment_results.append({
        "Fusion Strategy": fusion,
        "Fusion Alpha": alpha,
        "Pooling Type": pooling,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    # Close TensorBoard writer
    writer.close()

log_file.close()