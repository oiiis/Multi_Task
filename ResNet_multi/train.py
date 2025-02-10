import torch
from OANet.dual_sharedlayer import DualSharedLayer
from OANet.fusion_layer import FusionLayer
from utils import load_model


def initialize_multitask_model(device):
    # Load the dual-stream networks
    model_mri, model_xray = DualSharedLayer().to(device), DualSharedLayer().to(device)
    fusion_layer = FusionLayer(512, 10).to(device)  # Assuming these dimensions match your pre-trained model

    # Load the pre-trained model weights
    load_model(model_mri, 'oanet_pretrained_mri.pth')
    load_model(model_xray, 'oanet_pretrained_xray.pth')

    return model_mri, model_xray, fusion_layer

def train_multitask_model(model_mri, model_xray, fusion_layer, train_loader, optimizer, device):
    model_mri.train()
    model_xray.train()
    fusion_layer.train()

    for epoch in range(num_epochs):
        for data in train_loader:
            inputs_mri, inputs_xray, seg_labels, class_labels, risk_labels = data
            inputs_mri = inputs_mri.to(device)
            inputs_xray = inputs_xray.to(device)

            optimizer.zero_grad()

            features_mri = model_mri(inputs_mri)
            features_xray = model_xray(inputs_xray)
            combined_features = fusion_layer(features_mri, features_xray)

            # Assuming you have loss functions defined for segmentation, classification, and risk assessment
            seg_loss = seg_criterion(combined_features, seg_labels)
            class_loss = class_criterion(combined_features, class_labels)
            risk_loss = risk_criterion(combined_features, risk_labels)

            total_loss = seg_loss + class_loss + risk_loss
            total_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {total_loss.item()}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_mri, model_xray, fusion_layer = initialize_multitask_model(device)
    optimizer = torch.optim.Adam(list(model_mri.parameters()) + list(model_xray.parameters()) + list(fusion_layer.parameters()), lr=0.001)

    # Assuming train_loader is defined and loads your data appropriately
    train_multitask_model(model_mri, model_xray, fusion_layer, train_loader, optimizer, device)

if __name__ == "__main__":
    main()
