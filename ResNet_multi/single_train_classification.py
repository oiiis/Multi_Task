import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import OAI_Dataset
from classification_branch import ClassificationBranch3D
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(outputs, targets, task_type='classification'):
    if task_type == 'classification':
        outputs = np.argmax(outputs, axis=1).flatten()
        targets = targets.flatten()

        pre = precision_score(targets, outputs, average='macro', zero_division=1)
        rec = recall_score(targets, outputs, average='macro', zero_division=1)
        f1 = f1_score(targets, outputs, average='macro', zero_division=1)
        return pre, rec, f1
    else:
        raise ValueError("Unknown task type")

image_dir = '/gpfs/work/aac/kewu23/projects/ResNet_multi/data/segmentation_masks_A'
label_file = '/gpfs/work/aac/kewu23/projects/ResNet_multi/data/id_part_a.xlsx'
input_channels = 1
num_classification_classes = 5
batch_size = 24
num_epochs = 2
learning_rate = 0.0001

limit =  120


# 数据加载器
train_dataset = OAI_Dataset(image_dir, label_file)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ClassificationBranch3D(input_channels, num_classification_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss_values = []
all_class_preds = []
all_class_targets = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for batch_idx, (images, _, class_labels) in enumerate(train_loader):
        if torch.isnan(images).any() or torch.isinf(images).any():
           print(f"Invalid values found in input images at batch {batch_idx + 1}")
           continue

        images = images.to(device)
        class_labels = class_labels.to(device)
        optimizer.zero_grad()

        try:
            class_outputs = model(images)
        except ValueError as e:
            print(f"Error in model forward pass at batch {batch_idx + 1}: {e}")
            continue

        clf_target = class_labels.long()
        loss = F.cross_entropy(class_outputs, clf_target)

        if torch.isnan(loss):
            print("NaN loss detected! Debug information:")
            print(f"images: {images}")
            print(f"class_labels: {class_labels}")
            print(f"class_outputs: {class_outputs}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_train_loss += loss.item()
        all_class_preds.append(class_outputs.cpu().detach().numpy())
        all_class_targets.append(class_labels.cpu().detach().numpy())

    average_train_loss = running_train_loss / len(train_loader)
    train_loss_values.append(average_train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}')

all_class_preds = np.concatenate(all_class_preds, axis=0)
all_class_targets = np.concatenate(all_class_targets, axis=0)

class_pre, class_rec, class_f1 = calculate_metrics(all_class_preds, all_class_targets, task_type='classification')

print(f'Classification - Precision: {class_pre:.4f}, Recall: {class_rec:.4f}, F1-Score: {class_f1:.4f}')

model_save_path = './classification_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

plt.plot(train_loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epoch')
plt.legend()
plt.savefig('./training_validation_loss_classification.png')
plt.show()
