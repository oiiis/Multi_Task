import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import OAI_Dataset
from seg_for_single_task import SegmentationModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def dice_loss(pred, target, smooth=1e-5):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=(2, 3, 4))
    loss = (1 - ((2. * intersection + smooth) / 
                 (pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + smooth)))

    return loss.mean()

def calculate_metrics(outputs, targets, task_type='segmentation'):
    if task_type == 'segmentation':
        outputs = outputs.flatten()
        targets = targets.flatten()

        if torch.is_floating_point(outputs):
            outputs = (outputs > 0.5).int()
        
        targets = targets.int()

        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()

        pre = precision_score(targets, outputs, average='macro', zero_division=1)
        rec = recall_score(targets, outputs, average='macro', zero_division=1)
        dice = f1_score(targets, outputs, average='macro', zero_division=1)
        jaccard = jaccard_score(targets, outputs, average='macro', zero_division=1)
        return pre, rec, dice, jaccard
    else:
        raise ValueError("Unknown task type")

image_dir = '/gpfs/work/aac/kewu23/projects/ResNet_multi/data/segmentation_masks_A'
label_file = '/gpfs/work/aac/kewu23/projects/ResNet_multi/data/id_part_a.xlsx'
input_channels = 1
num_segmentation_classes = 1
batch_size = 12
num_epochs = 10
learning_rate = 0.0001
limit =  120


# 数据加载器
train_dataset = OAI_Dataset(image_dir, label_file)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegmentationModel(input_channels, num_segmentation_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss_values = []
all_seg_preds = []
all_seg_targets = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for batch_idx, (images, seg_labels, _) in enumerate(train_loader):
        if torch.isnan(images).any() or torch.isinf(images).any():
           print(f"Invalid values found in input images at batch {batch_idx + 1}")
           continue

        images = images.to(device)
        seg_labels = seg_labels.to(device)
        optimizer.zero_grad()

        try:
            seg_outputs = model(images)
        except ValueError as e:
            print(f"Error in model forward pass at batch {batch_idx + 1}: {e}")
            continue

        if seg_labels.dim() == 2 and seg_labels.shape[1] != seg_outputs.shape[2]:
            seg_labels = seg_labels.view(seg_labels.shape[0], seg_outputs.shape[2], seg_outputs.shape[3], seg_outputs.shape[4])
        
        if seg_outputs.shape[2:] != seg_labels.shape[1:]:
            seg_labels = F.interpolate(seg_labels.float().unsqueeze(1), size=seg_outputs.shape[2:], mode='nearest').squeeze(1)
        
        if seg_labels.dim() == 4:
            seg_labels = seg_labels.unsqueeze(1)

        loss = dice_loss(torch.sigmoid(seg_outputs), seg_labels)
        
        if torch.isnan(loss):
            print("NaN loss detected! Debug information:")
            print(f"images: {images}")
            print(f"seg_labels: {seg_labels}")
            print(f"seg_outputs: {seg_outputs}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_train_loss += loss.item()
        all_seg_preds.append(torch.sigmoid(seg_outputs).cpu().detach())
        all_seg_targets.append(seg_labels.cpu().detach())

    average_train_loss = running_train_loss / len(train_loader)
    train_loss_values.append(average_train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}')

all_seg_preds = torch.cat(all_seg_preds, dim=0)
all_seg_targets = torch.cat(all_seg_targets, dim=0)

seg_pre, seg_rec, seg_dice, seg_jaccard = calculate_metrics(all_seg_preds, all_seg_targets, task_type='segmentation')

print(f'Segmentation - Precision: {seg_pre:.4f}, Recall: {seg_rec:.4f}, Dice: {seg_dice:.4f}, Jaccard: {seg_jaccard:.4f}')

model_save_path = './segmentation_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

plt.plot(train_loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epoch')
plt.legend()
plt.savefig('./training_validation_loss_segmentation.png')
plt.show()
