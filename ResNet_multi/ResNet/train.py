import os
import torch

print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("PATH:", os.environ.get("PATH"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("PyTorch CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import OAI_Dataset  # 确保导入的自定义数据集是正确的
from multi_task_model import MultiTaskModel  # 确保导入的自定义模型是正确的
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

def multi_task_loss(seg_output, seg_target, clf_output, clf_target, dice_weight=0.5, ce_weight=0.5):
    seg_output = torch.sigmoid(seg_output)
    dice = dice_loss(seg_output, seg_target)
    cross_entropy = F.cross_entropy(clf_output, clf_target)
    return dice_weight * dice + ce_weight * cross_entropy

def calculate_metrics(outputs, targets, task_type='classification'):
    if task_type == 'classification':
        if outputs.ndim > 1:
            outputs = np.argmax(outputs, axis=1).flatten()
        targets = targets.flatten()

        pre = precision_score(targets, outputs, average='macro', zero_division=1)
        rec = recall_score(targets, outputs, average='macro', zero_division=1)
        f1 = f1_score(targets, outputs, average='macro', zero_division=1)
        return pre, rec, f1
    elif task_type == 'segmentation':
        outputs = outputs.flatten()
        targets = targets.flatten()

        if outputs.dtype in [np.float32, np.float64]:
            outputs = (outputs > 0.5).astype(int)
        targets = targets.astype(int)

        pre = precision_score(targets, outputs, average='macro', zero_division=1)
        rec = recall_score(targets, outputs, average='macro', zero_division=1)
        dice = f1_score(targets, outputs, average='macro', zero_division=1)
        jaccard = jaccard_score(targets, outputs, average='macro', zero_division=1)
        return pre, rec, dice, jaccard
    else:
        raise ValueError("Unknown task type")

# 配置
image_dir = '/gpfs/work/aac/kewu23/projects/ResNet_multi/data/segmentation_masks_A'
label_file = '/gpfs/work/aac/kewu23/projects/ResNet_multi/data/id_part_a.xlsx'

#hyperparameters
input_channels = 1
num_segmentation_classes = 1
num_classification_classes = 5
batch_size = 5
num_epochs = 10
learning_rate = 0.0001
limit =  120


# 数据加载器
train_dataset = OAI_Dataset(image_dir, label_file)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 数据加载器
train_dataset = OAI_Dataset(image_dir, label_file, limit=limit)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskModel(input_channels, num_segmentation_classes, num_classification_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 记录损失值
train_loss_values = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    
    all_seg_preds = []
    all_seg_targets = []
    all_class_preds = []
    all_class_targets = []

    for batch_idx, (images, seg_labels, class_labels) in enumerate(train_loader):
        if torch.isnan(images).any() or torch.isinf(images).any():
           print(f"Invalid values found in input images at batch {batch_idx + 1}")
           continue

        images = images.to(device)
        seg_labels = seg_labels.to(device)
        class_labels = class_labels.to(device)
        clf_target = class_labels.long()
        optimizer.zero_grad()
        
        try:
            seg_outputs, class_outputs = model(images)
        except ValueError as e:
            print(f"Error in model forward pass at batch {batch_idx + 1}: {e}")
            continue

        if seg_labels.dim() == 2 and seg_labels.shape[1] != seg_outputs.shape[2]:
            seg_labels = seg_labels.view(seg_labels.shape[0], seg_outputs.shape[2], seg_outputs.shape[3], seg_outputs.shape[4])
        
        if seg_outputs.shape[2:] != seg_labels.shape[1:]:
            seg_labels = F.interpolate(seg_labels.float().unsqueeze(1), size=seg_outputs.shape[2:], mode='nearest').squeeze(1)
        
        if seg_labels.dim() == 4:
            seg_labels = seg_labels.unsqueeze(1)
        
        loss = multi_task_loss(seg_outputs, seg_labels, class_outputs, clf_target)
        
        if torch.isnan(loss):
            print("NaN loss detected! Debug information:")
            print(f"images: {images}")
            print(f"seg_labels: {seg_labels}")
            print(f"class_labels: {class_labels}")
            print(f"seg_outputs: {seg_outputs}")
            print(f"class_outputs: {class_outputs}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_train_loss += loss.item()
        all_seg_preds.append(torch.sigmoid(seg_outputs).cpu().detach().numpy())
        all_seg_targets.append(seg_labels.cpu().detach().numpy())
        all_class_preds.append(torch.argmax(class_outputs, dim=1).cpu().detach().numpy())
        all_class_targets.append(class_labels.cpu().detach().numpy())

    average_train_loss = running_train_loss / len(train_loader)
    train_loss_values.append(average_train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}')

    # 计算评估指标
    all_seg_preds = np.concatenate(all_seg_preds)
    all_seg_targets = np.concatenate(all_seg_targets)
    all_class_preds = np.concatenate(all_class_preds)
    all_class_targets = np.concatenate(all_class_targets)

    # 调试信息：输出和目标的形状
    print(f'After epoch {epoch+1}:')
    print(f'all_seg_preds shape: {all_seg_preds.shape}')
    print(f'all_seg_targets shape: {all_seg_targets.shape}')
    print(f'all_class_preds shape: {all_class_preds.shape}')
    print(f'all_class_targets shape: {all_class_targets.shape}')

    seg_pre, seg_rec, seg_dice, seg_jaccard = calculate_metrics(all_seg_preds, all_seg_targets, task_type='segmentation')
    class_pre, class_rec, class_f1 = calculate_metrics(all_class_preds, all_class_targets, task_type='classification')

    print(f'Segmentation - Precision: {seg_pre:.4f}, Recall: {seg_rec:.4f}, Dice: {seg_dice:.4f}, Jaccard: {seg_jaccard:.4f}')
    print(f'Classification - Precision: {class_pre:.4f}, Recall: {class_rec:.4f}, F1-Score: {class_f1:.4f}')

print('Training finished.')

model_save_path = './multi_task_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

plt.plot(train_loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epoch')
plt.legend()
plt.savefig('./training_validation_loss.png')
plt.show()

