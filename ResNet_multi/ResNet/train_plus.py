import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import OAI_Dataset  # 确保导入更新后的OAI_Dataset类
from multi_task_model import MultiTaskModel
from maml import MAML
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

def compute_loss(outputs, targets):
    # Ensure targets['segmentation'] is the same size as outputs[0]
    if targets['segmentation'].size() != outputs[0].size():
        targets['segmentation'] = nn.functional.interpolate(targets['segmentation'].unsqueeze(1).float(), size=outputs[0].shape[2:], mode='trilinear', align_corners=False)
        targets['segmentation'] = targets['segmentation'].squeeze(1)  # Remove the channel dimension after interpolation

    # Ensure targets['segmentation'] has the same shape as outputs[0]
    targets['segmentation'] = targets['segmentation'].unsqueeze(1)  # Add the channel dimension back

    print(f"seg_outputs shape: {outputs[0].shape}")
    print(f"seg_labels shape after resize: {targets['segmentation'].shape}")

    classification_loss = nn.CrossEntropyLoss()(outputs[1], targets['classification'].long())
    segmentation_loss = dice_loss(torch.sigmoid(outputs[0]), targets['segmentation'])
    return classification_loss + segmentation_loss

def calculate_metrics(outputs, targets, task_type='classification'):
    if task_type == 'classification':
        outputs = outputs.numpy().flatten()
        targets = targets.numpy().flatten()

        # # 转换为 NumPy 数组并拼接
        # outputs = np.concatenate([output.numpy() for output in valid_outputs]).flatten()
        # targets = np.concatenate([target.numpy() for target in valid_targets]).flatten()

        pre = precision_score(targets, outputs, average='macro', zero_division=1)
        rec = recall_score(targets, outputs, average='macro', zero_division=1)
        f1 = f1_score(targets, outputs, average='macro', zero_division=1)
        return pre, rec, f1
    elif task_type == 'segmentation':
        outputs = outputs.flatten()
        targets = targets.flatten()

        # Ensure that outputs and targets have the same number of elements
        if outputs.numel() != targets.numel():
            min_size = min(outputs.numel(), targets.numel())
            outputs = outputs[:min_size]
            targets = targets[:min_size]

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

# 配置
image_dir = '/gpfs/work/aac/kewu23/projects/ResNet_multi/data/segmentation_masks_A'
label_file = '/gpfs/work/aac/kewu23/projects/ResNet_multi/data/id_part_a.xlsx'
input_channels = 1
num_segmentation_classes = 1
num_classification_classes = 5
batch_size = 15 # 减小批量大小以减少内存占用
num_epochs = 10
learning_rate = 1e-6  # 减少学习率
inner_lr = 0.01
outer_lr = 0.001
accumulation_steps = 8  # 梯度累积步数
limit =  120 # 限制数据集的大小

# 数据加载器
train_dataset = OAI_Dataset(image_dir, label_file, limit=limit)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 检查是否有 GPU 可用，并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskModel(input_channels, num_segmentation_classes, num_classification_classes).to(device)
maml = MAML(model, lr_inner=inner_lr, lr_outer=outer_lr)

# 设置混合精度训练，仅当有 GPU 可用时启用
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# 记录损失值
train_loss_values = []
all_seg_preds = []
all_seg_targets = []
all_class_preds = []
all_class_targets = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    
    for batch_idx, (images, seg_labels, class_labels) in enumerate(train_loader):
        if torch.isnan(images).any() or torch.isinf(images).any():
           print(f"Invalid values found in input images at batch {batch_idx + 1}")
           continue

        images = images.to(device)
        seg_labels = seg_labels.to(device)
        class_labels = class_labels.to(device)

        # 打印输入数据的形状和检查NaN/Inf
        print(f"Batch {batch_idx + 1} - images shape: {images.shape}, seg_labels shape: {seg_labels.shape}, class_labels shape: {class_labels.shape}")
        if torch.isnan(images).any() or torch.isinf(images).any():
            print(f"NaN/Inf values found in images at batch {batch_idx + 1}")
            continue
        if torch.isnan(seg_labels).any() or torch.isinf(seg_labels).any():
            print(f"NaN/Inf values found in seg_labels at batch {batch_idx + 1}")
            continue
        if torch.isnan(class_labels).any() or torch.isinf(class_labels).any():
            print(f"NaN/Inf values found in class_labels at batch {batch_idx + 1}")
            continue

        support_set = (images, {'segmentation': seg_labels, 'classification': class_labels})
        query_set = (images, {'segmentation': seg_labels, 'classification': class_labels})

        # 进行meta-training步骤
        maml.train_step([(support_set, query_set)])

        if use_amp:
            with torch.cuda.amp.autocast():
                seg_outputs, class_outputs = model(images)
                loss = compute_loss((seg_outputs, class_outputs), {'segmentation': seg_labels, 'classification': class_labels})
        else:
            seg_outputs, class_outputs = model(images)
            loss = compute_loss((seg_outputs, class_outputs), {'segmentation': seg_labels, 'classification': class_labels})

        running_train_loss += loss.item()

        # 打印输出数据的形状
        print(f"Batch {batch_idx + 1} - seg_outputs shape: {seg_outputs.shape}, class_outputs shape: {class_outputs.shape}")

        all_seg_preds.append(torch.sigmoid(seg_outputs).cpu().detach())
        all_seg_targets.append(seg_labels.cpu().detach())
        all_class_preds.append(torch.argmax(class_outputs, dim=1).cpu().detach())
        all_class_targets.append(class_labels.cpu().detach())

        # 梯度累积
        loss = loss / accumulation_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        # 清理内存
        del images, seg_labels, class_labels, seg_outputs, class_outputs, loss
        torch.cuda.empty_cache()

    average_train_loss = running_train_loss / len(train_loader)
    train_loss_values.append(average_train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}')

# 计算评估指标前打印形状
all_seg_preds = torch.cat(all_seg_preds, dim=0)
all_seg_targets = torch.cat(all_seg_targets, dim=0)
all_class_preds = torch.cat(all_class_preds, dim=0)
all_class_targets = torch.cat(all_class_targets, dim=0)

print(f"all_seg_preds shape: {all_seg_preds.shape}")
print(f"all_seg_targets shape: {all_seg_targets.shape}")
print(f"all_class_preds shape: {all_class_preds.shape}")
print(f"all_class_targets shape: {all_class_targets.shape}")



seg_pre, seg_rec, seg_dice, seg_jaccard = calculate_metrics(all_seg_preds, all_seg_targets, task_type='segmentation')
print(f'Segmentation - Precision: {seg_pre:.4f}, Recall: {seg_rec:.4f}, Dice: {seg_dice:.4f}, Jaccard: {seg_jaccard:.4f}')


# print(f'Classification - Precision: {class_pre:.4f}, Recall: {class_rec:.4f}, F1-Score: {class_f1:.4f}')
# class_pre, class_rec, class_f1 = calculate_metrics(all_class_preds, all_class_targets, task_type='classification')


if len(all_class_preds) > 0 and len(all_class_targets) > 0:
    all_class_preds = torch.cat([all_class_preds], dim=0).long()
    all_class_targets = torch.cat([all_class_targets], dim=0).long()

    if torch.is_tensor(all_class_preds) and torch.is_tensor(all_class_targets):
        print(f"all_class_preds shape: {all_class_preds.shape}")
        print(f"all_class_targets shape: {all_class_targets.shape}")
        print(all_class_preds)
        print(all_class_targets)

        class_pre, class_rec, class_f1 = calculate_metrics(all_class_preds, all_class_targets, task_type='classification')
        print(f'Classification - Precision: {class_pre:.4f}, Recall: {class_rec:.4f}, F1: {class_f1:.4f}')
    else:
        print("all_class_preds or all_class_targets is not a tensor.")
else:
    print("There is not enough classified forecast or target data to calculate indicators.")



print('Training finished.')

model_save_path = './multi_task_model_maml.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

plt.plot(train_loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epoch')
plt.legend()
plt.savefig('./training_validation_loss_maml.png')
plt.show()
