import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
import torch

class OAI_Dataset(Dataset):
    def __init__(self, image_dir, label_file):
        self.image_dir = image_dir
        
        # 读取分类标签
        self.classification_labels = pd.read_excel(label_file, header=None, index_col=0, engine='openpyxl')
        self.classification_labels.index.name = 'id'
        self.classification_labels.columns = ['kl_grade']

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.mhd') and 'segmentation_masks' in f]
        self.image_files.sort()  # 确保文件按顺序排列

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        
        # 读取图像
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)

        # 数据检查
        if np.isnan(image_array).any() or np.isinf(image_array).any():
            raise ValueError(f"Invalid values found in image: {image_file}")

        # 调整形状，确保通道维度正确
        if len(image_array.shape) == 3:
            # 添加通道维度（假设输入图像是单通道的）
            image_array = np.expand_dims(image_array, axis=0)
        elif len(image_array.shape) == 4 and image_array.shape[0] == 1:
            # 输入已经有通道维度
            pass
        else:
            raise ValueError(f"Unexpected image shape: {image_array.shape}")

        # 获取对应的标签ID
        file_id = int(image_file.split('.')[0].split('_')[0])
        
        # 读取分割标签
        seg_label_path = image_path  # 分割标签路径和图像路径相同
        seg_label = sitk.ReadImage(seg_label_path)
        seg_label_array = sitk.GetArrayFromImage(seg_label)

        # 检查分割标签的维度是否为3
        if len(seg_label_array.shape) != 3:
            raise ValueError(f"Unexpected label array shape: {seg_label_array.shape}. Expected 3 dimensions.")

        # 调整分割标签的形状和空间分辨率以匹配 seg_outputs
        desired_shape = (image_array.shape[-3], image_array.shape[-2], image_array.shape[-1])  # 根据输入图像的形状确定目标形状
        seg_label_array_resampled = []
        
        for label_slice in seg_label_array:
            if label_slice.shape != desired_shape[-2:]:
                zoom_factors = (desired_shape[-2] / label_slice.shape[0],
                                desired_shape[-1] / label_slice.shape[1])
                try:
                    resampled_slice = ndimage.zoom(label_slice, zoom=zoom_factors, order=0)
                except Exception as e:
                    print(f"Error while resampling label slice: {e}")
                    resampled_slice = np.zeros(desired_shape[-2:], dtype=label_slice.dtype)
            else:
                resampled_slice = label_slice
            
            seg_label_array_resampled.append(resampled_slice)

        seg_label_array_resampled = np.array(seg_label_array_resampled)

        if np.isnan(seg_label_array_resampled).any() or np.isinf(seg_label_array_resampled).any():
            raise ValueError(f"Invalid values found in segmentation label: {image_file}")

        # 将分割标签转换为 PyTorch 的 LongTensor 类型
        seg_label_tensor = torch.LongTensor(seg_label_array_resampled)

        # 获取分类标签
        class_label = self.classification_labels.loc[file_id, 'kl_grade']

        return image_array, seg_label_tensor, class_label

# 测试数据集
image_dir = 'ResNet_multi/data/segmentation_masks_A'
label_file = 'ResNet_multi/id_part_a.xlsx'

dataset = OAI_Dataset(image_dir, label_file)
loader = DataLoader(dataset, batch_size=4, shuffle=True)



# 检查每个批次的数据
# for batch_idx, (images, seg_labels, class_labels) in enumerate(loader):
#     print(f"Batch {batch_idx + 1}:")
#     print(f"Images shape: {images.shape}")
#     print(f"Segmentation labels shape: {seg_labels.shape}")
#     print(f"Classification labels: {class_labels}")
#     if torch.isnan(images).any() or torch.isinf(images).any():
#         print(f"Invalid values found in images at batch {batch_idx + 1}")
#     if torch.isnan(seg_labels).any() or torch.isinf(seg_labels).any():
#         print(f"Invalid values found in segmentation labels at batch {batch_idx + 1}")
#     if torch.isnan(class_labels).any() or torch.isinf(class_labels).any():
#         print(f"Invalid values found in classification labels at batch {batch_idx + 1}")
