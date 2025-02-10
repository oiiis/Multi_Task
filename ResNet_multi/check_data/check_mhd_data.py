import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import time

def check_mhd_files(image_dir):
    """检查目录中的所有 .mhd 文件，并打印它们的基本信息和元数据。"""
    mhd_files = [f for f in os.listdir(image_dir) if f.endswith('.mhd')]
    
    for mhd_file in mhd_files:
        file_path = os.path.join(image_dir, mhd_file)
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)
        
        print(f'Checking file: {mhd_file}')
        print(f'Image shape: {image_array.shape}')
        
        # 假设图像是三维的单通道图像
        if len(image_array.shape) == 3:
            print('Segmentation label exists.')
        else:
            print('Segmentation label does NOT exist.')
        
        # 提取并打印元数据
        print('Extracting metadata...')
        meta_keys = image.GetMetaDataKeys()  # 获取元数据键
        for key in meta_keys:
            meta_value = image.GetMetaData(key)  # 获取每个键对应的值
            print(f'{key}: {meta_value}')
            
            # 检查是否有可能的分类标记
            if 'class' in key.lower() or 'label' in key.lower() or 'diagnosis' in key.lower():
                print(f'Possible classification label found: {meta_value}')
        
        print('-' * 50)


def get_unique_labels(image_dir, num_files=3):
    """获取目录中每个 .mhd 文件的唯一标签，并打印它们。"""
    mhd_files = [f for f in os.listdir(image_dir) if f.endswith('.mhd')]
    all_unique_labels = []

    for i, mhd_file in enumerate(mhd_files[:num_files]):
        try:
            start_time = time.time()
            file_path = os.path.join(image_dir, mhd_file)
            print(f"Processing file: {file_path}")

            # 读取图像
            read_start = time.time()
            image = sitk.ReadImage(file_path)
            image_array = sitk.GetArrayFromImage(image)
            read_end = time.time()
            print(f"Time to read image: {read_end - read_start:.2f} seconds")

            # 假设图像是三维的单通道图像，标签也是三维的单通道图像
            label = image_array  # 在这种情况下，图像本身就是标签
            image_data = image_array  # 如果你有单独的图像数据，可以在这里读取

            # 获取唯一标签值
            unique_labels = np.unique(label)
            all_unique_labels.append(unique_labels)

            print(f'File: {mhd_file}')
            print(f'Unique labels in segmentation: {unique_labels}')
            print('-' * 50)

            # 可视化
            slice_index = image_array.shape[0] // 2  # 中间切片
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image_data[slice_index, :, :], cmap='gray')
            plt.title('Image')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(image_data[slice_index, :, :], cmap='gray')
            plt.imshow(label[slice_index, :, :], cmap='jet', alpha=0.5)  # 使用透明度以重叠显示标签
            plt.title('Segmentation Label')
            plt.axis('off')
            plt.suptitle(f'File: {mhd_file}')
            plt.show()

        except Exception as e:
            print(f"Error processing file {mhd_file}: {e}")

    return all_unique_labels

if __name__ == "__main__":
    image_dir = './data/segmentation_masks_A'
    # 检查目录中的所有 .mhd 文件
    check_mhd_files(image_dir)
    # 获取所有 .mhd 文件的唯一标签
    unique_labels_list = get_unique_labels(image_dir)

    # 打印所有文件的唯一标签值
    for i, unique_labels in enumerate(unique_labels_list):
        print(f'File {i+1} unique labels: {unique_labels}')


