import os
import torch
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MedicalImageDatasetWithoutLabel(Dataset):
    """
    PyTorch Dataset for MRI (3D) and X-ray (2D) images.
    This is used for segmentation tasks (no classification labels).
    """

    def __init__(self, mri_dirs, xray_dirs, target_shape_mri=(160, 384, 384), target_shape_xray=(384, 384)):
        """
        Args:
            mri_dirs (list of str): Paths to MRI folders.
            xray_dirs (list of str): Paths to X-ray folders.
            target_shape_mri (tuple): Target shape for MRI scans (D, H, W).
            target_shape_xray (tuple): Target shape for X-ray scans (H, W).
        """
        self.mri_files = []
        self.xray_files = []
        self.target_shape_mri = target_shape_mri
        self.target_shape_xray = target_shape_xray

        # Collect MRI and X-ray file paths
        for mri_dir, xray_dir in zip(mri_dirs, xray_dirs):
            if not os.path.exists(mri_dir) or not os.path.exists(xray_dir):
                print(f"Skipping missing folder: {mri_dir} or {xray_dir}")
                continue

            self.mri_files.extend([
                os.path.join(mri_dir, f) for f in os.listdir(mri_dir)
                if f.endswith((".nii", ".nii.gz", ".mhd")) and not f.startswith(".")
            ])
            self.xray_files.extend([
                os.path.join(xray_dir, f) for f in os.listdir(xray_dir)
                if f.endswith((".jpg", ".png")) and not f.startswith(".")
            ])

        # Validate MRI and X-ray files
        self.mri_files = sorted([f for f in self.mri_files if self.is_valid_mri(f)])
        self.xray_files = sorted([f for f in self.xray_files if self.is_valid_xray(f)])

        print(f"Found {len(self.mri_files)} valid MRI files")
        print(f"Found {len(self.xray_files)} valid X-ray files")
        print(f"Using {min(len(self.mri_files), len(self.xray_files))} MRI-Xray pairs.")

        if len(self.mri_files) == 0 or len(self.xray_files) == 0:
            raise RuntimeError("No valid MRI-Xray pairs found.")

    def is_valid_mri(self, filepath):
        """Check if an MRI file can be loaded properly."""
        try:
            if filepath.endswith((".nii", ".nii.gz")):
                nib.load(filepath).get_fdata()
            elif filepath.endswith(".mhd"):
                sitk.GetArrayFromImage(sitk.ReadImage(filepath))
            return True
        except Exception as e:
            print(f"Skipping {filepath} (Failed to load MRI: {e})")
            return False

    def is_valid_xray(self, filepath):
        """Check if an X-ray image file can be loaded."""
        try:
            Image.open(filepath)
            return True
        except Exception:
            return False

    def resize_mri(self, image):
        """Resize MRI to a fixed shape (zero-padding or cropping)."""
        resized = np.zeros(self.target_shape_mri, dtype=np.float32)
        d, h, w = min(image.shape[0], self.target_shape_mri[0]), \
                  min(image.shape[1], self.target_shape_mri[1]), \
                  min(image.shape[2], self.target_shape_mri[2])
        resized[:d, :h, :w] = image[:d, :h, :w]
        return resized

    def load_mri(self, filepath):
        """Load and preprocess a 3D MRI scan."""
        try:
            if filepath.endswith((".nii", ".nii.gz")):
                mri_data = nib.load(filepath).get_fdata()
            elif filepath.endswith(".mhd"):
                mri_data = sitk.GetArrayFromImage(sitk.ReadImage(filepath))

            return torch.tensor(self.resize_mri(mri_data), dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            print(f"Error loading MRI {filepath}: {e}")
            return None

    def load_xray(self, filepath):
        """Load and preprocess a 2D X-ray image."""
        try:
            img = Image.open(filepath).convert("L")
            img = img.resize(self.target_shape_xray)
            img = np.array(img, dtype=np.float32)
            return torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            print(f"Error loading X-ray {filepath}: {e}")
            return None

    def __len__(self):
        return min(len(self.mri_files), len(self.xray_files))

    def __getitem__(self, idx):
        """Return an MRI-Xray pair."""
        if idx >= len(self.mri_files):
            raise IndexError("Index out of range.")

        mri_tensor = self.load_mri(self.mri_files[idx])
        xray_tensor = self.load_xray(self.xray_files[idx])

        if mri_tensor is None or xray_tensor is None:
            return None

        return {
            "mri": mri_tensor,
            "xray": xray_tensor
        }


class MedicalImageDatasetWithLabel(MedicalImageDatasetWithoutLabel):
    """
    PyTorch Dataset for MRI (3D) and X-ray (2D) images with classification labels.
    This is used for multitask learning (MTL).
    """

    def __init__(self, mri_dirs, xray_dirs, mri_label_path, xray_label_path, target_shape_mri=(160, 384, 384), target_shape_xray=(384, 384)):
        """
        Args:
            mri_dirs (list of str): Paths to MRI folders.
            xray_dirs (list of str): Paths to X-ray folders.
            mri_label_path (str): Path to the MRI label Excel file.
            xray_label_path (str): Path to the X-ray label Excel file.
            target_shape_mri (tuple): Target shape for MRI scans.
            target_shape_xray (tuple): Target shape for X-ray scans.
        """
        super().__init__(mri_dirs, xray_dirs, target_shape_mri, target_shape_xray)

        # Load labels
        self.mri_labels = pd.read_excel(mri_label_path)
        self.xray_labels = pd.read_excel(xray_label_path)

        # Standardize ID column format
        self.mri_labels["ID"] = self.mri_labels["ID"].astype(str).str.strip()
        self.xray_labels["ID"] = self.xray_labels["ID"].astype(str).str.strip()

        # Remove duplicate IDs
        self.mri_labels = self.mri_labels.drop_duplicates(subset=["ID"], keep="first")
        self.xray_labels = self.xray_labels.drop_duplicates(subset=["ID"], keep="first")

        # Convert labels to dictionary format
        self.mri_labels_dict = self.mri_labels.set_index("ID")[["OA_label", "Risk_label"]].to_dict(orient="index")
        self.xray_labels_dict = self.xray_labels.set_index("ID")[["OA_label", "Risk_label"]].to_dict(orient="index")

    def __getitem__(self, idx):
        """Return a single MRI, X-ray sample, and classification labels."""
        sample = super().__getitem__(idx)

        if sample is None:
            return None

        mri_file = self.mri_files[idx]
        mri_id = os.path.basename(mri_file).split('.')[0]

        # Get labels (default to 0 if not found)
        labels = self.mri_labels_dict.get(mri_id, {'OA_label': 0, 'Risk_label': 0})
        sample["oa_label"] = torch.tensor(labels["OA_label"], dtype=torch.long)
        sample["risk_label"] = torch.tensor(labels["Risk_label"], dtype=torch.long)

        return sample




# if __name__ == "__main__":
#     mri_paths = [
#         "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/MOST_MRI",
#         "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/OAI_MRI",
#         "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/SkI1Data"
#     ]

#     xray_paths = [
#         "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/MOST_Xray",
#         "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/Digital_Xray",
#         "/Users/jiangwengyao/Desktop/Multi_Task/ResNet_multi/data/MRI_Xray/KOSGD"
#     ]

#     dataset_without_labels = MedicalImageDatasetWithoutLabel(mri_paths, xray_paths)
#     dataset_with_labels = MedicalImageDatasetWithLabel(mri_paths, xray_paths, "OAI_MRI.xlsx", "OAI_Xray.xlsx")

#     print(f"Dataset without labels: {len(dataset_without_labels)} samples")
#     print(f"Dataset with labels: {len(dataset_with_labels)} samples")
