import os
import torch
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MedicalImageDataset(Dataset):
    """
    Custom PyTorch Dataset for MRI (3D) and X-ray (2D) images with classification labels.
    """

    def __init__(self, mri_dirs, xray_dirs, mri_label_path, xray_label_path, target_shape=(128, 384, 384)):
        """
        Args:
            mri_dirs (list of str): Paths to MRI folders.
            xray_dirs (list of str): Paths to X-ray folders.
            mri_label_path (str): Path to the MRI label Excel file.
            xray_label_path (str): Path to the X-ray label Excel file.
            target_shape (tuple, optional): Target shape for MRI scans (D, H, W).
        """
        self.mri_files = []
        self.xray_files = []
        self.target_shape = target_shape  # Standardized shape for MRI

        # Load label data
        self.mri_labels = pd.read_excel(mri_label_path)
        self.xray_labels = pd.read_excel(xray_label_path)

        # Standardize ID column format (remove trailing spaces if any)
        self.mri_labels["ID"] = self.mri_labels["ID"].astype(str).str.strip()
        self.xray_labels["ID"] = self.xray_labels["ID"].astype(str).str.strip()

        # Remove duplicates (keep first occurrence)
        self.mri_labels = self.mri_labels.drop_duplicates(subset=["ID"], keep="first")
        self.xray_labels = self.xray_labels.drop_duplicates(subset=["ID"], keep="first")

        # Convert labels to dictionaries
        self.mri_labels_dict = self.mri_labels.set_index("ID")[["OA_label", "Risk_label"]].to_dict(orient="index")
        self.xray_labels_dict = self.xray_labels.set_index("ID")[["OA_label", "Risk_label"]].to_dict(orient="index")

        # Collect MRI and X-ray file paths
        for mri_dir, xray_dir in zip(mri_dirs, xray_dirs):
            if not os.path.exists(mri_dir) or not os.path.exists(xray_dir):
                print(f"Skipping missing folder: {mri_dir} or {xray_dir}")
                continue

            mri_files = [os.path.join(mri_dir, f) for f in os.listdir(mri_dir) 
                         if f.endswith((".nii", ".nii.gz", ".mhd")) and not f.startswith(".")]
            xray_files = [os.path.join(xray_dir, f) for f in os.listdir(xray_dir) 
                          if f.endswith((".jpg", ".png")) and not f.startswith(".")]

            self.mri_files.extend(mri_files)
            self.xray_files.extend(xray_files)

        # Validate MRI files
        self.mri_files = [f for f in self.mri_files if self.is_valid_mri(f)]
        self.xray_files = [f for f in self.xray_files if self.is_valid_xray(f)]

        # Ensure MRI and X-ray files are properly paired (matching order)
        self.mri_files.sort()
        self.xray_files.sort()

        print(f"Found {len(self.mri_files)} valid MRI files")
        print(f"Found {len(self.xray_files)} valid X-ray files")
        print(f"Using {len(self.mri_files)} MRI-Xray pairs.")

        if len(self.mri_files) == 0 or len(self.xray_files) == 0:
            raise RuntimeError("No valid MRI-Xray pairs found.")

    def is_valid_mri(self, filepath):
        """Check if an MRI file can be loaded properly."""
        try:
            if filepath.endswith((".nii", ".nii.gz")):
                img = nib.load(filepath)
                data = img.get_fdata()
            elif filepath.endswith(".mhd"):
                img = sitk.ReadImage(filepath)
                data = sitk.GetArrayFromImage(img)
            else:
                return False
            return len(data.shape) == 3
        except Exception as e:
            print(f"Skipping {filepath} (Failed to load MRI file: {e})")
            return False

    def is_valid_xray(self, filepath):
        """Check if an X-ray image file can be loaded."""
        try:
            Image.open(filepath)
            return True
        except Exception:
            return False

    def normalize_intensity(self, image_array):
        """Normalize image intensity using percentile clipping."""
        lower, upper = np.percentile(image_array, [0.5, 99.5])
        image_array = np.clip(image_array, lower, upper)
        mean, std = np.mean(image_array), np.std(image_array)
        return (image_array - mean) / (std + 1e-8)

    def load_mri(self, filepath):
        """Load and preprocess a 3D MRI scan."""
        try:
            if filepath.endswith((".nii", ".nii.gz")):
                nii_img = nib.load(filepath)
                mri_data = nii_img.get_fdata()
            elif filepath.endswith(".mhd"):
                mhd_img = sitk.ReadImage(filepath)
                mri_data = sitk.GetArrayFromImage(mhd_img)

            mri_data = self.normalize_intensity(mri_data)
            return torch.tensor(mri_data, dtype=torch.float32).unsqueeze(0)

        except Exception as e:
            print(f"Error loading MRI {filepath}: {e}")
            return None

    def load_xray(self, filepath):
        """Load and preprocess a 2D X-ray image."""
        try:
            img = Image.open(filepath).convert("L")
            img = img.resize((256, 256))
            img = np.array(img, dtype=np.float32)
            img = self.normalize_intensity(img)
            return torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            print(f"Error loading X-ray {filepath}: {e}")
            return None

    def __len__(self):
        return len(self.mri_files)

    def __getitem__(self, idx):
        """Return a single MRI, X-ray sample, and corresponding labels."""
        if idx >= len(self.mri_files):
            raise IndexError("Index out of range.")

        mri_file = self.mri_files[idx]
        xray_file = self.xray_files[idx]

        mri_tensor = self.load_mri(mri_file)
        xray_tensor = self.load_xray(xray_file)

        if mri_tensor is None or xray_tensor is None:
            print(f"Skipping index {idx} due to missing MRI or X-ray file.")
            return None

        # Extract ID from filename
        mri_id = os.path.basename(mri_file).split('.')[0]

        # Get labels (default to 0 if not found)
        labels = self.mri_labels_dict.get(mri_id, {'OA_label': 0, 'Risk_label': 0})
        oa_label = torch.tensor(labels['OA_label'], dtype=torch.long)
        risk_label = torch.tensor(labels['Risk_label'], dtype=torch.long)

        return {
            "mri": mri_tensor, 
            "xray": xray_tensor,
            "oa_label": oa_label,
            "risk_label": risk_label
        }
