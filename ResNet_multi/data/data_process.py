import os
import torch
import nibabel as nib
import SimpleITK as sitk
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MedicalImageDataset(Dataset):
    """
    Custom PyTorch Dataset for MRI (3D) and X-ray (2D) images.
    Implements intensity normalization, spatial normalization, and resolution adjustment.
    """

    def __init__(self, mri_dirs, xray_dirs, transform=None, target_shape=(128, 384, 384)):
        """
        Args:
            mri_dirs (list of str): List of paths to MRI folders.
            xray_dirs (list of str): List of paths to X-ray folders.
            transform (callable, optional): Optional transformations for X-ray images.
            target_shape (tuple, optional): Target shape for MRI scans (D, H, W).
        """
        self.mri_files = []
        self.xray_files = []
        self.transform = transform
        self.target_shape = target_shape  # Standardized shape for MRI

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

        # Validate and load MRI files
        self.mri_files = [f for f in self.mri_files if self.is_valid_mri(f)]

        # Ensure MRI and X-ray pairs exist
        min_length = min(len(self.mri_files), len(self.xray_files))
        self.mri_files = self.mri_files[:min_length]
        self.xray_files = self.xray_files[:min_length]

        print(f"Found {len(self.mri_files)} valid MRI files")
        print(f"Found {len(self.xray_files)} X-ray files")
        print(f"Using {len(self.mri_files)} MRI-Xray pairs.")

        if len(self.mri_files) == 0 or len(self.xray_files) == 0:
            raise RuntimeError("No valid MRI-Xray pairs found.")

    def is_valid_mri(self, filepath):
        """Check if an MRI file (.nii, .nii.gz, .mhd) can be loaded properly."""
        try:
            if filepath.endswith((".nii", ".nii.gz")):
                img = nib.load(filepath)
                data = img.get_fdata()
            elif filepath.endswith(".mhd"):
                img = sitk.ReadImage(filepath)
                data = sitk.GetArrayFromImage(img)
            else:
                return False

            return len(data.shape) == 3  # Ensure it's a 3D image
        except Exception as e:
            print(f"Skipping {filepath} (Failed to load MRI file: {e})")
            return False

    def normalize_intensity(self, image_array):
        """Apply intensity normalization using 0.5th to 99.5th percentile clipping."""
        lower, upper = np.percentile(image_array, [0.5, 99.5])
        image_array = np.clip(image_array, lower, upper)
        mean, std = np.mean(image_array), np.std(image_array)
        return (image_array - mean) / (std + 1e-8)  # Prevent division by zero

    def resample_image(self, image, target_size):
        """Resample a SimpleITK image to a target voxel spacing."""
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()

        new_spacing = [
            original_spacing[i] * (original_size[i] / target_size[i])
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(target_size)
        resampler.SetInterpolator(sitk.sitkLinear)

        return sitk.GetArrayFromImage(resampler.Execute(image))

    def load_mri(self, filepath):
        """Load and preprocess a 3D MRI scan."""
        try:
            if filepath.endswith((".nii", ".nii.gz")):
                nii_img = nib.load(filepath)
                mri_data = nii_img.get_fdata()
            elif filepath.endswith(".mhd"):
                mhd_img = sitk.ReadImage(filepath)
                mri_data = sitk.GetArrayFromImage(mhd_img)

                # Resample MRI to fixed shape
                mri_data = self.resample_image(mhd_img, self.target_shape)

            # Apply intensity normalization
            mri_data = self.normalize_intensity(mri_data)

            return torch.tensor(mri_data, dtype=torch.float32).unsqueeze(0)

        except Exception as e:
            print(f"Error loading MRI {filepath}: {e}")
            return None

    def load_xray(self, filepath):
        """Load and preprocess a 2D X-ray image."""
        try:
            img = Image.open(filepath).convert("L")  # Convert to grayscale
            img = img.resize((256, 256))  # Resize to match model input
            img = np.array(img, dtype=np.float32)
            
            # Apply intensity normalization
            img = self.normalize_intensity(img)

            return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        except Exception as e:
            print(f"Error loading X-ray {filepath}: {e}")
            return None

    def __len__(self):
        return len(self.mri_files)

    def __getitem__(self, idx):
        """Returns a single MRI and X-ray sample."""
        mri_file, xray_file = self.mri_files[idx], self.xray_files[idx]

        mri_tensor = self.load_mri(mri_file)
        xray_tensor = self.load_xray(xray_file)

        if mri_tensor is None or xray_tensor is None:
            print(f"Skipping index {idx} due to missing MRI or X-ray file.")
            return None

        return {
            "mri": mri_tensor, 
            "xray": xray_tensor
        }

# Example usage
if __name__ == "__main__":
    # Define MRI and X-ray paths manually
    mri_paths = [
        "/Users/jiangwengyao/Desktop/ResNet_multi/data/MRI_Xray/MOST_MRI",
        "/Users/jiangwengyao/Desktop/ResNet_multi/data/MRI_Xray/OAI_MRI",
        "/Users/jiangwengyao/Desktop/ResNet_multi/data/MRI_Xray/SkI1Data"
    ]
    
    xray_paths = [
        "/Users/jiangwengyao/Desktop/ResNet_multi/data/MRI_Xray/MOST_Xray",
        "/Users/jiangwengyao/Desktop/ResNet_multi/data/MRI_Xray/Digital_Xray",
        "/Users/jiangwengyao/Desktop/ResNet_multi/data/MRI_Xray/KOSGD"
    ]

    dataset = MedicalImageDataset(mri_paths, xray_paths)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        print(f"Batch {idx+1}")
        print("MRI Shape:", batch["mri"].shape)  # Expected: (batch_size, 1, 128, 384, 384)
        print("X-ray Shape:", batch["xray"].shape)  # Expected: (batch_size, 1, 256, 256)
        break  # Remove to process all batches
