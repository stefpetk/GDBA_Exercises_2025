import os
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd

class CreateDataset:
    """"Class definition for handling the following tasks:
    1. Create aligned patches from Sentinel-2 images and ground truth masks
    2. Create a dataset for training and validation of the Unet CNN (both S2 images and corresponding 
    ground truth maps are saved) or inference (no ground truth maps are saved).
    3. Apply geometric and radiometric augmentations to the patches as well as normalization 
    (please note that for the ground truth patches only geometric augmentations are implemented in order to avoid corruption)."""

    def __init__(self, s2_image_paths, gt_mask_paths, transform=None, normalize=True):
        """
        Args:
            s2_image_paths (list): List of paths to Sentinel-2 image bands.
            gt_mask_paths (list): List of paths to ground truth maps.
            transform (callable, optional): Optional transform (augmentations) to be applied on a sample.
            normalize (bool): Whether to normalize (divide by 10000) the Sentinel-2 images.
        """
        self.s2_image_paths = s2_image_paths
        self.gt_mask_paths = gt_mask_paths
        self.normalize = normalize
        self.patch_index = []
        
        # Define some random augmentations if not provided
        self.transform = transform or A.Compose([
            # Geometric augmentations (applied to both image and mask)
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),

            # Radiometric augmentations (applied only to image)
            A.RandomBrightnessContrast(p=0.3),
            A.Blur(blur_limit=3, p=0.3),

            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def create_patch_index(self, patch_size=256, output_dir="patches", pred=False):
        """Create patches with a specified size for the Sentinel-2 images and the corresponding ground truth maps with index metadata.
        Args:
            patch_size (int): Size of the patches to be created.
            output_dir (str): Directory to save the patches and index dataframe.
            pred (bool): If True, only S2 patches are saved for inference.
        Returns:
            self.patch_index (pd.DataFrame): DataFrame containing the paths to the patches and their indices.
        """
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        index = []

        for img_idx, (s2_bands, gt_path) in enumerate(zip(self.s2_image_paths, self.gt_mask_paths)):
            # iterate over all Sentinel-2 images and their corresponding ground truth masks and get the size of the first band
            with rasterio.open(gt_path) as gt_src, \
                 rasterio.open(s2_bands[0]) as sample_band:
                
                # Get the raster dimensions and compute the number of patches
                height = sample_band.height
                width = sample_band.width
                num_patches_x = width // patch_size
                num_patches_y = height // patch_size

                # Process patches without loading full images
                for i in range(num_patches_x):
                    for j in range(num_patches_y):
                        x_start = i * patch_size
                        y_start = j * patch_size
                        window = Window(x_start, y_start, patch_size, patch_size)

                        # Process and save S2 patch to the output directory
                        s2_patch = self._read_patch(s2_bands, window)
                        s2_path = os.path.join(output_dir, f"img_{img_idx}_patch_{i}_{j}_s2.npy")
                        np.save(s2_path, s2_patch)

                        # Process and save GT patch to the output directory. 
                        # For inference save only the Sentinel-2 patches and skip the ground truth patches creation.
                        if not pred:
                            gt_patch = gt_src.read(1, window=window)
                            gt_path = os.path.join(output_dir, f"img_{img_idx}_patch_{i}_{j}_gt.npy")
                            np.save(gt_path, gt_patch)

                            index.append({
                                "s2_path": s2_path,
                                "gt_path": gt_path,
                                "image_idx": img_idx,
                                "patch_x": i,
                                "patch_y": j
                            })
                        else:
                            index.append({
                                "s2_path": s2_path,
                                "image_idx": img_idx,
                                "patch_x": i,
                                "patch_y": j
                            })

            print(f"Processed image {img_idx+1}: {num_patches_x*num_patches_y} patches")

        # Save the dataframe with the filenames and patch indices to a CSV file
        self.patch_index = pd.DataFrame(index)
        self.patch_index.to_csv(os.path.join(output_dir, 'patch_index.csv'), index=False)
        return self.patch_index

    def _read_patch(self, band_paths, window):
        """Read and normalize a single patch from the Sentinel-2 images directory.
        Args:
            band_paths (list): List of paths to Sentinel-2 image bands.
            window (rasterio.windows.Window): Window object defining the patch location.
        Returns:
            np.ndarray: Stacked patch data from all bands."""
        patch = []
        for band_path in band_paths:
            with rasterio.open(band_path) as src:
                band_data = src.read(1, window=window)
                if self.normalize:
                    band_data = band_data.astype(np.float32) / 10000.0
                patch.append(band_data)
        return np.stack(patch, axis=0)

    def create_splits(self, test_size=0.2, patch_df=None):
        """Create trainining/validation set splits using the index of each patch.
        Args:
            test_size (float): Proportion of the dataset to include in the validation split.
            patch_df (pd.DataFrame, optional): DataFrame containing the paths to the patches and their indices.
        Returns:
            tuple: DataFrames with the indices of the patches that are assigned to the training and validation sets."""
        
        # If no dataframe with paths and indices to the patches is provided, use the one returned by the create_patch_index method.
        if patch_df is None:
            train_idx, val_idx = train_test_split(
            self.patch_index.index,
            test_size=test_size,
            shuffle=False,
            random_state=42
            )
            return self.patch_index.iloc[train_idx], self.patch_index.iloc[val_idx]

        # If a dataframe is provided, use it to create the splits.
        else:
            train_idx, val_idx = train_test_split(
                patch_df.index,
                test_size=test_size,
                shuffle=False,
                random_state=42
            )
            return patch_df.iloc[train_idx], patch_df.iloc[val_idx]

class PatchDataset(Dataset):
    """Dataset loader for the patches created by the CreateDataset class for training/validation or inference.
    Also implements the augmentations and normalization defined in the CreateDataset class"""

    def __init__(self, index_df, transform=None, pred=False):
        """
        Args:
            index_df (pd.DataFrame): DataFrame containing the paths to the patches and their indices.
            transform (callable, optional): Optional transform (augmentations) to be applied on a sample.
            pred (bool): If True, only S2 patches are loaded for inference.
        """
        self.index_df = index_df
        self.transform = transform
        self.pred = pred

    def __len__(self):
        """Return the number of patches in the dataset."""
        return len(self.index_df)

    def __getitem__(self, idx):
        """Get a patch and its corresponding ground truth mask (if available) by index."""
        row = self.index_df.iloc[idx]
        s2_patch = np.load(row['s2_path'])
        if not self.pred:
            gt_patch = np.load(row['gt_path']) #
        else:
            pass

        if self.transform:
            transformed = self.transform(
                image=s2_patch.transpose(1, 2, 0), # HWC format for augmentations with the albumentations module
                mask=gt_patch
            )
            s2_patch = transformed['image']
            gt_patch = transformed['mask']
        elif self.pred:
            s2_patch = torch.from_numpy(s2_patch).float()
        else:
            s2_patch = torch.from_numpy(s2_patch).float()
            gt_patch = torch.from_numpy(gt_patch).long()

        if self.transform or not self.pred:
            return s2_patch, gt_patch
        else:
            return s2_patch