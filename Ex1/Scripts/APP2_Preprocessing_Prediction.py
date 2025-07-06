from A_Read_Preprocess_S2_Ims import PreprocessRasters
from B_Create_Dataset import CreateDataset, PatchDataset
from C_Unet_Arch import UNet
from D_Predict_Merge import PredTest
from scipy import ndimage
from torch.utils.data import DataLoader
import glob
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.transform import Affine

# define the preprocessor class instance for prediction/inference
preprocessor_pred = PreprocessRasters()

# Unzip, read and preprocess (apply pansharpening/resampling and necessary reprojections) to the Sentinel-2 image and the ground truth map
preprocessor_pred.read_and_preprocess_S2_images('/mnt/c/path/to/S2_Ims_Unzipped')

# Crop the Sentinel-2 images to match the ground truth maps extents
fgt_paths = glob.glob('/mnt/c/path/to/*ref_data.tif') # final ground truth maps
s2_ims_path = glob.glob('/mnt/c/path/to/S2_Ims_Unzipped/*_T34SEH_*') # path to the S2 imagery folders
s2_bands_paths = [glob.glob(os.path.join(s2_im_path, 'GRANULE', '*', 'IMG_DATA', '*.jp2')) for s2_im_path in s2_ims_path]

for s2_bands_path in s2_bands_paths:
    for band_path in s2_bands_path:
        if 'TCI' in band_path:
            s2_bands_path.remove(band_path)

# Create the patches which will be fed to the U-net, split the dataset to training and validation sets and  finally create the respective data loader
if __name__ == "__main__":
    # Initialize and create patch index
    dataset = CreateDataset(s2_bands_paths, fgt_paths)
    patch_index = dataset.create_patch_index(output_dir=os.path.join('/mnt/c/path/to', 'patches_pred'), pred=True, patch_size=2196) # split the dataset to patches of 2196x2196 pixels
    patch_df = pd.read_csv('/mnt/c/path/to/patches_pred/patch_index.csv')

    batch_size = 1 # no need to set a large batch size for inference
    test_ds = PatchDataset(patch_df, dataset.transform, pred=True)

    # determine device for model training and memory pinning
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = True if device == "cuda" else False

    # Initialize model architecture and load trained weights for the patches
    unet = UNet(encChannels=(13, 32, 64, 128), decChannels=(128, 64, 32), nbClasses=8, retainDim=True, outSize=(2196, 2196)).to(device)
    unet.load_state_dict(torch.load('/mnt/c/path/to/unet_model_bs8.pth', map_location=device))
    unet.eval()

    # Get the images positions from the patch index dataframe
    patch_coords = [(x, y) for x, y in zip(patch_df['patch_y'].values, patch_df['patch_x'].values)]

    # Create an instance of the class for prediction and image merging
    patches_paths = glob.glob('mnt/c/path/to/patches_pred/*.npy')

    patches_pred = [torch.from_numpy(np.load(patch_path)) for patch_path in patches_paths]
    patches_coords = [(x, y, patch) for (x, y), patch in zip(patch_coords, patches_pred)] # add image coordinates to the patches

    # Define the instance of the class for prediction and image merging
    pred_merge = PredTest(patches_coords, 16, device, unet, (10980, 10980), {0:10, 1:20, 2:30, 3:40, 4:50, 5:60, 6:80, 7:90}, 2196)

    # Make predictions, stitch patches and calculate the receptive field
    pred_arr = pred_merge.prediction_inference()
    rec_fd = pred_merge.calculate_receptive_field()
    print(f"Receptive field: {rec_fd}")

    # save the predictions as a georeferenced tif
    left_x = 499975
    top_y = 4300026
    transform = Affine(10.0, 0.0, left_x, 0.0, -10.0, top_y)  # From original image metadata
    pred_merge.save_georeferenced_tif(pred_arr,  crs='EPSG:32634', transform=transform, 
    output_path='/mnt/c/path/to/prediction/GDBA25_ex1_34SEH_prediction_test.tif',
    compression='jpeg')

    # test the ground truth input
    test_gt = pred_merge.eval_prediction('/mnt/c/path/to/GBDA24_ex2_34SEH_ref_data.tif')
    print(f"Dice score: {test_gt[0]}")
    print(f"IoU: {test_gt[1]}")