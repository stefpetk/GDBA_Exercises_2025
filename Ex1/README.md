# Exercise 1 - Large-scale land cover mapping with satellite imagery and convolutional neural networks

1) The *Instr_TechDesc* directory contains the exercise instructions as well as the technical description in Greek.
2) The *Scripts* directory containing the relevant scripts for the project, their structure is listed below:
 * [A_Read_Preprocess_S2_Ims.py](Scripts/A_Read_Preprocess_S2_Ims.py) # Read the Sentinel-2 images with the rasterio module and implement some preprocessing steps (resampling, reprojection etc.)
 * [B_Create_Dataset.py](Scripts/B_Create_Dataset.py) # Create a dataset containing patches of the augmented Sentinel-2 images and the ground truth segmentation maps
 * [C_Unet_Arch](Scripts/C_Unet_Arch.py) # Class containing the convolutional (downscaling) blocks of the encoder part of the U-net architecture which will be implemented for the task of semantic segmentation and the up-convolutional (upscaling) of the decoder part. All the "parallel" blocks of the encoder and decoder parts are "linked" via skip connections.
 * [APP1_Preprocessing_Training.py](Scripts/APP1_Preprocessing_Training.py) # Script for training the U-net model
 * [D_Predict_Merge.py](Scripts/D_Predict_Merge.py) # Class containing all the necessary methods for predicting new segmentation maps from the model trained
 * [APP2_Preprocessing_Prediction.py](Scripts/APP2_Preprocessing_Prediction.py) # Script for predicting new ground truth maps from Sentinel-2 images.
 * [EXTR_Helper_Funcs.py](Scripts/EXTR_Helper_Funcs.py) # Helper functions for the implementation

Below is an excerpt of the ground truth and predicted segmentation map.

![image](https://github.com/user-attachments/assets/1fb78a56-ab91-492f-9dff-b3c9ce00c8f9)

