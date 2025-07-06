from A_Read_Preprocess_S2_Ims import PreprocessRasters
import torch
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import MulticlassJaccardIndex
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
import glob
from B_Create_Dataset import CreateDataset, PatchDataset
from C_Unet_Arch import UNet
from EXTR_Helper_Funcs import HelperFunctions
import pandas as pd
from torchgeo.models import ResNet50_Weights

# define the preprocessor class instance
preprocessor = PreprocessRasters()

# Unzip, read and preprocess (apply pansharpening/resampling and necessary reprojections) to the Sentinel-2 images and the ground truth maps
preprocessor.read_and_preprocess_S2_images('/mnt/c/path/to/S2_Ims')

# List of Sentinel-2 image paths (only one band is needed) to crop the ground truth maps to the Sentinel-2 images extents
sentinel2_paths = [
    '/mnt/c/path/to/S2_Ims_Unzipped/S2A_MSIL1C_20210925T092031_N0500_R093_T34SEJ_20230118T233535.SAFE/GRANULE/L1C_T34SEJ_A032694_20210925T092343/IMG_DATA/T34SEJ_20210925T092031_B02.jp2',
    '/mnt/c/path/to/S2_Ims_Unzipped/S2A_MSIL1C_20210925T092031_N0500_R093_T34SFJ_20230118T233535.SAFE/GRANULE/L1C_T34SFJ_A032694_20210925T092343/IMG_DATA/T34SFJ_20210925T092031_B02.jp2',
    '/mnt/c/path/to/S2_Ims_Unzipped/S2A_MSIL1C_20210925T092031_N0500_R093_T34TEK_20230118T233535.SAFE/GRANULE/L1C_T34TEK_A032694_20210925T092343/IMG_DATA/T34TEK_20210925T092031_B02.jp2',
    '/mnt/c/path/to/S2_Ims_Unzipped/S2A_MSIL1C_20210925T092031_N0500_R093_T34TFK_20230118T233535.SAFE/GRANULE/L1C_T34TFK_A032694_20210925T092343/IMG_DATA/T34TFK_20210925T092031_B02.jp2']

gtruth_path = '/mnt/c/GBDA25_ex1_ref_data.tif' # path to the ground truth map

preprocessor.read_preprocess_gtruth(
    gt_path=gtruth_path,
    sentinel2_paths=sentinel2_paths,
    output_dir=os.path.dirname(gtruth_path)) # reproject the ground truth map and split it to match the S2 images

# Crop the Sentinel-2 images to match the ground truth maps extents
fgt_paths = glob.glob('/mnt/c/path/to/Ground_Truth/*B02_final.tif') # final ground truth maps
s2_ims_path = glob.glob('/mnt/c/path/to/S2_Ims_Unzipped/*') # path to the S2 imagery folders
s2_bands_paths = [glob.glob(os.path.join(s2_im_path, 'GRANULE', '*', 'IMG_DATA', '*.jp2')) for s2_im_path in s2_ims_path]

for s2_bands_path in s2_bands_paths:
    for band_path in s2_bands_path:
        if 'TCI' in band_path:
            s2_bands_path.remove(band_path)

preprocessor.clip_preprocessed_s2(s2_bands_paths, fgt_paths) # align pixelwise the ground truth maps with the Sentinel-2 images

# Create the patches which will be fed to the U-net, split the dataset to training and validation sets and  finally create the respective data loader
if __name__ == "__main__":
    # Initialize and create patch index
    dataset = CreateDataset(s2_bands_paths, fgt_paths)
    patch_index = dataset.create_patch_index(output_dir=os.path.join('/mnt/c/path/to', 'patches'))

    # Create splits
    patch_df = pd.read_csv('/mnt/c/path/to/patches/patch_index.csv') # Load the saved patch path and index from the CSV file
    train_index, val_index = dataset.create_splits(test_size=0.3, patch_df=patch_df)

    # Create datasets with lazy loading
    train_ds = PatchDataset(train_index, transform=dataset.transform)
    val_ds = PatchDataset(val_index)

    # Create DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # determine device for model training and memory pinning
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = True if device == "cuda" else False

    # initialize learning rate, batch size and number of training epochs (in this script the parameters of the best model are shown)
    init_lr = 0.005
    epochs = 50

    # list the original labels of the ground truth map
    original_labels = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
    label_mapping = {label.item(): idx for idx, label in enumerate(original_labels)}

    # Initialize model with MoCo weights
    nbClasses = len(label_mapping)  # Number of classes in the dataset
    unet = UNet(nbClasses=nbClasses).to(device)

    # import the helper functions
    help_funcs = HelperFunctions(model=unet, checkpoint_path='/mnt/c/path/to/pre_trained_weights/B13_rn50_moco_0099.pth',
                                  train_loader=train_loader, num_classes=nbClasses, device=device, label_mapping=label_mapping)
    unet = help_funcs.load_moco_weights() # Load the pre-trained MoCo ResNet into the model
    class_weights = help_funcs.compute_class_weights() # Compute class weights for the cross entropy loss function

    # Get the weights from the MoCo Sentinel-2 pretrained model using the torchgeo library
    weights = ResNet50_Weights.SENTINEL2_ALL_MOCO

    # Load the pretrained weights on the encoder
    encoder_state_dict = weights.get_state_dict()
    unet.encoder.load_state_dict(encoder_state_dict, strict=False)

    # Freeze the encoder layers
    for param in unet.encoder.parameters():
        param.requires_grad = False
    
    # Modify the 1st convolutional layer for 13 channels using the pretrained weights
    with torch.no_grad():

        # Get the initial weights from the state dict
        pre_trained_conv1 = encoder_state_dict['conv1.weight']

        # Get the number of input channels from the 1st convolutional layer
        c_in = pre_trained_conv1.shape[1]
        
        # Compute the number of repeats to cover 13 channels.
        n_repeats = (13 // c_in) + 1
        new_weights = pre_trained_conv1.repeat(1, n_repeats, 1, 1)[:, :13]
        
        # Modify the weights to match the number of input channels (13) and scale them accordingly
        new_weights *= 13 / c_in
        
        # Copy the new weights to the 1st convolutional layer of the UNet encoder
        unet.encoder.conv1.weight.data.copy_(new_weights)

    # append the different learning rates for encoder/decoder to the optimizer initialization
    optimizer = Adam([
        {'params': unet.encoder.parameters(), 'lr': init_lr/10},
        {'params': unet.decoder.parameters(), 'lr': init_lr},
        {'params': unet.head.parameters(), 'lr': init_lr}
    ], lr=init_lr)

    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights) # create the loss function instance

    # calculate steps per epoch for training and validation set
    train_steps = len(train_loader.dataset) // batch_size
    val_steps = len(val_loader.dataset) // batch_size

    # initialize a dictionary to store training loss and metrics
    Hist = {"train_loss": [], "val_loss": [], 
            "train_dice": [], "val_dice": [],
            "train_iou": [], "val_iou": []}

    # Initialize metrics (dice score and intersection over union)
    train_dice = DiceScore(num_classes=nbClasses, input_format='index', average='weighted').to(device)
    val_dice = DiceScore(num_classes=nbClasses, input_format='index', average='weighted').to(device)
    train_iou = MulticlassJaccardIndex(num_classes=nbClasses, average='weighted').to(device)
    val_iou = MulticlassJaccardIndex(num_classes=nbClasses, average='weighted').to(device)

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()

    for e in tqdm(range(epochs)):
        # initialize the total training and validation loss
        totalTrainLoss = 0.0
        totalValLoss = 0.0
        train_dice.reset()
        val_dice.reset()
        train_iou.reset()
        val_iou.reset()

        # set the model in training mode
        unet.train()

    	# loop over the training set
        for (x, y) in train_loader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device).long())

            # Map labels to the contiguous range
            ym = help_funcs.map_labels(labels=y).to(device).long()

            # perform a forward pass and calculate the training loss
            pred = unet(x)
            pred_probs = torch.argmax(pred, dim=1)  # Convert logits to probabilities
            loss = criterion(pred, ym)

            # Update the metrics for training
            train_dice(pred_probs, ym)
            train_iou(pred_probs, ym)

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and update the model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add the loss to the total training loss so far
            totalTrainLoss += loss

    	# switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()

            # loop over the validation set
            for (x, y) in val_loader:
                # send the input to the device
                (x, y) = (x.to(device), y.to(device).long())
        
                # Map labels to the contiguous range
                ym = help_funcs.map_labels(labels=y).to(device).long()

                # make the predictions and calculate the validation loss
                pred = unet(x)
                pred_probs = torch.argmax(pred, dim=1)
                totalValLoss += criterion(pred, ym)

                # Update metrics
                val_dice(pred_probs, ym)
                val_iou(pred_probs, ym)

        # Calculate steps per epoch for training and validation sets
        trainSteps = len(train_loader.dataset) // batch_size
        valSteps = len(val_loader.dataset) // batch_size

    	# calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # Calculate dice score
        train_dice_score = train_dice.compute()
        val_dice_score = val_dice.compute()
        train_iou_score = train_iou.compute()
        val_iou_score = val_iou.compute()

        # update the learning rate based on the validation loss
        scheduler.step(val_dice_score)

        # update training history
        Hist["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        Hist["val_loss"].append(avgValLoss.cpu().detach().numpy())
        Hist["train_dice"].append(train_dice_score.cpu().detach().numpy())
        Hist["val_dice"].append(val_dice_score.cpu().detach().numpy())
        Hist["train_iou"].append(train_iou_score.cpu().detach().numpy())
        Hist["val_iou"].append(val_iou_score.cpu().detach().numpy())

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print(f"Train loss: {avgTrainLoss:.6f}, Dice: {train_dice_score:.4f}, IoU: {train_iou_score:.4f}")
        print(f"Val loss: {avgValLoss:.4f}, Dice: {val_dice_score:.4f}, IoU: {val_iou_score:.4f}")
        
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time")

    # Plot the training and validation Dice and IoU metrics
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(Hist["train_dice"], label="train_dice")
    plt.plot(Hist["val_dice"], label="val_dice")
    plt.title("Training and Validation Dice Score")
    plt.xlabel("Epoch #")
    plt.ylabel("Dice Score")
    plt.legend(loc="lower left")
    plt.savefig('/mnt/c/path/to/train_val_plots/training_val_metrics_bs16.png')
    plt.show()

    # Plot the training and validation loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(help_funcs.filter_outliers(Hist["train_loss"]), label="train_loss")
    plt.plot(help_funcs.filter_outliers(Hist["val_loss"]), label="val_loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig('/mnt/c/path/to/train_val_plots/training_val_loss_bs16.png')
    plt.show()

    # Plot the training and validation intersection over union (IoU)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(Hist["train_iou"], label="train_IoU")
    plt.plot(Hist["val_iou"], label="val_IoU")
    plt.title("Training and Validation IoU")
    plt.xlabel("Epoch #")
    plt.ylabel("IoU")
    plt.legend(loc="lower right")
    plt.savefig('/mnt/c/path/to/train_val_plots/training_val_iou_bs8.png')
    plt.show()

    # save the best model state dictionary to a directory
    torch.save(unet.state_dict(),'/mnt/c/path/to/model_params/unet_model_bs8.pth')