from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import MulticlassJaccardIndex
import numpy as np
from scipy.ndimage import distance_transform_edt
import rasterio
import torch
import torch.nn as nn
from tqdm import tqdm

class PredTest:
	def __init__(self, patches_coords, margin, device, model, original_shape, label_map, patch_size=256):
		"""
		patches_coords: List of tuples (row_start, col_start, tensor),
        where tensor is a torch.Tensor of shape (C, H, W) without margins.
		margin: number of pixels to pad on each side before inference
		device: torch device
		model: U-net segmentation model
		original_shape: (height, width) of the full output map
		label_map: dictionary to map class indices to labels
		patch_size: size of the central region per patch (without margins)
		"""
		self.patches_coords = patches_coords
		self.margin = margin
		self.device = device
		self.model = model
		self.original_shape = original_shape
		self.patch_size = patch_size
		self.label_map = label_map
	
	def nearest_fill(self, arr):
		"""
		Replace zeros in arr by nearest non-zero neighbor (nearest-label interpolation).
		Args:
		    arr: [H, W] numpy array with prediction indices
		"""

		mask = (arr == 0)  # Create a mask for pixels with value 0
		if not mask.any():
		    return arr  # If no zeros, return the original array
		
		# Compute the distance transform and indices of the nearest non-zero pixels
		_, indices = distance_transform_edt(mask, return_indices=True)
		
		# Replace zeros with the values of the nearest non-zero pixels
		filled = arr[*tuple(indices)].astype(np.uint8)
		return filled

	def stitch_patches(self, preds_list):
		"""
		Stitch cropped patches into the full-size array using dynamic sizes.
		Args:
		    preds_list: List of tuples (row_idx, col_idx, pred), where pred is a torch.Tensor of shape (C, H, W) with margins
		"""
		H_full, W_full = self.original_shape
		full = np.zeros((H_full, W_full), dtype=np.uint8) # initialize full array with zeros

		for row_idx, col_idx, pred in preds_list:			
			# pixel start with margin offset
			row_im = row_idx * self.patch_size
			col_im = col_idx * self.patch_size
			full[row_im:row_im+self.patch_size, col_im:col_im+self.patch_size] = pred.cpu().numpy().astype(np.uint8)

		# interpolate zeros
		full = self.nearest_fill(full)

		if self.label_map is not None:
		    full_array = np.vectorize(self.label_map.get)(full) # map class indices to labels

		return full_array
	
	def prediction_inference(self, color_ill=True):
		"""Run inference with margin-cropping, then stitch the patches to produce the final predictions map."""
		self.model.eval()
		pred_patches = []	
		with tqdm(total=len(self.patches_coords), desc="Predicting patches") as pbar:
			for (row_idx, col_idx, patch) in self.patches_coords:
				# patch: torch.Tensor (C, H, W) without margins

				inp = patch.unsqueeze(0).to(self.device)
				
				with torch.no_grad():
					logits = self.model(inp)  # compute logits
					pred = logits.argmax(dim=1).squeeze(0).cpu() 
					pred = pred[self.margin:-self.margin, self.margin:-self.margin]  # remove margins

					# Add zero padding to the prediction maps
					pred_padded = torch.nn.functional.pad(
					pred, (self.margin, self.margin, self.margin, self.margin),
					mode='constant', value=-1)
					pred_patches.append((row_idx, col_idx, pred_padded))
				pbar.update(1)

		# Clean up to release memory
		if torch.cuda.is_available():
		    torch.cuda.empty_cache()

		# Stitch & map labels
		stitched = self.stitch_patches(pred_patches)

		return self.apply_color_map(stitched) if color_ill else stitched

	def apply_color_map(self, pred_array):
		"""Convert class indices to RGB colors
		Args:
			pred_array: [H, W] numpy array with class labels"""
		color_map = {
			10: (0, 100, 0), # Tree cover
			20: (255, 187, 34), # Shrubland
			30: (255, 255, 76), # Grassland
			40: (240, 150, 255), # Cropland
			50: (250, 0, 0), # Built-up
			60: (180, 180, 180), # Bare/sparse vegetation
			80: (0, 100, 200), # Permanent water bodies
			90: (0, 150, 160) # Herbaceous wetland
		}

		rgb = np.zeros((*pred_array.shape, 3), dtype=np.uint8) # Initialize RGB array
		for class_val, color in color_map.items():
			rgb[pred_array == class_val] = color # Assign colors to class labels

		return rgb.transpose(2, 0, 1)  # Convert to CHW
	
	def calculate_receptive_field(self):
		"""
		Calculate the receptive field of the model.
		"""	
		receptive_field = 1
		for layer in self.model.encoder.children():
			if isinstance(layer, nn.Conv2d):
				stride = layer.stride[0]
				kernel = layer.kernel_size[0]
				receptive_field = receptive_field * stride + (kernel - stride)	
		return receptive_field

	def save_georeferenced_tif(self, array, crs, transform, output_path, compression='lzw',
							   color_ill=True):
		"""
		Save a numpy array as a georeferenced TIFF.

		Args:
		    array: [H, W] numpy array
		    crs: CRS (e.g., "EPSG:32634")
		    transform: Affine transform from the original image
		    output_path: Path to save the TIFF
		"""
		if color_ill:
			with rasterio.open(
				output_path,
				'w',
				driver='GTiff',
				height=array.shape[1],
				width=array.shape[2],
				count=3,
				dtype='uint8',
				crs=crs,
				transform=transform,
				compression=compression) as dst:
				dst.write(array, [1, 2, 3])
		else:
			with rasterio.open(
				output_path,
				'w',
				driver='GTiff',
				height=array.shape[0],
				width=array.shape[1],
				count=1,
				dtype='uint8',
				crs=crs,
				transform=transform,
				compression=compression) as dst:
				dst.write(array, 1)

def eval_prediction(self, gt_arr, pred_arr=None):
		"""
		Evaluate the prediction against the ground truth.
		Args:
		    gt_arr: Ground truth array
			pred_arr: Prediction array (optional). If not provided, the model will run inference.
		Returns:
		    dice score: Dice score metric of the prediction
		    iou: Intersection over Union metric of the prediction
		"""

		# First read the ground truth array and convert it to a numpy array
		with rasterio.open(gt_arr) as src:
			gt_arr = src.read(1)
		
		if not pred_arr is None:
			with rasterio.open(pred_arr) as src:
				pred_arr = src.read(1)
		else:
			pred_arr = self.prediction_inference(color_ill=False)

		gt_arr = gt_arr[:pred_arr.shape[0], :pred_arr.shape[1]]  # Ensure the ground truth array matches the prediction shape

		# Mask pixels with 0 values and optionally assign them to another class (e.g., 10)
		mask = (gt_arr == 0)  # Create a mask for pixels with value 0
		gt_arr[mask] = 10     # Assign these pixels to class 10

		# Dynamically adjust num_classes
		num_classes = len(np.unique(pred_arr))

		inv_mapping = {v:k for k, v in self.label_map.items()}

		gt_arr = np.vectorize(inv_mapping.get)(gt_arr)
		pred_arr = np.vectorize(inv_mapping.get)(pred_arr)

		# Define the metrics instances
		dice_score = DiceScore(num_classes=num_classes, input_format='index', average=None).cpu()
		iou = MulticlassJaccardIndex(num_classes=num_classes).cpu()

		# Compute the metrics
		dice_score_comp = dice_score(torch.tensor(pred_arr).long(), torch.tensor(gt_arr).long()).tolist()
		iou_comp = iou(torch.tensor(pred_arr).long(), torch.tensor(gt_arr).long()).tolist()

		# Create a dictionary to match the class indices to their descriptions
		classes_dict = {10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland', 50: 'Built-up', 60: 'Bare/sparse vegetation', 
			 	   80: 'Permanent water bodies', 90: 'Herbaceous wetland'} 
		
		# Assign the metrics to the respective classes
		dice_score_dict = {class_name: dice_score_val for class_name, dice_score_val in zip(classes_dict.values(), dice_score_comp)}
		iou_dict = {class_name: iou_comp_val for class_name, iou_comp_val in zip(classes_dict.values(), iou_comp)}

		return dice_score_dict, iou_dict