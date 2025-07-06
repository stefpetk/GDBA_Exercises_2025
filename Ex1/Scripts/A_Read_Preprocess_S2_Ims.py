import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_origin
import numpy as np
import glob
import os
import zipfile

class PreprocessRasters:
    """
    A class definition for handling the following tasks:
    1. Reprojecting input rasters to new coordinate reference systems (CRS) or executing resampling operations.
    2. Reading and preprocessing Sentinel-2 images and ground truth maps.
    3. Crop and align the preprocessed Sentinel-2 images with the ground truth maps to pixel level."""
    def __init__(self):
        pass
    
    def reproject_raster(self, src_path, dst_path, dst_crs, target_resolution=None, target_bounds=None, 
                         res_method=Resampling.bilinear, force_overwrite=False):
        """
        Reproject a raster to a new CRS with optional target resolution and extent or just resample it if needed.
        Parameters:
        - src_path: path to source raster
        - dst_path: path for output raster
        - dst_crs: target CRS as EPSG code
        - target_resolution: tuple of (x_res, y_res) in target CRS units
        - target_bounds: tuple of (left, bottom, right, top) in target CRS
        - res_method: resampling method (default is Resampling.bilinear, can also be Resampling.nearest or Resampling.cubic)
        * Force_overwrite (bool): If True, will overwrite existing files. Default is False.
        """

        with rasterio.open(src_path) as src:
            # Calculate transform and dimensions
            if target_bounds is None:
                # Use the default bounds (transformed from source)
                transform, width, height = calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=target_resolution
                )
            else:
                # Use specified bounds
                left, bottom, right, top = target_bounds
                # Use the target bounds and resolution to calculate the transform
                if target_resolution:
                    xres, yres = target_resolution
                    width = int((right - left) / xres)
                    height = int((top - bottom) / yres)
                    transform = from_origin(left, top, xres, yres)
                else:
                    # Keep approximately same resolution as source if no target resolution is given
                    src_res = src.transform[0]
                    width = int((right - left) / src_res)
                    height = int((top - bottom) / src_res)
                    transform = from_origin(left, top, (right-left)/width, (top-bottom)/height)

            # Update metadata for the destination file
            kwargs = src.meta.copy()
            kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            })

            # Create destination dataset
            with rasterio.open(dst_path, 'w', **kwargs) as dst:
                # Reproject each band or/and resample to target resolution
                for i in range(1, src.count + 1):
                    reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=res_method)

        # Delete the source file if force_overwrite=True 
        if force_overwrite and os.path.exists(src_path):
            os.remove(src_path)

    def read_and_preprocess_S2_images(self, images_path):
        """
        Reads and preprocesses (only resampling needed for bands with pixel resolution != 10m) Sentinel-2 sattelite imagery.

        Parameters:
        images_path (str): Path to the folder containing the Sentinel-2 images with their respective bands.
        (Note: The bands are stored in specific folders according to the pixel resolution)
        """

        # Unzip the Sentinel-2 satellite imagery data to a new folder 
        s2_ims_paths = glob.glob(images_path+'/*.zip')
        extr_dir = 'S2_Ims_Unzipped'
        joined_path = os.path.join('/'.join(images_path.split('/')[:-1]), extr_dir) 

        # Create the directory if it doesn't exist
        if not os.path.exists(joined_path):
            os.mkdir(joined_path)

        elif os.path.exists(joined_path):
            print(f"Folder '{joined_path}' already exists. Skipping extraction if the contents have already been extracted.")

        # Extract the zip file contents to the new directory
        for zip_path in s2_ims_paths:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(joined_path)

        # Get the identifying string for each S2 image and the paths to the bands
        id_strs = [s2_im_path.split('/')[-1].split('_')[-2] for s2_im_path in s2_ims_paths]
        bands_data_dir = [glob.glob(joined_path+f'/*_{id_str}_*/GRANULE/*/IMG_DATA/*.jp2') for id_str in id_strs]

        # Delete the TCI band from the bands list (not needed for training)
        for img_bands in bands_data_dir:
            img_bands[:] = [band_path for band_path in img_bands if 'TCI' not in band_path]
            for band_path in img_bands:
                if 'TCI' in band_path:
                    os.remove(band_path)
                    
        # Resample bands to 10m resolution
        for img_bands in bands_data_dir:  # Iterate over each Sentinel-2 image's bands
            for band_path in img_bands:

                # Check band resolution from filename (e.g., B05_20m.jp2 -> 20m)
                band_name = os.path.basename(band_path).split('_')[-1]
                resolution = band_name.split('.')[0].split('_')[-1]  # Extract resolution from filename
                bands_10m = ['B02', 'B03', 'B04', 'B08']  # list with the bands that don't need resampling

                if resolution not in bands_10m:
                    # Resample to 10m, overwriting the original file
                    self.reproject_raster(
                        src_path=band_path,  # Use the band itself as source
                        dst_path=band_path.replace('.jp2', '_pr.jp2'),  # Overwrite original
                        dst_crs="EPSG:32634",
                        target_resolution=(10, 10),
                        res_method=Resampling.bilinear,
                        force_overwrite=True
                    )
    
    def read_preprocess_gtruth(self, gt_path, sentinel2_paths, output_dir):
        """
        1. Reproject ground truth map to match Sentinel-2 CRS (EPSG:32634) and resolution (10m)
        2. Split into tiles matching the intersection of Sentinel-2 and ground truth extents

        Parameters:
        gt_path (str): Path to the ground truth map.
        sentinel2_paths (list): List of paths to Sentinel-2 images.
        output_dir (str): Directory to save the reprojected and split ground truth maps.
        """
        # Reproject ground truth map
        reproj_gt_path = os.path.join(output_dir, "gt_reprojected.tif")
        self.reproject_raster(
            src_path=gt_path,
            dst_path=reproj_gt_path,
            dst_crs="EPSG:32634",
            target_resolution=(10, 10),
            res_method=Resampling.nearest,
        )

        # Get bounds of the reprojected ground truth map
        with rasterio.open(reproj_gt_path) as gt_src:
            gt_bounds = gt_src.bounds

        # Split ground truth map to intersecting extents with each Sentinel-2 image
        for s2_path in sentinel2_paths:
            with rasterio.open(s2_path) as s2_src:
                s2_bounds = s2_src.bounds

            # Calculate intersection bounds
            intersection_left = max(gt_bounds.left, s2_bounds.left)
            intersection_right = min(gt_bounds.right, s2_bounds.right)
            intersection_bottom = max(gt_bounds.bottom, s2_bounds.bottom)
            intersection_top = min(gt_bounds.top, s2_bounds.top)

            # Check if there is a valid intersection
            if (intersection_left >= intersection_right) or (intersection_bottom >= intersection_top):
                print(f"No overlap between {s2_path} and ground truth. Skipping.")
                continue
            
            # Calculate target bounds for clipping the ground truth maps to the extents of the Sentinel-2 images 
            target_bounds = (intersection_left, intersection_bottom, intersection_right, intersection_top)

            # Generate output path
            base_name = os.path.splitext(os.path.basename(s2_path))[0]
            clipped_gt_path = os.path.join(output_dir, f"gt_{base_name}.tif")

            # Clip to the intersection bounds
            self.reproject_raster(
                src_path=reproj_gt_path,
                dst_path=clipped_gt_path,
                dst_crs="EPSG:32634",
                target_bounds=target_bounds,
                target_resolution=(10, 10),
                res_method=Resampling.nearest)

    def get_true_bounds(self, gt_path):
        """
        Compute the bounding box of valid (non-NoData) pixels in each ground truth raster.
        Returns (left, bottom, right, top) in the raster's CRS.
        """
        with rasterio.open(gt_path) as src:
            # Read the first band and create a mask where data is valid
            data = src.read(1)
            mask = (data != src.nodata)  # Adjust if NoData is defined differently

            # Find rows/columns with valid data
            valid_rows = np.where(np.any(mask, axis=1))[0]
            valid_cols = np.where(np.any(mask, axis=0))[0]

            if len(valid_rows) == 0 or len(valid_cols) == 0:
                raise ValueError("No valid data in the ground truth raster.")

            # Get pixel coordinates of the bounding box
            min_row = valid_rows.min()
            max_row = valid_rows.max()
            min_col = valid_cols.min()
            max_col = valid_cols.max()
            
            # Convert pixel coordinates to geographic coordinates
            left, bottom = src.transform * (min_col, max_row)
            right, top = src.transform * (max_col, min_row)

            return (left, bottom, right, top)

    def clip_preprocessed_s2(self, s2_paths, gt_paths):
        """
        Clip Sentinel-2 images to match the extents of preprocessed ground truth maps.

        Parameters:
        - s2_paths (list): List of lists. Each sublist contains paths to Sentinel-2 bands 
                           for one image (e.g., ["B02.jp2", "B03.jp2", ...]).
        - gt_paths (list): List of paths to preprocessed ground truth rasters (one per Sentinel-2 image).
        """
        # Validate input lengths
        if len(s2_paths) != len(gt_paths):
            raise ValueError("s2_paths and gt_paths must have the same number of elements")

        for idx, (gt_path, s2_band_paths) in enumerate(zip(gt_paths, s2_paths)):
            # Get TRUE bounds of the ground truth (excluding NoData)
            true_bounds = self.get_true_bounds(gt_path)
            
            # Get ground truth bounds and metadata
            with rasterio.open(gt_path) as gt_src:
                gt_crs = gt_src.crs
                gt_res = gt_src.res  # Should be (10, 10) meters

            # Clip each Sentinel-2 band to the ground truth bounds
            for s2_band_path in s2_band_paths:
                self.reproject_raster(
                    src_path=s2_band_path,
                    dst_path=s2_band_path.replace('.jp2', '_final.jp2'),  # Overwrite original file
                    dst_crs=gt_crs,  # Ensure alignment with ground truth
                    target_bounds=true_bounds,
                    target_resolution=gt_res,  # Use ground truth resolution (10m)
                    res_method=Resampling.nearest,
                    force_overwrite=True  # Force deletion of existing file
                )