import torch
import numpy as np

class HelperFunctions():
    """
    A class to handle loading weights from a MoCo checkpoint into a modified ResNet model,
    mapping labels to a contiguous range, and computing class weights for a dataset.
    """
    def __init__(self, model, checkpoint_path, train_loader, num_classes, device, label_mapping):
        """"
        Args:
            model (torch.nn.Module): The modified ResNet model.
            checkpoint_path (str): Path to the MoCo checkpoint file.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            num_classes (int): Number of classes in the dataset.
            device (torch.device): Device to load the model onto (CPU or GPU).
            label_mapping (dict): Mapping from original labels to contiguous labels.
        """
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.train_loader = train_loader
        self.num_classes = num_classes
        self.device = device
        self.label_mapping = label_mapping

    def load_moco_weights(self):
        # Load MoCo checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Extract state_dict from checkpoint
        state_dict = {}
    
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('encoder_q.'):
                state_dict[k[10:]] = v
    
        # Load weights into modified ResNet
        msg = self.model.encoder.load_state_dict(state_dict, strict=False)
        #print(f"Loaded MoCo weights: {msg}")
        return self.model
	
    def map_labels(self, labels):
        """
        Maps original labels to a contiguous range based on the provided label mapping.
        """
        mapped_labels = labels.clone()
        for original, mapped in self.label_mapping.items():
            mapped_labels[labels == original] = mapped
        return mapped_labels

    def compute_class_weights(self):
        """
        Computes class weights based on the training dataset to handle class imbalance.
        """
        # Initialize array to collect class counts
        class_counts = np.zeros(self.num_classes, dtype=np.int64)

        # Iterate through training data (only once!)
        for _, y in self.train_loader:
            # Map labels to the contiguous range
            y = self.map_labels(y)

            # Convert mask to numpy array and flatten
            mask = y.cpu().numpy().flatten()
    
            # Count class occurrences
            counts = np.bincount(mask, minlength=self.num_classes)
            class_counts += counts

        # Handle potential zero-count classes
        epsilon = 1e-7
        class_counts = np.maximum(class_counts, epsilon)

        # Compute class weights directly
        total_pixels = class_counts.sum()
        weights = total_pixels / (self.num_classes * class_counts)
        weights /= weights.min()  # Normalize to prevent extreme values

        return torch.tensor(weights, dtype=torch.float32).to(self.device)
    
    def filter_outliers(self, data):
        """
        Filters outliers from the data using the IQR.
        """
        # Compute the first and third percentiles
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
    
        # Define the lower and upper bounds for outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
    
        # Replace outliers with NaN
        filtered_data = np.where((data < lower_bound) | (data > upper_bound), np.nan, data)
        return filtered_data