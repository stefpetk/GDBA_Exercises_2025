import torch
from torch.nn import ReflectionPad2d
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
from torchvision.models.resnet import ResNet, Bottleneck

class Block(Module):
	"""A block which consists of two convolutional layers and a ReLU activation function.
	The class inherits from the PyTorch Module class and implements the forward pass."""

	def __init__(self, inChannels, outChannels):
		"""Args:
			inChannels (int): Number of input channels.
			outChannels (int): Number of output channels.
		"""
		super().__init__()

		# store the convolutional (kernel size = 3) and ReLU layers of each block
		self.pad = ReflectionPad2d(1) # apply reflection padding to the input tensor
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3)		

	def forward(self, x):
		"""Args:
			x (Tensor): Input tensor of shape (batch_size, in_channels, height, width)."""
		# return the forward pass of the block
		x = self.pad(x)
		return self.conv2(self.relu(self.conv1(x)))

class MoCoResNetEncoder(ResNet):
	"""A modified ResNet encoder that uses a custom number of input channels."""

	def __init__(self, in_channels=13):
		"""Args:
			in_channels (int): Number of input channels. Default is 13 (Sentinel-2 bands).
		"""
		super().__init__(block=Bottleneck, layers=[3, 4, 6, 3])
		
		# Modify first convolutional layer for 13-channel input
		self.conv1 = Conv2d(in_channels, 64, kernel_size=7, 
							stride=2, padding=3, bias=False)
		
		# Remove original fully connected layer
		del self.fc

	def forward(self, x):
		""""Args:
			x (Tensor): Input tensor of shape (batch_size, in_channels, height, width)."""
		# Forward pass with intermediate outputs
		features = []
		x = self.conv1(x)
		x = self.bn1(x) # batch normalization operation after convolution
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x); features.append(x)  # 256 channels
		x = self.layer2(x); features.append(x)  # 512 channels
		x = self.layer3(x); features.append(x)  # 1024 channels
		x = self.layer4(x); features.append(x)  # 2048 channels

		return features


class Decoder(Module):
	"""A decoder that upsamples (after each ) the input features and concatenates them with the
	encoder features. The class inherits from the PyTorch Module class and implements
	the forward pass through the decoder."""

	def __init__(self, channels=(128, 64, 32)):
		"""Args:
			channels (tuple): Number of channels for each layer in the decoder.
		"""
		super().__init__()

		# Initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])

	def forward(self, x, encFeatures):
		"""Args:
			x (Tensor): Input tensor of shape (batch_size, channels, height, width).
			encFeatures (list): List of encoder features from the ResNet encoder."""
		
		# Loop through the number of channels
		for i in range(len(self.channels) - 1):
			
            # Pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)

			# Crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)

		# Return the final decoder output
		return x

	def crop(self, encFeatures, x):
		""""Args:
			encFeatures (Tensor): Encoder features of shape (batch_size, channels, height, width).
			x (Tensor): Input tensor of shape (batch_size, channels, height, width)."""
		
		# Grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)

		# return the cropped features
		return encFeatures


class UNet(Module):
	def __init__(self, encChannels=(13, 32, 64, 128), decChannels=(128, 64, 32), nbClasses=8, retainDim=True, outSize=(256, 256)):
		"""Args:
			encChannels (tuple): Number of channels for each layer in the encoder.
			decChannels (tuple): Number of channels for each layer in the decoder.
			nbClasses (int): Number of output classes.
			retainDim (bool): Whether to retain the original dimensions of the input.
			outSize (tuple): Output size of the segmentation map."""
		
		super().__init__()

		# Initialize encoder with ResNet50
		self.encoder = MoCoResNetEncoder(in_channels=encChannels[0])

		# Adjust decoder channels to match ResNet50 features
		self.decoder = Decoder(channels=(2048, 1024, 512, 256))

		# Final convolution for the segmentation map
		self.head = Conv2d(256, nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize

	def forward(self, x):
		"""Args:
			x (Tensor): Input tensor of shape (batch_size, channels, height, width)."""
		encFeatures = self.encoder(x)
		decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
		seg_map = self.head(decFeatures)
		if self.retainDim:
			seg_map = F.interpolate(seg_map, self.outSize)
		return seg_map