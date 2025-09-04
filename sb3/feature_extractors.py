from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch as th
from torch import nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.models import ResNet18_Weights
from torchvision.models import EfficientNet_B0_Weights
from torchvision import transforms
from config import Hyperparameters

""" 

This includes Feature extractors that utalize pretrained extractor models. E.g. ResNet, Mobilenet, ..."

"""

class ResnetExtractorClass(BaseFeaturesExtractor):
	"""
	Feature extractor that uses ResNet18 architecture.

	:param observation_space: The observation space of the environment.
	:param features_dim: Number of features extracted. This corresponds to the number of units for the last layer.
	:param normalized_image: Whether to assume that the image is already normalized or not.
	"""
	def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False):
		super().__init__(observation_space, features_dim)
		# ResNet18 expects images with three channels (RGB) and a minimum input size of 224x224 pixels

		# Load the pretrained ResNet18 model
		self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
		
		# Replace the classifier part of ResNet18 to match the desired features_dim
		num_ftrs = self.resnet18.fc.in_features
		self.resnet18.fc = nn.Linear(num_ftrs, features_dim)
		self.train_all_layer =  False

		
		# Freeze all layers except for the final classifier
		for param in self.resnet18.parameters():
			param.requires_grad = self.train_all_layer
		for param in self.resnet18.fc.parameters():
			param.requires_grad = True
				
		# Define the preprocessing transformation
		#self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
		
		# Define the preprocessing transformation
		self.my_preprocess = transforms.Compose([
			transforms.Pad((0, 122)),  # Add padding to make the image square (top and bottom)
			transforms.Resize(224), #resize does not cut the image
			transforms.CenterCrop(224), #if resize larger then 224 than center crop to focus on center of image
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		
	def forward(self, observations: th.Tensor) -> th.Tensor:
		# Normalize the input images
		observations = observations.float() / 255.0  # Scale pixel values to [0, 1]
		
		# [batch, channel, time, height, width]
		batch_size, channels, time, height, width = observations.shape
		observations = observations.view(batch_size * time, channels, height, width)  # Flatten time into batch for CNN processing

		# Apply the preprocessing transformations
		observations = self.my_preprocess(observations)
		
		
		# Pass the observations through the ResNet18 model
		out = self.resnet18(observations)
		return out

class CustomMobileNetV3SmallFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using the MobileNetV3 Small architecture.

    This module extracts features from input images and outputs a vector of size `features_dim`.

    :param observation_space: The observation space of the environment.
    :param features_dim: Number of features to output from the extractor (size of the last layer).
    :param normalized_image: Whether the input images are already normalized.
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, normalized_image: bool = False):
        super().__init__(observation_space, features_dim)
        self.train_all_layer = False  # Whether to train all layers of MobileNet

        # Load pretrained MobileNetV3 Small model
        self.mobilenet = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Determine number of input features for the classifier
        num_ftrs = self.mobilenet.classifier[0].in_features
        print("Train full feature network:", self.train_all_layer)

        # Replace classifier to match desired feature dimension
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, int(features_dim)),
            nn.Tanh()
        )

        # Freeze all feature layers if train_all_layer is False
        for param in self.mobilenet.features.parameters():
            param.requires_grad = self.train_all_layer

        # Define preprocessing transforms
        self.my_preprocess = transforms.Compose([
            transforms.Pad((0, 112)),  # Pad to make the image square (top and bottom)
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),  # Crop center if resizing larger than 224
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass of the feature extractor.

        :param observations: Input tensor of shape [batch, channel, time, height, width].
        :return: Extracted features tensor of shape [batch*time, features_dim].
        """
        # Normalize input images to [0, 1]
        observations = observations.float() / 255.0

        # Flatten time dimension into batch dimension for CNN processing
        batch_size, channels, time, height, width = observations.shape
        observations = observations.view(batch_size * time, channels, height, width)

        # Apply preprocessing
        observations = self.my_preprocess(observations)

        # Pass through MobileNetV3 Small
        out = self.mobilenet(observations)
        return out

class CustomEfficientNetFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using the EfficientNet-B0 architecture.

    This module extracts features from input images and outputs a vector of size `features_dim`.

    :param observation_space: The observation space of the environment.
    :param features_dim: Number of features to output from the extractor (size of the last layer).
    :param normalized_image: Whether the input images are already normalized.
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, normalized_image: bool = False):
        super().__init__(observation_space, features_dim)
        self.train_all_layer = False  # Whether to train all layers of EfficientNet

        # Load pretrained EfficientNet-B0 model
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Determine number of input features for the classifier
        num_ftrs = self.efficientnet.classifier[1].in_features
        print("Train full feature network:", self.train_all_layer)

        # Replace the classifier to match desired feature dimension
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, features_dim),
            nn.Tanh()
        )

        # Freeze all feature layers if train_all_layer is False
        for param in self.efficientnet.features.parameters():
            param.requires_grad = self.train_all_layer

        # Define preprocessing transforms
        self.my_preprocess = transforms.Compose([
            transforms.Pad((0, 112)),  # Add padding to make image square
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(224), # optional center crop if resizing larger than 224
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass of the feature extractor.

        :param observations: Input images tensor of shape [batch, channel, time, height, width].
        :return: Extracted features tensor of shape [batch*time, features_dim] and None (placeholder).
        """
        # Normalize input images to [0, 1]
        observations = observations.float() / 255.0

        # Flatten time dimension into batch dimension for CNN processing
        batch_size, channels, time, height, width = observations.shape
        observations = observations.view(batch_size * time, channels, height, width)

        # Apply preprocessing
        observations = self.my_preprocess(observations)

        # Pass through EfficientNet-B0
        out = self.efficientnet(observations)
        return out

class CustomResNetBlocksExtractor(BaseFeaturesExtractor):
	# Drops the last layer of ResNet
	"""
	Feature extractor that uses the first four blocks of the ResNet18 architecture.

	:param observation_space: The observation space of the environment.
	:param features_dim: Number of features extracted. This corresponds to the number of units for the last layer.
	:param normalized_image: Whether to assume that the image is already normalized or not.
	"""
	def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False):
		super().__init__(observation_space, features_dim)
		self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
		self.train_all_layer = False
		self.channels_in = Hyperparameters().NUM_CHANNELS

		if self.channels_in == 4:
			# weights of the original 3-channel conv1
			original_weights = self.resnet18.conv1.weight.data.clone()

			# adjust conv1 to have 4 input channels
			self.resnet18.conv1 = nn.Conv2d(
				in_channels=4,
				out_channels=self.resnet18.conv1.out_channels,
				kernel_size=self.resnet18.conv1.kernel_size,
				stride=self.resnet18.conv1.stride,
				padding=self.resnet18.conv1.padding,
				bias=self.resnet18.conv1.bias,
			)

			# Initialize the 4th channel with the mean of the first 3 channels
			self.resnet18.conv1.weight.data[:, :3, :, :] = original_weights
			self.resnet18.conv1.weight.data[:, 3:4, :, :] = original_weights.mean(dim=1, keepdim=True)
		
		# Extract the first five blocks of ResNet18
		self.resnet_blocks = nn.Sequential(
			self.resnet18.conv1,
			self.resnet18.bn1,
			self.resnet18.relu,
			self.resnet18.maxpool,
			self.resnet18.layer1,
			self.resnet18.layer2,
			self.resnet18.layer3,
			#self.resnet18.layer4
		)
		
		# Freeze all layers
		for param in self.resnet_blocks.parameters():
			param.requires_grad = self.train_all_layer
		
		# Define the new classifier
		self.classifier = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Flatten(),
			nn.Linear(self.resnet18.layer3[-1].bn2.num_features, features_dim)  # Adjust input features
		)
		
		# Unfreeze the classifier
		for param in self.classifier.parameters():
			param.requires_grad = True
		
		# Define the preprocessing transformation
		#self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
		
		if self.channels_in == 4:
			# Freeze the first three channels of the first convolutional layer, only train the depth channel as random initialization
			if not self.train_all_layer: # in case of training all layers, the depth channel is also trained
				self.resnet18.conv1.weight.requires_grad = True

				# Freeze gradients for the first three channels
				with th.no_grad():
					self.resnet18.conv1.weight[:, :3, :, :] = self.resnet18.conv1.weight[:, :3, :, :].detach()

			self.my_preprocess = transforms.Compose([
				transforms.Pad((0, 112)),  # Add padding to make the image square (top and bottom)
				transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR), #bilinear ist standard
				transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.3], std=[0.229, 0.224, 0.225, 0.3])
			])
		else:
			self.my_preprocess = transforms.Compose([
				transforms.Pad((0, 112)),  # Add padding to make the image square (top and bottom)
				transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR), #bilinear ist standard
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
		
	def forward(self, observations: th.Tensor) -> th.Tensor:
		# Normalize the input images
		observations = observations.float() / 255.0  # Scale pixel values to [0, 1]
		
		# [batch, channel, time, height, width]
		batch_size, channels, time, height, width = observations.shape
		observations = observations.view(batch_size * time, channels, height, width)  # Flatten time into batch for CNN processing

		# Apply the preprocessing transformations
		observations = self.my_preprocess(observations)
		
		# Pass the observations through the ResNet18 blocks
		features = self.resnet_blocks(observations)
		
		# Pass the features through the classifier
		out = self.classifier(features)
		return out