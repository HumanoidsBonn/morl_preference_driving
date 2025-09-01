from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from stable_baselines3.common.type_aliases import TensorDict
import numpy as np
import torch as th
from torch import nn
from typing import Dict, List, Tuple, Type, Union
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from cnn import CustomCNN
from feature_extractors import ResnetExtractorClass, CustomMobileNetV3SmallFeaturesExtractor, CustomResNetBlocksExtractor, CustomEfficientNetFeaturesExtractor
# from llm_extractor import LLMextractor
from config import Hyperparameters

class CustomCombinedExtractor(BaseFeaturesExtractor):
	"""
	Select the feature extractor to use here.

	Ensure that the input to the Extractor:
	- Is properly normalized
	- Has the correct shape (e.g., [channels, time steps, height, width] for RGB sequences) by setting values in config

	Combined features extractor for Dict observation spaces.
	Builds a features extractor for each key of the space. Input from each space
	is fed through a separate submodule (CNN or MLP, depending on input shape),
	the output features are concatenated and fed through additional MLP network ("combined").

	:param observation_space:
	:param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
		256 to avoid exploding network sizes.
	:param normalized_image: Whether to assume that the image is already normalized
		or not (this disables dtype and bounds checks): when True, it only checks that
		the space is a Box and has 3 dimensions.
		Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
	"""
	def __init__(
		self,
		observation_space: spaces.Dict,
		cnn_output_dim: int = 48,
		normalized_image: bool = False,
	) -> None:
		super().__init__(observation_space, features_dim=1)
		self.tmp_config = Hyperparameters()
		self.output_dim = self.tmp_config.feature_dim
		self.llm_outdim = 4
		extractors: Dict[str, nn.Module] = {}

		total_concat_size = 0
		llm_input_dim = 0
		for key, subspace in observation_space.spaces.items():
			if self.custom_is_image_space(subspace, normalized_image=normalized_image):
				
				""" select here """
				
				# extractors[key] = CustomCNN(subspace, features_dim=self.output_dim, normalized_image=normalized_image)
				# extractors[key] = CustomResNetFeaturesExtractor(subspace, features_dim=self.output_dim, normalized_image=normalized_image)
				# extractors[key] = CustomMobileNetV3SmallFeaturesExtractor(subspace, features_dim=self.output_dim, normalized_image=normalized_image)
				# extractors[key] = CustomEfficientNetFeaturesExtractor(subspace, features_dim=self.output_dim, normalized_image=normalized_image)
				extractors[key] = CustomResNetBlocksExtractor(subspace, features_dim=self.output_dim, normalized_image=normalized_image) #pretrained weights
			
				total_concat_size += self.output_dim
			else:
				# The observation key is a vector, flatten it if needed
				extractors[key] = nn.Flatten()
				size = get_flattened_obs_dim(subspace)
				total_concat_size += size
				llm_input_dim += size

		self.extractors = nn.ModuleDict(extractors)

		# Update the features dim manually
		self._features_dim = total_concat_size

	def forward(self, observations: TensorDict) -> th.Tensor:
		encoded_tensor_list = []

		for key, extractor in self.extractors.items():
			if key == "camera":
				out = extractor(observations[key])
				encoded_tensor_list.append(out)
			else:
				encoded_tensor_list.append(extractor(observations[key]))

		return th.cat(encoded_tensor_list, dim=1)
	
	# def forward(self, observations: TensorDict) -> th.Tensor:
	# 	encoded_tensor_list = []

	# 	for key, extractor in self.extractors.items():
	# 		out = extractor(observations[key])
	# 		print(type(observations[key]), observations[key].shape)
	# 		# print(f"{key} output shape: {out.shape}")  # <- check each feature tensor shape
	# 		encoded_tensor_list.append(out)


	# 	concatenated = th.cat(encoded_tensor_list, dim=1)
	# 	print(f"Concatenated output shape: {concatenated.shape}")  # <- final shape
	# 	return concatenated
		
	def custom_is_image_space(
		self,
		observation_space: spaces.Space,
		check_channels: bool = False,
		normalized_image: bool = False,
	) -> bool:
		"""
		Check if a observation space has the shape, limits and dtype
		of a valid image.
		The check is conservative, so that it returns False if there is a doubt.

		Valid images: RGB, RGBD, GrayScale with values in [0, 255]

		:param observation_space:
		:param check_channels: Whether to do or not the check for the number of channels.
			e.g., with frame-stacking, the observation space may have more channels than expected.
		:param normalized_image: Whether to assume that the image is already normalized
			or not (this disables dtype and bounds checks): when True, it only checks that
			the space is a Box and has 3 dimensions.
			Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
		:return:
		"""
		check_dtype = check_bounds = not normalized_image
		if isinstance(observation_space, spaces.Box) and (observation_space.shape == self.tmp_config.camera_shape): #adjusted to use time time dimension
			# Check the type
			if check_dtype and observation_space.dtype != np.uint8:
				return False

			# Check the value range
			incorrect_bounds = np.any(observation_space.low != 0) or np.any(observation_space.high != 255)
			if check_bounds and incorrect_bounds:
				return False

			# Skip channels check
			if not check_channels:
				return True
		return False
