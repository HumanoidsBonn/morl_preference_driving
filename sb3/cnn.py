from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space

class CustomCNN(BaseFeaturesExtractor):
	"""
	:param observation_space:
	:param features_dim: Number of features extracted.
		This corresponds to the number of unit for the last layer.
	:param normalized_image: Whether to assume that the image is already normalized
		or not (this disables dtype and bounds checks): when True, it only checks that
		the space is a Box and has 3 dimensions.
		Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
	"""

	def __init__(
		self,
		observation_space: gym.Space,
		features_dim: int = -1, #wird vorher überschrieben in customFeatureExtractor
		normalized_image: bool = False,
	) -> None:
		assert isinstance(observation_space, spaces.Box), (
			"NatureCNN must be used with a gym.spaces.Box ",
			f"observation space, not {observation_space}",
		)
		super().__init__(observation_space, features_dim)
		print("observation_space", observation_space)

		shape = observation_space.shape
		self.channels = shape[0]  
		self.time_steps = shape[1]  
		self.height = shape[2]    
		self.width = shape[3]    
		self.filters = 128
		self.n_flatten = -1 #feature dimension
		n_input_channels = observation_space.shape[0]
		self.train_all_layer = True
		print("CNN n_input_channels", n_input_channels)

		self.cnn = nn.Sequential(
			nn.Conv2d(n_input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
			
			nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
			
			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

			nn.Conv2d(128, self.filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.BatchNorm2d(self.filters),
			nn.LeakyReLU(),
			nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
			nn.AdaptiveAvgPool2d((6, 11))
		)

		with th.no_grad():
			sample_input = th.as_tensor(observation_space.sample()[None]).float()
			output = self.cnn(sample_input[:,:,0,:,:])
			# print("output cnn shape", output.shape)
			self.n_flatten = output.shape[1] * output.shape[2] * output.shape[3] # feature_channels * new_h * new_w
			print("CNN_feature_dim: ", self.n_flatten)
			print("CNN_img_shape: ", output.shape[2], output.shape[3])

		self.fc = nn.Sequential(
			nn.Linear(self.n_flatten, 512),
            nn.LeakyReLU(),
			nn.Linear(512, features_dim),
			nn.Tanh()
		)

		self.apply(lambda m: CustomCNN.init_weights_sb3(m, gain=1.2))

	def forward(self, observations: th.Tensor) -> th.Tensor:
		x = observations.float() / 255.0  # Umwandeln von uint8 in float32 und Normalisieren der Eingabedaten auf den Bereich [0, 1]
		
		# noise = th.randn_like(x) * 0.01  # Rauschen mit einer Standardabweichung von 0.05
		# x = x + noise
		
		# [batch, channel, time, height, width]
		batch_size, channels, time, height, width = x.shape

		x = x.view(batch_size * time, channels, height, width)  # Flatten time into batch for CNN processing
		x = self.cnn(x)  # Apply CNN

		#x = x.view(batch_size, time, -1)  # Reshape back to include time
		x = x.view(batch_size, -1)  # Flatten

		# Abschluss mit einer voll verbundenen Schicht
		out = self.fc(x)

		return out 
	
	def init_weights_sb3(m, gain):
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
			th.nn.init.orthogonal_(m.weight, gain=gain)
			if m.bias is not None:
				th.nn.init.constant_(m.bias, 0)