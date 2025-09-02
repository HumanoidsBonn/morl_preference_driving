from gymnasium import spaces
from torch import nn
import torch as th
from typing import Any, Dict, List, Optional, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, get_actor_critic_arch, create_mlp

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.policies import BasePolicy#, ContinuousCritic

class CustomActor(BasePolicy):
	"""
	Actor network (policy) for TD3.

	:param observation_space: Obervation space
	:param action_space: Action space
	:param net_arch: Network architecture
	:param features_extractor: Network to extract features
		(a CNN when using images, a nn.Flatten() layer otherwise)
	:param features_dim: Number of features
	:param activation_fn: Activation function
	:param normalize_images: Whether to normalize images or not,
		 dividing by 255.0 (True by default)
	"""

	def __init__(
		self,
		observation_space: spaces.Space,
		action_space: spaces.Box,
		net_arch: List[int],
		features_extractor: nn.Module,
		features_dim: int,
		activation_fn: Type[nn.Module] = nn.ReLU, #LeakyReLU
		normalize_images: bool = True,
	):
		super().__init__(
			observation_space,
			action_space,
			features_extractor=features_extractor,
			normalize_images=normalize_images,
			squash_output=True,
		)

		self.net_arch = net_arch
		self.features_dim = features_dim
		self.activation_fn = activation_fn

		self.train_feature = False
		# print("Use the actor to train the feature extractor: ", self.train_feature)

		action_dim = get_action_dim(self.action_space)
		actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)

		# Deterministic action
		self.mu = nn.Sequential(*actor_net)
		
	def _get_constructor_parameters(self) -> Dict[str, Any]:
		data = super()._get_constructor_parameters()

		data.update(
			dict(
				net_arch=self.net_arch,
				features_dim=self.features_dim,
				activation_fn=self.activation_fn,
				features_extractor=self.features_extractor,
			)
		)
		return data

	def forward(self, obs, get_hidden_activations=False):
		with th.set_grad_enabled(self.train_feature):
			features = self.extract_features(obs, self.features_extractor)
		actions = self.mu(features)
	
		if get_hidden_activations:
			hidden_activations = self.mu[:4](features)
			return actions, hidden_activations
		else:
			return actions

	def _predict(self, observation: th.Tensor,  get_hidden_activations=False, deterministic: bool = False) -> th.Tensor:
		if get_hidden_activations:
			# Not fully supported; must implement a catch for the tuple return value!
			return self(observation, get_hidden_activations=True)
		else:
			return self(observation)
