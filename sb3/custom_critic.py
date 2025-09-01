from typing import List, Type, Tuple
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from config import Hyperparameters

class CustomContinuousCritic(BaseModel):
	"""
	Critic network(s) for DDPG/SAC/TD3 algorithms.
	Extended for PDMORL: supports multi-dimensional critics based on Stable-Baselines3 (SB3).


	It represents the action-state value function (Q-value function).
	Compared to A2C/PPO critics, this one represents the Q-value
	and takes the continuous action as input. It is concatenated with the state
	and then fed to the network which outputs a single value: Q(s, a).
	For more recent algorithms like SAC/TD3, multiple networks
	are created to give different estimates.

	By default, it creates two critic networks used to reduce overestimation
	thanks to clipped Q-learning (cf TD3 paper).

	:param observation_space: Obervation space
	:param action_space: Action space
	:param net_arch: Network architecture
	:param features_extractor: Network to extract features
		(a CNN when using images, a nn.Flatten() layer otherwise)
	:param features_dim: Number of features
	:param activation_fn: Activation function
	:param normalize_images: Whether to normalize images or not,
		 dividing by 255.0 (True by default)
	:param n_critics: Number of critic networks to create.
	:param share_features_extractor: Whether the features extractor is shared or not
		between the actor and the critic (this saves computation time)
	"""

	def __init__(
		self,
		observation_space: spaces.Space,
		action_space: spaces.Box,
		net_arch: List[int],
		features_extractor: BaseFeaturesExtractor,
		features_dim: int,
		activation_fn: Type[nn.Module] = nn.ReLU,
		normalize_images: bool = True,
		n_critics: int = 2,
		share_features_extractor: bool = True,
	):
		super().__init__(
			observation_space,
			action_space,
			features_extractor=features_extractor,
			normalize_images=normalize_images,
		)

		self.train_feature = False
		#print("Use the critic to train the feature extractor: ", self.train_feature)

		action_dim = get_action_dim(self.action_space)

		self.share_features_extractor = share_features_extractor
		self.n_critics = n_critics
		self.q_networks = []
		self.tmp_config = Hyperparameters()
		self.num_qVals = self.tmp_config.num_rewards + 1
		for idx in range(n_critics):
			q_net = create_mlp(features_dim + action_dim, self.num_qVals, net_arch, activation_fn)
			q_net = nn.Sequential(*q_net)
			self.add_module(f"qf{idx}", q_net)
			self.q_networks.append(q_net)

	def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
		# Learn the features extractor using the policy loss only
		# when the features_extractor is shared with the actor
		with th.set_grad_enabled(self.train_feature):
			features = self.extract_features(obs, self.features_extractor)
		
		qvalue_input = th.cat([features, actions], dim=1)
		return tuple(q_net(qvalue_input) for q_net in self.q_networks)

	def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
		"""
		Only predict the Q-value using the first network.
		This allows to reduce computation when all the estimates are not needed
		(e.g. when updating the policy in TD3).
		"""
		with th.no_grad():
			features = self.extract_features(obs, self.features_extractor)
		return self.q_networks[0](th.cat([features, actions], dim=1))