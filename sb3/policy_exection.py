import torch as th
import numpy as np

from gymnasium import spaces
from torch import nn
from typing import Any, Dict, List, Optional, Type, Union, Tuple

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.utils import is_vectorized_observation

from select_extractor import CustomCombinedExtractor
from custom_critic import CustomContinuousCritic
from custom_actor import CustomActor

from stable_baselines3.common.distributions import (
	Distribution
)

"""
Based on StableBaselines3 td3 policy concept!
"""

class CustomMultiInputPolicy(TD3Policy):
	"""
	Policy class (with both actor and critic) for TD3 to be used with Dict observation spaces.

	:param observation_space: Observation space
	:param action_space: Action space
	:param lr_schedule: Learning rate schedule (could be constant)
	:param net_arch: The specification of the policy and value networks.
	:param activation_fn: Activation function
	:param features_extractor_class: Features extractor to use.
	:param features_extractor_kwargs: Keyword arguments
		to pass to the features extractor.
	:param normalize_images: Whether to normalize images or not,
		 dividing by 255.0 (True by default)
	:param optimizer_class: The optimizer to use,
		``th.optim.Adam`` by default
	:param optimizer_kwargs: Additional keyword arguments,
		excluding the learning rate, to pass to the optimizer
	:param n_critics: Number of critic networks to create.
	:param share_features_extractor: Whether to share or not the features extractor
		between the actor and the critic (this saves computation time)
	"""

	def __init__(
		self,
		observation_space: spaces.Dict,
		action_space: spaces.Box,
		lr_schedule: Schedule,
		net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
		activation_fn: Type[nn.Module] = nn.ReLU,
		features_extractor_class: Type[BaseFeaturesExtractor] = CustomCombinedExtractor,
		features_extractor_kwargs: Optional[Dict[str, Any]] = None,
		normalize_images: bool = True,
		optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
		optimizer_kwargs: Optional[Dict[str, Any]] = None,
		n_critics: int = 2,
		share_features_extractor: bool = False,
	):
	
		super().__init__(
			observation_space,
			action_space,
			lr_schedule,
			net_arch,
			activation_fn,
			features_extractor_class,
			features_extractor_kwargs,
			normalize_images,
			optimizer_class,
			optimizer_kwargs,
			n_critics,
			share_features_extractor
		)
		critic: CustomContinuousCritic
		critic_target: CustomContinuousCritic

		print("observation_space\n", observation_space)

	def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
		critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
		return CustomContinuousCritic(**critic_kwargs).to(self.device)

	def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
		actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
		return CustomActor(**actor_kwargs).to(self.device)

	def my_predict(
		self,
		observation: Union[np.ndarray, Dict[str, np.ndarray]],
		state: Optional[Tuple[np.ndarray, ...]] = None,
		episode_start: Optional[np.ndarray] = None,
		hidden_activation: bool = False,
	) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
		"""
		modified to get the action and also the activation of the x actor layer
		"""
		# Switch to eval mode (this affects batch norm / dropout)
		self.set_training_mode(False)
		
		observation, vectorized_env = self.obs_to_tensor(observation) #here struggle

		with th.no_grad():
			actions, hidden_activation = self.actor(observation, get_hidden_activations=True)

		# Convert to numpy, and reshape to the original action shape
		actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))
		hidden_activation = hidden_activation.cpu().numpy()

		if isinstance(self.action_space, spaces.Box):
			if self.squash_output:
				# Rescale to proper domain when using squashing
				actions = self.unscale_action(actions)
			else:
				# Actions could be on arbitrary scale, so clip the actions to avoid
				# out of bound error (e.g. if sampling from a Gaussian distribution)
				actions = np.clip(actions, self.action_space.low, self.action_space.high)

		# Remove batch dimension if needed
		if not vectorized_env:
			actions = actions.squeeze(axis=0)

		return actions, hidden_activation
	
	# adjusted to ignore lr_scedual fnc need
	def _build(self, lr_scedual) -> None:
		# Create actor and target
		self.actor = self.make_actor(features_extractor=None)
		self.actor_target = self.make_actor(features_extractor=None)
		# Initialize the target to have the same weights as the actor
		self.actor_target.load_state_dict(self.actor.state_dict())

		self.actor.optimizer = self.optimizer_class(
			self.actor.parameters(),
			# normally sets lr-scedual here
			**self.optimizer_kwargs,
		)

		print("share_features_extractor: ", self.share_features_extractor)
		if self.share_features_extractor:
			self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
			self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
		else:
			self.critic = self.make_critic(features_extractor=None)
			self.critic_target = self.make_critic(features_extractor=None)

		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic.optimizer = self.optimizer_class(
			self.critic.parameters(),
			**self.optimizer_kwargs,
		)

		# Target networks should always be in eval mode
		self.actor_target.set_training_mode(False)
		self.critic_target.set_training_mode(False)

	def get_distribution(self, obs: th.Tensor) -> Distribution:
		"""
		Get the current policy distribution given the observations.

		:param obs:
		:return: the action distribution.
		"""
		features = super().extract_features(obs, self.pi_features_extractor)
		latent_pi = self.mlp_extractor.forward_actor(features)
		return self._get_action_dist_from_latent(latent_pi)