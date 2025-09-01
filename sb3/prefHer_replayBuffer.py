import psutil
import warnings
from typing import Any, Dict, List, Optional, Union

import random
import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
	DictReplayBufferSamples,
)

from stable_baselines3.common.buffers import ReplayBuffer
from config import Hyperparameters

class PDMORL_DictReplayBuffer(ReplayBuffer):
	"""
	Stable-Baselines-based TD3 replay buffer extended to include preferences.
	Based on the PDMORL-HER replay buffer.
	The idea is to sample `config.additionalPrefs` for each transition 
	to enhance preference space exploration during training.

	Dict Replay buffer used in off-policy algorithms like SAC/TD3.
	Extends the ReplayBuffer to use dictionary observations
	
	:param buffer_size: Max number of element in the buffer
	:param observation_space: Observation space
	:param action_space: Action space
	:param device: PyTorch device
	:param n_envs: Number of parallel environments
	:param optimize_memory_usage: Enable a memory efficient variant
		Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
	:param handle_timeout_termination: Handle timeout termination (due to timelimit)
		separately and treat the task as infinite horizon task.
		https://github.com/DLR-RM/stable-baselines3/issues/284
	"""

	def __init__(
		self,
		buffer_size: int,
		observation_space: spaces.Space,
		action_space: spaces.Space,
		device: Union[th.device, str] = "auto",
		n_envs: int = 1,
		optimize_memory_usage: bool = False,
		handle_timeout_termination: bool = True,
	):
		self.config = Hyperparameters()
		self.reward_cnt = self.config.num_rewards + 1
		#print("HER_Buffer")
		super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

		assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
		self.buffer_size = max(buffer_size // n_envs, 1)
		if(self.config.evaluate):
		   self.buffer_size = 10000

		# Check that the replay buffer can fit into the memory
		if psutil is not None:
			mem_available = psutil.virtual_memory().available

		assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
		# disabling as this adds quite a bit of complexity
		# https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
		self.optimize_memory_usage = optimize_memory_usage

		self.observations = {
			key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
			for key, _obs_shape in self.obs_shape.items()
		}
		self.next_observations = {
			key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
			for key, _obs_shape in self.obs_shape.items()
		}

		self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
		self.rewards = np.zeros((self.buffer_size, self.n_envs, self.reward_cnt), dtype=np.float32)
		self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
		self.pref_weights_round = np.zeros(self.config.num_rewards, dtype=np.float32)

		# Handle timeouts termination properly if needed
		# see https://github.com/DLR-RM/stable-baselines3/issues/284
		self.handle_timeout_termination = handle_timeout_termination
		self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

		if psutil is not None:
			obs_nbytes = 0
			for _, obs in self.observations.items():
				obs_nbytes += obs.nbytes

			total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
			if self.next_observations is not None:
				next_obs_nbytes = 0
				for _, obs in self.observations.items():
					next_obs_nbytes += obs.nbytes
				total_memory_usage += next_obs_nbytes

			gb_size = total_memory_usage/1e9
			print("\nBuffer estimated size:", gb_size ,"GB", "with elements:", self.buffer_size, "\n")

			if total_memory_usage > mem_available:
				# Convert to GB
				total_memory_usage /= 1e9
				mem_available /= 1e9
				warnings.warn(
					"This system does not have apparently enough memory to store the complete "
					f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
				)

	def load_config(self, config):
		self.config = config

	def add(
		self,
		obs: Dict[str, np.ndarray],
		next_obs: Dict[str, np.ndarray],
		action: np.ndarray,
		reward: np.ndarray,
		done: np.ndarray,
		infos: List[Dict[str, Any]],
	) -> None:  # pytype: disable=signature-mismatch
		# Copy to avoid modification by reference
		for key in self.observations.keys():
			# Reshape needed when using multiple envs with discrete observations
			# as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
			if isinstance(self.observation_space.spaces[key], spaces.Discrete):
				obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
			self.observations[key][self.pos] = np.array(obs[key]).copy()

		for key in self.next_observations.keys():
			if isinstance(self.observation_space.spaces[key], spaces.Discrete):
				next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
			self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

		# Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
		action = action.reshape((self.n_envs, self.action_dim))

		#HER - sample additionalPrefs weights when not in key phase
		for pref_ in range(self.config.additionalPrefs+1):
			if not pref_ == 0:
				# sample additional prefs for real transition to implement pdmorl HER 
				# (prevent bias of different scaled q_values)
				self.get_pref_weights() #to get new self.pref_weights_round

				for key in self.observations.keys():
					if key == 'pref_weights':
						self.observations[key][self.pos] = self.get_pref_weights_step().copy()
					else:
						self.observations[key][self.pos] = np.array(obs[key])#.copy() might not be needed as not modified

				for key in self.next_observations.keys():
					if key == 'pref_weights':
						self.next_observations[key][self.pos] = self.get_pref_weights_step().copy()
					else:
						self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

			self.actions[self.pos] = np.array(action).copy()
			self.rewards[self.pos] = np.array(reward).copy()
			self.dones[self.pos] = np.array(done).copy()

			if self.handle_timeout_termination:
				self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
		
			self.pos += 1
			if self.pos == self.buffer_size:
				self.full = True
				self.pos = 0

	def sample(
		self,
		batch_size: int,
		env: Optional[VecNormalize] = None,
	) -> DictReplayBufferSamples:  # type: ignore[signature-mismatch] #FIXME:
		"""
		Sample elements from the replay buffer.

		:param batch_size: Number of element to sample
		:param env: associated gym VecEnv
			to normalize the observations/rewards when sampling
		:return:
		"""
		return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

	def _get_samples(
		self,
		batch_inds: np.ndarray,
		env: Optional[VecNormalize] = None,
	) -> DictReplayBufferSamples:  # type: ignore[signature-mismatch] #FIXME:
		# Sample randomly the env idx
		env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

		# Normalize if needed and remove extra dimension (we are using only one env for now)
		obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
		next_obs_ = self._normalize_obs(
			{key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
		)

		# Convert to torch tensor
		observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
		next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

		return DictReplayBufferSamples(
			observations=observations,
			actions=self.to_torch(self.actions[batch_inds, env_indices]),
			next_observations=next_observations,
			# Only use dones that are not due to timeouts
			# deactivated by default (timeouts is initialized as an array of False)
			dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
				-1, 1
			),
			rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices, :], env)), #.reshape(-1, 1)
		)
	
	def to(self, device):
		# Verschiebe alle Tensoren im Replay Buffer auf das spezifizierte Gerät
		self.observations = {k: v.to(device) for k, v in self.observations.items()}
		self.next_observations = {k: v.to(device) for k, v in self.next_observations.items()}
		self.actions = self.actions.to(device)
		self.rewards = self.rewards.to(device)
		self.dones = self.dones.to(device)

	def get_pref_weights(self):
		"""
		Theses functions are kind of doubled in carla env...
	
		Generate a base set of preference weights for multiple objectives.
		
		- Weights correspond to objectives such as Aggressiveness, Comfort, Speed, 
		Efficiency, Energy, Weather, Safety, etc.
		- Randomly generated weights are squared to emphasize larger values.
		- Occasionally (1% chance), a single weight is set to 1 and others to 0 
		to simulate extreme preference scenarios.
		- Normalized so the sum equals 1, with a small epsilon to avoid division by zero.
		"""
		# Generate random weights as float32
		weights = np.random.uniform(low=0.0, high=1.0, size=(self.config.num_rewards,)).astype(np.float32)
		weights = np.square(weights)  # Emphasize larger weights
		
		# Rarely choose extreme single-objective preference
		if random.random() < 0.01:
			weights = np.zeros_like(weights)
			random_index = random.randrange(self.config.num_rewards)
			weights[random_index] = 1.0

		# Normalize weights with epsilon to prevent division by zero
		sum_weights = np.sum(weights).astype(np.float32)
		epsilon = 1e-8
		if sum_weights < epsilon:
			normalized_weights = np.ones_like(weights) / len(weights)
		else:
			normalized_weights = weights / sum_weights

		# Store normalized weights for incremental steps
		self.pref_weights_round = normalized_weights


	def get_pref_weights_step(self):
		"""
		Theses functions are kind of doubled in carla env...
	
		Generate a small-step variant of the current preference weights.
		
		- Adds Gaussian noise to the base preference weights to explore nearby preferences.
		- Clips the resulting weights to [0, 1].
		- Normalizes so the sum equals 1.
		- Returns weights rounded to 3 decimal places.
		"""
		# Add small Gaussian noise to previous preference weights
		local_w = np.random.normal(loc=0, scale=0.05, size=(self.config.num_rewards,)).astype(np.float32)
		step_weights = self.pref_weights_round + local_w

		# Round and clip weights
		step_weights = np.round(step_weights, 4)
		step_weights = np.clip(step_weights, 0, 1).astype(np.float32)

		# Normalize to sum to 1
		normalized_step_weights = step_weights / np.sum(step_weights).astype(np.float32)

		return np.round(normalized_step_weights, 3)


	def clear_buffer(self):
		self.pos = 0
		self.full = False

		self.observations = {
			key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=self.observation_space[key].dtype)
			for key, _obs_shape in self.obs_shape.items()
		}
		self.next_observations = {
			key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=self.observation_space[key].dtype)
			for key, _obs_shape in self.obs_shape.items()
		}

		self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
		self.rewards = np.zeros((self.buffer_size, self.n_envs, self.reward_cnt), dtype=np.float32)
		self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
		self.pref_weights_round = np.zeros(self.config.num_rewards, dtype=np.float32)
		print("Buffer cleared")