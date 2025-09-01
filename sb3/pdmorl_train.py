import numpy as np
import torch as th
from torch.nn import functional as F
from torchvision import transforms

from stable_baselines3 import TD3
from stable_baselines3.common.off_policy_algorithm import TrainFreq, RolloutReturn
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise, NormalActionNoise
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps, get_parameters_by_name, polyak_update

from scipy.interpolate import RBFInterpolator
from typing import Optional
from config import Hyperparameters

import wandb
import atexit
import time
import random
import itertools

"""
This class extends the TD3 algorithm from Stable Baselines3 to implement
the PD-MORL (Preference-based Multi-Objective Reinforcement Learning) framework.
The algorithm leverages weight interpolation for multi-objective optimization.
Detailed behavior is defined in the train function.
"""
def finish_wandb():
	"""Finish the active wandb run cleanly."""
	wandb.finish()

class PDMORL_TD3(TD3):
	def __init__(self, *args, **kwargs):
		super(PDMORL_TD3, self).__init__(*args, **kwargs)

		# Load hyperparameters from configuration
		self.config = Hyperparameters()

		# Generate a batch of weight vectors for multi-objective optimization
		self.weight_batch = self.generate_weights_batch(
			self.config.w_step_size, self.config.num_rewards
		)
		self.th_weight_batch = th.tensor(
			self.weight_batch, dtype=th.float32, device=self.device
		)

		# Initialize interpolator values for tracking best objectives per weight
		self.bestObjectiveInterpotorValues = np.ones(
			(len(self.weight_batch), self.config.num_rewards), dtype=np.float32
		)
		# Store previous best objective values (used in first iteration)
		self.old_bestObjV = np.ones(
			(len(self.weight_batch), self.config.num_rewards), dtype=np.float32
		)

		# Initialize steering mean
		self.steering_mean = 0.0

		# Logging and tracking arrays for episodes and performance metrics
		self.log_interval = 2
		self.ep_rew = np.zeros(
			(self.log_interval, self.config.num_rewards + 1), dtype=np.float32
		)
		self.episode_median_rwd = [[] for _ in range(self.log_interval)]
		self.points_reached_cnt = np.zeros((self.log_interval), dtype=np.float32)
		self.detection_car_cnt = np.zeros((self.log_interval), dtype=np.float32)
		self.collision_cnt = np.zeros((self.log_interval), dtype=np.float32)
		self.collision_cnt_car = np.zeros((self.log_interval), dtype=np.float32)
		self.collision_cnt_env = np.zeros((self.log_interval), dtype=np.float32)
		self.lane_invasion_cnt = np.zeros((self.log_interval), dtype=np.float32)
		self.speeding_cnt = np.zeros((self.log_interval), dtype=np.float32)
		self.dist_to_center = np.zeros((self.log_interval), dtype=np.float32)
		self.timeout = np.zeros((self.log_interval), dtype=np.float32)
		self.driving_score = np.zeros((self.log_interval), dtype=np.float32)
		self.route_completion = np.zeros((self.log_interval), dtype=np.float32)
		self.episode_duration = np.zeros((self.log_interval), dtype=np.float32)
		self.base_th = th.tensor(self.config.num_rewards, dtype=th.float32, device=self.device)
		self.expo_th = th.tensor(1.0/4.5, dtype=th.float32, device=self.device)
		self.ep_steps = np.zeros((self.log_interval), dtype=np.float32)

		# Sliding window for logging driving score
		self.log_interval_window = 6
		self.driving_score_window = np.zeros((self.log_interval_window), dtype=np.float32)

		# Initialize example interpolator values if no prior key step data exists
		if self.config.key_steps == 0:
			self.bestObjectiveInterpotorValues = np.array(
				[
					[513.5, 403.5, 203.5, 1105.4],
					[588.6, 425.7, 244.6, 1049.9],
					[575.7, 488.9, 179.7, 1042.6],
					[613.7, 413.0, 198.3, 1050.3],
					[550.7, 398.6, 200.1, 1089.8],
				],
				dtype=np.float32
			)

		# Path for saving/loading model or results
		self.my_path = None

		# Flags for initialization and training phases
		self.inizialize = True
		self.secondPhase = False

		# Placeholder for interpolators per weight vector
		self.interpolator = []

		# Store temporary additional preferences for key training phase
		self.tmp_additionalPrefs = self.config.additionalPrefs
		self.config.additionalPrefs = 0  # reset in config

		# Learning rates and regularization
		self.actor_lr = self.config.lr_actor
		self.critic_lr = self.config.lr_critic
		self.weight_decay = self.config.weight_decay

		# Update learning rate if not in evaluation mode
		if not self.config.evaluate:
			self.update_lr()

	def update_interpolator(self):
		"""
		Updates the RBF (Radial Basis Function) interpolator used for approximating
		the best objective values across different preference weights.

		This allows smooth estimation of multi-objective rewards for arbitrary
		preference vectors based on previously observed best objectives.

		Steps:
		1. Prints the current best objective values for logging and debugging.
		2. Normalizes the objective values using L2 norm per weight vector.
		3. Constructs a new RBFInterpolator with the preference weights and normalized values.
		"""
		pref_weights = self.weight_batch
		values = self.bestObjectiveInterpotorValues

		# Format the output for logging: 1 decimal place, no scientific notation
		formatted_arr = np.array2string(
			np.round(values, 1), 
			formatter={'float_kind': lambda x: f"{x:.1f}"}
		)
		print("current_interpolator_values:\n", formatted_arr, "\nat episode ", self._episode_num)

		# Normalize values with L2 norm across each weight vector
		values_normed = values / np.linalg.norm(values, ord=2, axis=1, keepdims=True)

		# min_vals = np.min(values, axis=1, keepdims=True)
		# max_vals = np.max(values, axis=1, keepdims=True)
		# values_normed = (values - min_vals) / ((max_vals - min_vals) + 1e-8) 

		# Create RBF interpolator with linear kernel and no smoothing
		self.interpolator = RBFInterpolator(pref_weights, values_normed, smoothing=0, kernel='linear')

	def train(self, gradient_steps: int, batch_size: int = 100) -> None:
		"""
		Performs the PD-MORL training loop over the specified number of gradient steps.

		Training is split into two phases:

		1. Interpolator Training Phase:
		- The RBF interpolator is trained using key preference weights (e.g., (0,0,1,0)).
		- Networks and replay buffers may be reset, while the interpolator retains previous values.
		- Usually done only once or when the reward function changes.
		
		2. Real Training Phase:
		- Full training with all preference weights.
		- Critic networks are trained first for stability; feature extractor training may be delayed.
		- Learns the final policy.

		Modifications for autonomous driving:
		- Adds an extra non-preference-weighted dimension `[0]` to rewards and preferences.
		- Ensures core driving capability and is excluded from PD-MORL interpolation.
		"""

		# Enable train mode for policy networks (affects batch norm/dropout)
		self.policy.set_training_mode(True)

		# Update the interpolator if values have changed or during initialization
		update = not np.array_equal(self.old_bestObjV, self.bestObjectiveInterpotorValues)
		if update or self.inizialize:
			self.inizialize = False
			self.update_interpolator()  # Update RBF interpolator with current best objectives
			self.old_bestObjV = self.bestObjectiveInterpotorValues.copy()
			self.log_metrics({
				"running_max": np.sum(self.bestObjectiveInterpotorValues),
				"_n_updates": self._n_updates
			})

		# Initialize lists to store losses for logging
		actor_losses, critic_losses = [], []
		l1_losses, angle_losses = [], []
		actor_losses_l1, actor_angle_losses = [], []

		# -----------------------------
		# Main gradient step loop
		# -----------------------------
		for _ in range(gradient_steps):
			self._n_updates += 1

			# -----------------------------
			# Transition from key-step training to full objective-space training
			# -----------------------------
			if self._n_updates > self.config.key_steps and not self.secondPhase:
				self.secondPhase = True
				print("Start training with the second phase (full objective space)")

				# Optionally save best interpolator values
				# print("bestObjectiveInterpotorValues", self.bestObjectiveInterpotorValues)
				# self.save_best_bestObjectiveInterpotorValues(self.my_path)

				# Update additional preferences for the second phase
				self.config.additionalPrefs = self.tmp_additionalPrefs

				# Optionally reinitialize random seed for reproducibility
				# self.set_random_seed(self.seed)

				# Setup model and create network aliases
				self._setup_model()
				time.sleep(5)
				self._create_aliases()
				self.update_lr()  # Ensure learning rate schedule is applied

				print("Networks have been reset for phase 2")

				# Disable feature extractor training initially
				self.actor.train_feature = False
				self.critic.train_feature = False

				# Clear replay buffer for the new phase
				self.replay_buffer.clear_buffer()

				# Optional delay for safety before continuing training
				time.sleep(5)

				# Optionally, adjust training hyperparameters for phase 2
				# self.config.BATCH_SIZE += 64
				# self.config.policy_freq += 2
				# self.policy_delay = self.config.policy_delay
				# self.target_noise_clip = self.config.target_noise_clip
				# self.target_policy_noise = self.config.target_policy_noise
				# self.config.NUM_TRAFFIC_VEHICLES = 10

				break

			# -----------------------------
			# Sample replay buffer
			# -----------------------------
			replay_data = self.replay_buffer.sample(self.config.BATCH_SIZE, env=self._vec_normalize_env)

			# If neither actor nor critic is training the feature extractor, add random noise to input
			# This increases observation variance to avoid local minima until feature extractor is trained
			random_tensor = th.randint(
				low=0, high=256, 
				size=(self.config.BATCH_SIZE, self.config.NUM_CHANNELS, self.config.state_frames, self.config.IM_HEIGHT, self.config.IM_WIDTH),
				dtype=th.uint8, device=self.device
			)
			if not (self.actor.train_feature or self.critic.train_feature):
				replay_data.observations["camera"] = random_tensor
				replay_data.next_observations["camera"] = random_tensor

			with th.no_grad():
				# Generate key preference weights
				if self._n_updates < self.config.key_steps:
					# Randomly sample batch indices from predefined weight batch
					batch_indices = th.randint(0, self.th_weight_batch.size(0), (self.config.BATCH_SIZE,))
					key_weights = self.th_weight_batch[batch_indices]

					# Add small Gaussian noise to preferences during early training phase
					pref_noise = th.normal(mean=0.0, std=0.05, size=key_weights.shape, device=self.device).clamp(-0.05, 0.05)
					key_weights = key_weights + pref_noise
				else:
					# Use stored preference weights from replay buffer for full training
					key_weights = replay_data.observations["pref_weights"]

				# Clamp weights between 0 and 1, round to 3 decimals
				key_weights.clamp(0, 1)
				th.round(key_weights, decimals=3)

				# Apply the same preference weights to current and next observations
				replay_data.observations["pref_weights"] = key_weights
				replay_data.next_observations["pref_weights"] = key_weights

				# Core driving weight (first reward dimension) is fixed at 1, rest are preferences
				weights = th.ones((self.config.BATCH_SIZE, self.config.num_rewards + 1), device=self.device) * th.pow(self.base_th, self.expo_th)
				weights[:, 1:] = replay_data.observations["pref_weights"]

				# Compute next actions with target actor and clipped noise
				noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
				noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
				next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

				# Compute target Q-values from both critics
				critic_out1, critic_out2 = self.critic_target(replay_data.next_observations, next_actions)

				# Weighted Q-values to select minimum for double-Q
				critic_out1_weighted = th.sum(critic_out1 * weights, dim=1)
				critic_out2_weighted = th.sum(critic_out2 * weights, dim=1)
				min_index = th.argmin(th.stack((critic_out1_weighted, critic_out2_weighted), dim=1), dim=1)

				# Choose Q-values based on min index
				next_q_values = th.where(min_index.unsqueeze(-1) == 0, critic_out1, critic_out2)

				# Target Q: reward plus discounted next Q-values
				target_Q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

			# -----------------------------
			# Current Q-values and angle term
			# -----------------------------
			current_Q1, current_Q2 = self.critic(replay_data.observations, replay_data.actions)

			if self._n_updates < (2 * self.config.START_TIMESTEPS + self.config.delay_actor_timesteps + self.config.key_steps):
				# Early training: do not compute angle term
				with th.no_grad():
					angle_term_1 = th.zeros(1, device=self.device)
					angle_term_2 = th.zeros(1, device=self.device)
			else:
				# Compute cosine similarity between interpolated preferences and predicted Q-values
				with th.no_grad():
					np_weights = key_weights.cpu().numpy()
					weight_p = th.from_numpy(self.interpolator(np_weights).astype(np.float32)).to(self.device)

					tmp_1 = th.clamp(F.cosine_similarity(weight_p, current_Q1[:, 1:]), 0, 0.9999)
					tmp_2 = th.clamp(F.cosine_similarity(weight_p, current_Q2[:, 1:]), 0, 0.9999)

				angle_term_1 = th.rad2deg(th.acos(tmp_1))
				angle_term_2 = th.rad2deg(th.acos(tmp_2))

			# -----------------------------
			# Critic losses
			# -----------------------------
			l1_loss = (
				F.smooth_l1_loss(current_Q1, target_Q, reduction='none').mean(dim=0).mean() +
				F.smooth_l1_loss(current_Q2, target_Q, reduction='none').mean(dim=0).mean()
			)
			angle_loss = angle_term_1.mean() + angle_term_2.mean()
			critic_loss = l1_loss + self.config.zeta * angle_loss
			assert isinstance(critic_loss, th.Tensor)

			# -----------------------------
			# Optimize critics
			# -----------------------------
			if (self._n_updates == self.config.START_TIMESTEPS and self._n_updates < self.config.key_steps) or \
			(self._n_updates == (self.config.START_TIMESTEPS + self.config.key_steps)):
				print("Start training the Critic")
				angle_losses = []

			if ((self._n_updates > self.config.START_TIMESTEPS and self._n_updates < self.config.key_steps) or
				(self._n_updates > (self.config.START_TIMESTEPS + self.config.key_steps))):
				angle_losses.append(angle_loss.item())
				l1_losses.append(l1_loss.item())
				critic_losses.append(critic_loss.item())

				self.critic.optimizer.zero_grad()
				critic_loss.backward()
				th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2)
				self.critic.optimizer.step()
			else:
				# Reset losses outside critic training phase
				angle_losses = []
				l1_losses = []
				critic_losses = []
				angle_loss = th.zeros(1, device=self.device)
				critic_loss = th.zeros(1, device=self.device)

			# -----------------------------
			# Predicted actions for actor
			# -----------------------------
			pred_actions = self.actor(replay_data.observations)

			# Log action statistics periodically
			if self._n_updates % self.config.policy_freq == 0:
				with th.no_grad():
					mean_action = th.mean(pred_actions, dim=0).detach().cpu().numpy()
					std_action = th.std(pred_actions, dim=0).detach().cpu().numpy()
					if self._n_updates % 500 == 0:
						self.log_metrics({
							"steering_mean": mean_action[0],
							"throttle_mean": mean_action[1],
							"steering_std": std_action[0],
							"throttle_std": std_action[1],
							"_n_updates": self._n_updates
						})
	
				# -----------------------------
				# Compute actor loss and optimize
				# -----------------------------

				# Compute predicted Q-values from critic for current actions
				q_values_from_critic1 = self.critic.q1_forward(replay_data.observations, pred_actions)
				weighted_q = q_values_from_critic1 * weights

				# Compute L1 actor loss: negative mean over weighted Q-values
				actor_loss_l1 = -weighted_q.mean(dim=0).mean()
				# Alternative: mean(dim=0).sum()

				# Compute angle loss term (preference alignment)
				if self._n_updates < (3 * self.config.START_TIMESTEPS + self.config.delay_actor_timesteps + self.config.key_steps):
					with th.no_grad():
						angle_term = th.zeros(1, device=self.device)
				else:
					# Cosine similarity between interpolated preferences and predicted Q-values
					angle_term = th.rad2deg(
						th.acos(
							th.clamp(F.cosine_similarity(weight_p, q_values_from_critic1[:, 1:], dim=1), 0, 0.9999)
						)
					)

				angle_loss_actor = angle_term.mean()
				actor_loss = actor_loss_l1 + self.config.psi * angle_loss_actor
				actor_angle_losses.append(angle_loss_actor.item())

				# Start actor training message for debugging
				if (self._n_updates == self.config.delay_actor_timesteps and self._n_updates < self.config.key_steps) or \
				(self._n_updates == (self.config.delay_actor_timesteps + self.config.key_steps)):
					print("Start training the actor: ", self._n_updates)

				# Full actor optimization during the appropriate training phase
				if ((self._n_updates > self.config.delay_actor_timesteps and self._n_updates < self.config.key_steps) or 
					(self._n_updates > (self.config.delay_actor_timesteps + self.config.key_steps))):

					actor_losses.append(actor_loss.item())
					actor_losses_l1.append(actor_loss_l1.item())

					# Backpropagate actor loss
					self.actor.optimizer.zero_grad()
					actor_loss.backward()
					th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2)
					self.actor.optimizer.step()

					# Polyak averaging / target network update
					polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
					polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

				else:
					# Reset losses outside actor training phase
					actor_loss = th.zeros(1, device=self.device)
					angle_loss_actor = th.zeros(1, device=self.device)
					actor_loss_l1 = th.zeros(1, device=self.device)
					actor_losses = []
					actor_losses_l1 = []
					actor_angle_losses = []

				# Update critic target networks via Polyak averaging
				polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
				polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)

				# Feature extractor training at specific update
				if (self._n_updates == 1.2 * self.config.delay_actor_timesteps and self._n_updates < self.config.key_steps) or \
				(self._n_updates == ((1.2 * self.config.delay_actor_timesteps) + self.config.key_steps)):
					print("Start training the feature extractor with the actor loss")
					self.actor.train_feature = True
					self.critic.train_feature = False

				# Optional: full feature extractor training (commented)
				# if (self._n_updates == 8 * self.config.delay_actor_timesteps and self._n_updates < self.config.key_steps) or \
				#    self._n_updates == 8 * self.config.delay_actor_timesteps + self.config.key_steps:
				#     print("Start training the whole feature extractor")
				#     self.actor.features_extractor.extractors['camera'].train_all_layer = True

				# -----------------------------
				# Logging
				# -----------------------------
				critic_loss_log = th.mean(th.tensor(critic_losses)).detach().cpu().numpy()
				angle_loss_critic_log = th.mean(th.tensor(angle_losses)).detach().cpu().numpy()
				angle_loss_actor_log = th.mean(th.tensor(actor_angle_losses)).detach().cpu().numpy()
				l1_loss_log = th.mean(th.tensor(l1_losses)).detach().cpu().numpy()

				self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

				if len(actor_losses) > 0:
					actor_loss_log = th.mean(th.tensor(actor_losses)).detach().cpu().numpy()
					actor_loss_unscaled_log = th.mean(th.tensor(actor_losses_l1)).detach().cpu().numpy()
					self.logger.record("train/actor_loss", actor_loss_log)
					self.log_metrics({
						"actor_loss": actor_loss_log,
						"actor_losses_raw": actor_loss_unscaled_log,
						"angle_loss_actor": angle_loss_actor_log,
						"_n_updates": self._n_updates
					})

				self.log_metrics({
					"critic_loss": critic_loss_log,
					"l1_loss": l1_loss_log,
					"angle_loss_critic": angle_loss_critic_log,
					"_n_updates": self._n_updates
				})

				# Compute performance metric and check for best model saving
				if angle_loss_critic_log > 0.0 and self._n_updates > (self.config.key_steps + self.config.delay_actor_timesteps):
					angle_term = np.power((angle_loss_critic_log + 1), 0.98)  # Weight scaling for angle term
					performance = np.round(8 * self.config.currentScore / angle_term, 3)
					if performance > 0 and performance > (self.config.bestScore + 0.01):
						self.config.bestScore = performance
						self.config.bestModelSave = True
						print("Performance", performance, 
							", PrefsLoss: ", angle_loss_critic_log, 
							", DrivingScore: ", self.config.currentScore, 
							", Episode: ", self._episode_num, 
							", Updates: ", self._n_updates)
				else:
					self.config.bestModelSave = False
		
	def collect_rollouts(
		self,
		env: VecEnv,
		callback: BaseCallback,
		train_freq: TrainFreq,
		replay_buffer: ReplayBuffer,
		action_noise: Optional[ActionNoise] = None,
		learning_starts: int = 0,
		log_interval: Optional[int] = None,
	) -> RolloutReturn:
		"""
		This function is based on Stable Baselines!! and has been extended to:

		- Adjust the initial exploration noise.
		- Include additional logging with Weights & Biases (wandb).
				Collect experiences and store them into a ``ReplayBuffer``.

		:param env: The training environment
		:param callback: Callback that will be called at each step
			(and at the beginning and end of the rollout)
		:param train_freq: How much experience to collect
			by doing rollouts of current policy.
			Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
			or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
			with ``<n>`` being an integer greater than 0.
		:param action_noise: Action noise that will be used for exploration
			Required for deterministic policy (e.g. TD3). This can also be used
			in addition to the stochastic policy for SAC.
		:param learning_starts: Number of steps before learning for the warm-up phase.
		:param replay_buffer:
		:param log_interval: Log data every ``log_interval`` episodes
		:return:
		"""
		# Switch to eval mode (this affects batch norm / dropout)
		self.policy.set_training_mode(False)

		num_collected_steps, num_collected_episodes = 0, 0

		assert isinstance(env, VecEnv), "You must pass a VecEnv"
		assert train_freq.frequency > 0, "Should at least collect one step or episode."

		if env.num_envs > 1:
			assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

		# -------------------------------------------------------------------------
		# Determine exploration / action noise based on training phase and episode
		# -------------------------------------------------------------------------

		# Full training phase: apply standard action noise set in config
		if ((self._n_updates > self.config.delay_actor_timesteps and self._n_updates < self.config.key_steps) or 
			(self._n_updates > (self.config.delay_actor_timesteps + self.config.key_steps))):
			
			# Use default normal action noise for exploration
			action_noise = NormalActionNoise(
				mean=np.array([0, 0]), 
				sigma=np.array([self.config.action_noise, self.config.action_noise]), 
				dtype=np.float32
			)
			self.config.useAutopilot = False

		else:
			# Special exploration noise for early training phases
			self.config.useAutopilot = False

			# Periodically use Carla autopilot for environment exploration (disabled by default)
			if self._episode_num % 22 == 0 and self._episode_num > 0:
				# Uncomment to enable autopilot exploration
				# self.config.useAutopilot = True
				action_noise = NormalActionNoise(mean=np.array([0, 0]), sigma=np.array([0, 0]), dtype=np.float32)

			# Custom exploration noise patterns for initialization
			# Different noise for steering and throttle based on episode number
			elif self._episode_num % 8 == 0:
				action_noise = NormalActionNoise(mean=np.array([0, 0.22]), sigma=np.array([0.001, 0.001]), dtype=np.float32)
			elif self._episode_num % 6 == 0:
				action_noise = NormalActionNoise(
					mean=np.array([np.random.uniform(-1.0, 1.0), np.random.uniform(0.0, 1.0)]), 
					sigma=np.array([0.01, 0.01]), 
					dtype=np.float32
				)
			elif self._episode_num % 2 == 0:
				action_noise = NormalActionNoise(mean=np.array([0, 0.2]), sigma=np.array([0.05, 0.05]), dtype=np.float32)
			else:
				# Other cases: exploratory noise around current steering mean
				action_noise = NormalActionNoise(
					mean=np.array([self.steering_mean, np.random.uniform(0.0, 0.23)]), 
					sigma=np.array([0.01, 0.01]), 
					dtype=np.float32
				)
		# -------------------------------------------------------------------------

		if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
			action_noise = VectorizedActionNoise(action_noise, env.num_envs)

		if self.use_sde:
			self.actor.reset_noise(env.num_envs)

		callback.on_rollout_start()
		continue_training = True
		while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
			if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
				# Sample a new noise matrix
				self.actor.reset_noise(env.num_envs)

			# Select action randomly or according to policy
			actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

			# Rescale and perform action
			# if random.random() < 0.005: # random action with 1/x odds
			# 	action = np.array([np.random.uniform(-1.0, 1.0), np.random.uniform(-1, 1)])
			new_obs, rewards, dones, infos = env.step(actions)

			self.ep_rew[self._episode_num % self.log_interval] += rewards[0]
			
			self.num_timesteps += env.num_envs
			num_collected_steps += 1

			# Give access to local variables
			callback.update_locals(locals())
			# Only stop training if return value is False, not when it is None.
			if callback.on_step() is False:
				return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

			# Retrieve reward and episode length if using Monitor wrapper
			self._update_info_buffer(infos, dones)

			for idx, (action, reward, done, info) in enumerate(zip(buffer_actions, rewards, dones, infos)):
				
				# -------------------------------------------------------------------------
				# If the episode is done, log relevant metrics to Weights & Biases (wandb) using the info dictionary

				if done:
					if 'points_reached' in info:
						self.points_reached_cnt[self._episode_num % self.log_interval] = float(info['points_reached'])
					
					if 'detection_car' in info:
						self.detection_car_cnt[self._episode_num % self.log_interval] = float(info['detection_car'])

					if 'collision_cnt' in info:
						self.collision_cnt[self._episode_num % self.log_interval] = float(info['collision_cnt'])

					if 'collision_cnt_car' in info:
						self.collision_cnt_car[self._episode_num % self.log_interval] = float(info['collision_cnt_car'])

					if 'collision_cnt_env' in info:
						self.collision_cnt_env[self._episode_num % self.log_interval] = float(info['collision_cnt_env'])

					if 'lane_invasion' in info:
						self.lane_invasion_cnt[self._episode_num % self.log_interval] = float(info['lane_invasion'])

					if 'speeding_cnt' in info:
						self.speeding_cnt[self._episode_num % self.log_interval] = float(info['speeding_cnt'])
	
					if 'dist_to_center' in info:
						self.dist_to_center[self._episode_num % self.log_interval] = float(info['dist_to_center'])

					if 'route_completion' in info:
						self.route_completion[self._episode_num % self.log_interval] = float(info['route_completion'])
				
					if 'episode_duration' in info:
						self.episode_duration[self._episode_num % self.log_interval] = float(info['episode_duration'])

					if 'timeout' in info:
						self.timeout[self._episode_num % self.log_interval] = float(info['timeout'])
					
					if 'ep_steps' in info:
						self.ep_steps[self._episode_num % self.log_interval] = float(info['ep_steps'])
	
					if 'adjusted_driving_core' in info:
						self.driving_score[self._episode_num % self.log_interval] = float(info['adjusted_driving_core'])
	
					if 'terminal_observation' in info:
						# print("Überschreibe terminal_obs richtig")
						# Extrahiere die terminal_observation für die aktuelle Umgebung
						terminal_obs = {key: info['terminal_observation'][key] for key in new_obs.keys()}
						next_obs = terminal_obs
					else:
						next_obs = {key: obs[idx] for key, obs in new_obs.items()}
				else:
					next_obs = {key: obs[idx] for key, obs in new_obs.items()}

			# -------------------------------------------------------------------------


			# Store data in replay buffer (normalized action and unnormalized observation)
			self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

			self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

			# For DQN, check if the target network should be updated
			# and update the exploration schedule
			# For SAC/TD3, the update is dones as the same time as the gradient update
			# see https://github.com/hill-a/stable-baselines/issues/900
			self._on_step()

			for idx, done in enumerate(dones):
				if done:
					# Update stats
					self.steering_mean = np.clip(np.round(np.random.uniform(low=-1.2, high=1.2),2), -1, 1).astype(np.float32)
		
					num_collected_episodes += 1
					self._episode_num += 1

					if action_noise is not None:
						kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
						action_noise.reset(**kwargs)

					# Log metrics at defined intervals
					if self.log_interval is not None and self._episode_num % self.log_interval == 0:
						self._dump_logs()

						# Compute mean rewards over the last `log_interval` episodes
						mean_rewards = np.round(np.mean(self.ep_rew, axis=0), 3)
						total_sum_reward = np.sum(mean_rewards)

						# Average distance to center over steps (avoid division by zero)
						avg_dist = self.dist_to_center / (self.ep_steps + 1)

						# Compute preference scores excluding core driving reward (index 0)
						pref_score = np.round(np.mean(self.ep_rew[:, 1:], axis=0), 3)
						baseline_rwd = np.round(np.mean(self.ep_rew[:, 0]), 3)
						total_pref_score = np.round(np.sum(pref_score), 3)

						# Update driving score window (rolling)
						self.driving_score_window[1:] = self.driving_score_window[:-1]
						self.driving_score_window[0] = self.driving_score[0]

						mean_driving_score = np.round(np.mean(self.driving_score_window), 4)

						# Debug prints (kept commented)
						# print("driving_score: ", driving_score)
						# print("mean_driving_score ", mean_driving_score)
						# print("pref_score: ", pref_score)

						############
						# Carla driving score (Adjusted to this scenario)
						# Computed as percentage of route completion weighted by infraction penalties
						# Penalties follow CARLA leaderboard guidelines: https://leaderboard.carla.org/#task
						############

						##### Interrupt / infraction events ####
						# Off-road driving does not increase route completion - checked
						# Route deviation (Agent deviates more than 30 meters from the assigned route) - checked, max deviation here ~7m
						# Agent blocked (No action for 180 simulation seconds) - checked via timeout
						# Simulation timeout (No client-server communication in 60s) - checked (rare, else training crashes)
						# Route timeout - checked via max ticks per episode

						#### Penalty considerations ####
						# Running a red light or stop sign can be penalized but would require reward/state inclusion
						# Failure to maintain minimum speed - approximated using speeding counter
						# Failure to yield to emergency vehicle - not implemented (requires traffic lights)

						# Penalty coefficients
						p_collisions_car = 0.6
						p_collisions_env = 0.7
						p_timeout = 0.7
						p_speed = 0.7

						# Initialize penalties and apply multiplicative reductions for infractions
						infraction_penalty = np.ones((self.log_interval), dtype=np.float32)
						infraction_penalty *= np.power(p_collisions_env, self.collision_cnt_env)
						infraction_penalty *= np.power(p_collisions_car, self.collision_cnt_car)
						infraction_penalty *= np.power(p_timeout, self.timeout)
						infraction_penalty *= np.power(p_speed, self.speeding_cnt)

						# Compute final CARLA like driving score
						carla_driving_score = self.route_completion * infraction_penalty
						log_driving_score = np.round(np.mean(self.driving_score), 3)


						# log metrics to wandb that are averaged over the last x log_interval episodes
						self.log_metrics({"adjusted_driving_core": log_driving_score, "baseline_rwd": baseline_rwd, "pref_score_sum": total_pref_score, "carla_driving_score": np.mean(carla_driving_score), "timeout": np.mean(self.timeout),"speeding": np.mean(self.speeding_cnt), "route_completion": np.mean(self.route_completion), "episode_duration": np.mean(self.episode_duration), "avg_dist_to_center": np.mean(avg_dist), "points_reached": np.mean(self.points_reached_cnt),"car_detection": np.mean(self.detection_car_cnt), "collisions": np.mean(self.collision_cnt),  "collision_cnt_env": np.mean(self.collision_cnt_env),  "collision_cnt_car": np.mean(self.collision_cnt_car), "lane_invasions": np.mean(self.lane_invasion_cnt), "episode_reward_sum": total_sum_reward, "ep_steps": np.mean(self.ep_steps), "_n_updates": self._n_updates, "episode_num": self._episode_num})
						
						# reset episode loggers
						self.driving_score = np.zeros((self.log_interval)).astype(np.float32)
						self.points_reached_cnt = np.zeros((self.log_interval)).astype(np.float32)
						self.detection_car_cnt = np.zeros((self.log_interval)).astype(np.float32)
						self.collision_cnt = np.zeros((self.log_interval)).astype(np.float32)
						self.collision_cnt_car = np.zeros((self.log_interval)).astype(np.float32)
						self.collision_cnt_env = np.zeros((self.log_interval)).astype(np.float32)
						self.lane_invasion_cnt = np.zeros((self.log_interval)).astype(np.float32)
						self.speeding_cnt = np.zeros((self.log_interval)).astype(np.float32)	
						self.dist_to_center = np.zeros((self.log_interval)).astype(np.float32)	
						self.route_completion = np.zeros((self.log_interval)).astype(np.float32)	
						self.episode_duration = np.zeros((self.log_interval)).astype(np.float32)	
						self.timeout = np.zeros((self.log_interval)).astype(np.float32)	
						self.ep_steps = np.zeros((self.log_interval)).astype(np.float32)
						self.ep_rew = np.zeros((self.log_interval, self.config.num_rewards + 1)).astype(np.float32)

						# write the score here, this is used to save the best model
						if (((self._n_updates > (self.config.delay_actor_timesteps + self.config.key_steps)))):
							self.config.currentScore = mean_driving_score
							
		callback.on_rollout_end()
		
		if self._n_updates % self.config.eval_keys == 0 and ((self._n_updates > self.config.delay_actor_timesteps and self._n_updates < self.config.key_steps) or (self._n_updates > (self.config.delay_actor_timesteps + self.config.key_steps))):
			self.eval_key_objectives(env, learning_starts, num_collected_steps)

		return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

	def eval_key_objectives(self, env, learning_starts, num_collected_steps):
		"""
		Evaluate the agent on key objective weights in the environment.

		This function performs evaluation episodes for each key weight vector in
		`self.weight_batch`. The agent interacts with the environment without
		modifying the preference weights with noise, except for initial exploration
		if training has not yet started.

		Args:
			env: The environment to interact with (single or vectorized, not fully adapted for multiple envs).
			learning_starts (int): Number of steps before starting learning.
			num_collected_steps (int): Number of steps already collected for the current training phase.
		"""
		# Set a random spawn point for evaluation, fixed during this round
		self.config.eval_spawnpoint = random.randint(1, 256)

		# Ensure even step count for evaluation
		if self._n_updates == 0:
			self._n_updates += 2

		key_weights = self.weight_batch

		# Loop over each key weight vector
		for obj_num in range(len(key_weights)):
			current_weights = key_weights[obj_num]

			# Array to store rewards for each episode
			obj_reward = np.zeros((self.config.episodes_per_obj, self.config.num_rewards), dtype=np.float32)

			# Run evaluation episodes
			for key_episode_ in range(self.config.episodes_per_obj):
				_last_obs = env.reset()
				dones = False

				while not dones:
					# Apply current preference weights
					_last_obs['pref_weights'] = np.expand_dims(current_weights, axis=0)

					# Determine action noise
					if num_collected_steps > self.config.delay_actor_timesteps:
						# No noise during key objective evaluation
						action_noise = None
					else:
						# If training has not started, add exploration noise
						action_noise = NormalActionNoise(
							mean=np.array([self.steering_mean, 0.15]),
							sigma=np.array([0.2, 0.3]),
							dtype=np.float32
						)

					# Sample action from policy
					actions, _ = self.sample_action(_last_obs, learning_starts, action_noise, env.num_envs)

					# Step the environment
					new_obs, rewards, dones, infos = env.step(actions)
					_last_obs = new_obs

					# Accumulate rewards excluding core driving reward (index 0)
					obj_reward[key_episode_] += rewards[0][1:]

			# Compute mean objective reward across episodes
			current_obj = np.mean(obj_reward, axis=0).astype(np.float32)

			# Compute scalarized objectives for comparison
			scalarized_obj_prev = np.dot(current_weights, self.bestObjectiveInterpotorValues[obj_num])
			scalarized_obj_current = np.dot(current_weights, current_obj)

			# Update the best objective values if performance improved
			if scalarized_obj_current > scalarized_obj_prev:
				self.bestObjectiveInterpotorValues[obj_num] = current_obj.copy()
				time.sleep(0.1)

		# Reset eval_spawnpoint to zero for normal training with random spawn points
		self.config.eval_spawnpoint = 0

	def generate_weights_batch(self, step_size, reward_size):
		"""
		Generate a deterministic batch of weight vectors using a grid-based strategy.

		All possible combinations of weights in increments of `step_size` are generated.
		Only weight vectors whose elements sum exactly to 1 are kept. Additionally, a
		uniform weight vector (equal weights for all objectives) is always included.

		Args:
			step_size (float): Step size for generating weight combinations.
			reward_size (int): Number of objectives (dimensions of each weight vector).

		Returns:
			np.ndarray: Array of shape (num_vectors, reward_size) containing all valid
						weight vectors as float32, each summing to 1.
		"""
		mesh_array = []
		for i in range(reward_size):
			mesh_array.append(np.arange(0, 1 + step_size, step_size))

		# Generate all possible combinations of weights
		w_batch_test = np.array(list(itertools.product(*mesh_array)))

		# Keep only combinations where the sum of weights is exactly 1
		w_batch_test = w_batch_test[w_batch_test.sum(axis=1) == 1, :]

		# Remove duplicate rows
		w_batch_test = np.unique(w_batch_test, axis=0)

		# Add uniform weight vector if not already present
		additional_row = np.ones(reward_size) * float(1.0 / self.config.num_rewards)
		if not (additional_row in w_batch_test):
			w_batch_test = np.vstack([w_batch_test, additional_row])

		return w_batch_test.astype(np.float32)

	def generate_eval_weights(self, step_size, reward_size, samples):
		"""
		Generate stochastic evaluation weight vectors for testing policies.

		For each objective:
			- A fixed weight is assigned in increments of `step_size`.
			- The remaining weights are randomly sampled and normalized so the total sums to 1.
		
		This produces diverse, randomized weight vectors suitable for evaluation.

		Args:
			step_size (float): Step size for the fixed weight of the current objective.
			reward_size (int): Number of objectives (dimensions of each weight vector).
			samples (int): Number of random samples to generate for each step.

		Returns:
			list[list[float]]: List of weight vectors, each rounded to 3 decimal places 
							and summing to 1.
		"""
		weights = []
		num_objectives = reward_size

		for objective in range(num_objectives):
			for step in np.arange(0, 1 + step_size, step_size):
				# Ensure step does not exceed 1
				if step > 1:
					step = 1
				for _ in range(samples):
					# Randomly generate remaining weights
					other_weights = np.random.uniform(0, 1, num_objectives - 1)
					other_weights_sum = np.sum(other_weights)

					if step == 1:
						# If current objective weight is 1, all others are 0
						weight_vector = [0] * num_objectives
						weight_vector[objective] = 1
					else:
						# Normalize remaining weights so total sum is 1
						normalized_other_weights = other_weights * (1 - step) / other_weights_sum
						weight_vector = list(normalized_other_weights)
						weight_vector.insert(objective, step)

					# Round weights for numerical stability
					rounded_weights = [round(w, 3) for w in weight_vector]
					weights.append(rounded_weights)

		return weights

# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

	def update_lr(self):
		"""
		Update learning rates and weight decay for both Actor and Critic optimizers
		based on the current configuration.
		"""
		# Update Actor optimizer
		for param_group in self.actor.optimizer.param_groups:
			param_group['lr'] = self.config.lr_actor
			param_group['weight_decay'] = self.config.weight_decay
		
		# Update Critic optimizer
		for param_group in self.critic.optimizer.param_groups:
			param_group['lr'] = self.config.lr_critic
			param_group['weight_decay'] = self.config.weight_decay

		print("critic_optimizer\n", self.critic.optimizer)

	def load_config(self, config):
		"""
		Load a configuration object and apply it to this instance and the replay buffer.
		"""
		self.config = config
		self.replay_buffer.load_config(config)  # Allow replay buffer to update its config

	def load_running_max(self, path, device):
		"""
		Load the saved best objective values (interpolator maxima) from disk.
		
		Args:
			path (str): Path prefix to the saved numpy file.
			device (torch.device): Device to map loaded data if needed.
		"""
		filename = f"{path}_max.npy"
		self.bestObjectiveInterpotorValues = np.load(filename)

	def save_best_bestObjectiveInterpotorValues(self, path):
		"""
		Save the current best objective interpolator values to disk.
		
		Args:
			path (str): Path prefix to save the numpy file.
		"""
		filename = f"{path}_max"
		np.save(filename, self.bestObjectiveInterpotorValues)

	def set_path(self, path):
		"""
		Set the internal path for saving key-related files.
		
		Args:
			path (str): Base path for saving data.
		"""
		self.my_path = path + "_key"

	def save_replay_buffer(self, path):
		"""
		Save the current replay buffer to disk, including all relevant tensors.
		
		Args:
			path (str): Base path for the saved replay buffer file.
		"""
		filename = f"{path}_replay_buffer.pt"
		buffer_data = {
			"observations": self.replay_buffer.observations,
			"next_observations": self.replay_buffer.next_observations,
			"actions": self.replay_buffer.actions,
			"rewards": self.replay_buffer.rewards,
			"dones": self.replay_buffer.dones,
			"pointer": self.replay_buffer.pos,
		}
		th.save(buffer_data, filename)

	def load_replay_buffer(self, path, device):
		"""
		Load a previously saved replay buffer from disk and restore its state.
		
		Args:
			path (str): Base path for the saved replay buffer file.
			device (torch.device): Device to map loaded data.
		"""
		filename = f"{path}_replay_buffer.pt"
		buffer_data = th.load(filename, map_location=device)
		
		# Restore all components of the replay buffer
		self.replay_buffer.observations = buffer_data["observations"]
		self.replay_buffer.next_observations = buffer_data["next_observations"]
		self.replay_buffer.actions = buffer_data["actions"]
		self.replay_buffer.rewards = buffer_data["rewards"]
		self.replay_buffer.dones = buffer_data["dones"]
		self.replay_buffer.pos = buffer_data["pointer"]
		
		# Optional: move buffer to device
		# self.replay_buffer.to(device)
		
		print("loaded ReplayBuffer")

	def log_metrics(self, metrics):
		"""
		Log metrics to Weights & Biases (wandb).
		
		Args:
			metrics (dict): Dictionary of metric names and values.
		"""
		wandb.log(metrics)
		# Note: slow swap may occur if logging large objects

	def init_wandb_name(self, name):
		"""
		Initialize Weights & Biases (wandb) run with a given name and project settings.
		
		Args:
			name (str): Name of the wandb run. If empty, defaults to offline mode.
		"""
		print("wandb name", name)
		if not self.config.evaluate:
			if name != '':
				wandb.init(project="Carla_Master2", name=name,
						config={
							"device": th.device("cuda" if th.cuda.is_available() else "cpu"),
							"algo": "sb3_pdmorl",
							"hyperparameters": Hyperparameters().to_dict()
						})
			else:
				wandb.init(project="Carla_Master2", name=name,
						config={
							"device": th.device("cuda" if th.cuda.is_available() else "cpu"),
							"algo": "sb3_pdmorl",
							"hyperparameters": Hyperparameters().to_dict()
						},
						mode='offline')
			# Ensure wandb finishes cleanly on program exit
			atexit.register(finish_wandb)
