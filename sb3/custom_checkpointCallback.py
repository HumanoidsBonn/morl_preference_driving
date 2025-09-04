import random
import carla
from stable_baselines3.common.callbacks import BaseCallback
import time
import os
from config import Hyperparameters

class CustomCheckpointCallback(BaseCallback):
	"""
	Callback for saving a model every ``save_freq`` calls
	to ``env.step()``.
	By default, it only saves model checkpoints,
	you need to pass ``save_replay_buffer=True``,
	and ``save_vecnormalize=True`` to also save replay buffer checkpoints
	and normalization statistics checkpoints.

	.. warning::

	  When using multiple environments, each call to  ``env.step()``
	  will effectively correspond to ``n_envs`` steps.
	  To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

	:param save_freq: Save checkpoints every ``save_freq`` call of the callback.
	:param save_path: Path to the folder where the model will be saved.
	:param name_prefix: Common prefix to the saved models
	:param save_replay_buffer: Save the model replay buffer
	:param save_vecnormalize: Save the ``VecNormalize`` statistics
	:param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
	"""

	def __init__(
		self,
		save_freq: int,
		save_path: str,
		name_prefix: str = "rl_model",
		save_replay_buffer: bool = False,
		save_vecnormalize: bool = False,
		verbose: int = 0,
	):
		super().__init__(verbose)
		self.save_freq = save_freq
		self.save_path = save_path
		self.name_prefix = name_prefix
		self.save_replay_buffer = save_replay_buffer
		self.save_vecnormalize = save_vecnormalize
		self.offset = 0
		self.config = Hyperparameters()

	def load_config(self, config):
		self.config = config

	def set_step(self, n_calls):
		self.n_calls = n_calls
		self.offset = n_calls
		print("n_calls_offset", self.offset)

	def _init_callback(self) -> None:
		# Create folder if needed
		if self.save_path is not None:
			os.makedirs(self.save_path, exist_ok=True)

	def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
		"""
		Helper to get checkpoint path for each type of checkpoint.

		:param checkpoint_type: empty for the model, "replay_buffer_"
			or "vecnormalize_" for the other checkpoints.
		:param extension: Checkpoint file extension (zip for model, pkl for others)
		:return: Path to the checkpoint
		"""
		fixed = self.offset + self.num_timesteps
		if self.config.bestModelSave: # and self.config.bestPrefSave:
			return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}bestCombined.{extension}")
		# if self.config.bestModelSave:
		# 	return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}best.{extension}")
		# if self.config.bestPrefSave:
		# 	return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}bestPref.{extension}")
		return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{fixed}_steps.{extension}")

	def _on_step(self) -> bool:
		if self.n_calls % self.save_freq == 0 or (self.config.bestModelSave):
			model_path = self._checkpoint_path(extension="zip")
			print(f"Saving model checkpoint to {model_path}")
			print("at timestep ", self.offset + self.num_timesteps)
			self.model.save(model_path)
			time.sleep(7)
			#print("Saved succesfull.")

			if self.verbose >= 2:
				print(f"Saving model checkpoint to {model_path}")

			if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
				# If model has a replay buffer, save it too
				replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
				self.model.save_replay_buffer(replay_buffer_path)
				if self.verbose > 1:
					print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

			if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
				# Save the VecNormalize statistics
				vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
				self.model.get_vec_normalize_env().save(vec_normalize_path)
				if self.verbose >= 2:
					print(f"Saving model VecNormalize to {vec_normalize_path}")
			
			# reset flags
			self.config.bestModelSave = False
		return True
