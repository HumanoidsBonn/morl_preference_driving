from stable_baselines3.common.callbacks import BaseCallback

# List of weather presets for simulation
weather_presets = [
	"ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon",
	"HardRainNoon", "SoftRainNoon", "MidRainyNoon",
	"ClearSunset", "CloudySunset", "WetSunset", "WetCloudySunset",
	"HardRainSunset", "SoftRainSunset", "MidRainSunset"
]

# Towns to be used for training; the training will iterate through these towns
towns = ["Town01", "Town03", "Town02", "Town04"]

class MyCallback(BaseCallback):
	"""
	Custom callback to manage dynamic town switching during training.

	:param check_freq: Frequency for performing callback checks
	:param env: Training environment
	:param config: Configuration object containing environment and training settings
	:param save_freq: Frequency for saving models
	:param model: The RL model being trained
	:param path: Path to save checkpoints
	:param verbose: Verbosity level
	"""
	def __init__(self, check_freq, env, config, save_freq, model, path, verbose=0):
		super().__init__(verbose)
		self.check_freq = check_freq
		self.env = env
		self.config = config
		self.model = model
		self.save_freq = save_freq
		self.path = path

		# Number of timesteps before switching towns
		self.map_explore = int(2.5 * 1e4)
		self.town_counter = 0

	def set_step(self, n_calls):
		"""
		Manually update the step counter (SB3 style).
		"""
		self.n_calls = n_calls
		self.num_timesteps = n_calls

	def _on_step(self) -> bool:
		"""
		Called at each training step. Switches the town periodically
		to expose the agent to different maps during training.
		"""
		# Switch town every `map_explore` steps after the key steps phase
		if (self.n_calls + 2300) % self.map_explore == 0 and self.n_calls > self.config.key_steps:
			# Select next town from the list
			town = towns[self.town_counter]

			# Increment counter cyclically
			self.town_counter = (self.town_counter + 1) % len(towns)

			# Load the new map if it's different from the current one
			if town != self.config.next_map:
				self.config.next_map = town
				print("Next map:", self.config.next_map)
				self.config.load_new_world = True

		return True
