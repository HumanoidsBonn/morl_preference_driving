import carla

class Hyperparameters:
	def __init__(self):
		self.seed = 123456  # Random seed for reproducibility

		# Camera configuration
		self.IM_WIDTH = 448
		self.IM_HEIGHT = 224  # Padding may be added in the feature extractor if image is not square
		self.NUM_CHANNELS = 3  # Number of camera channels
		self.state_frames = 1  # Number of time frames in input; must be at least 1

		# Input tensor format: [batch_size, channels, depth, height, width]
		self.camera_shape = (self.NUM_CHANNELS, self.state_frames, self.IM_HEIGHT, self.IM_WIDTH)

		# CARLA simulation parameters
		self.MAX_TICKS_PER_EPISODE = 1400
		self.NUM_TRAFFIC_VEHICLES = 25  # Can be increased during training for realism
		self.max_waypoints = 500
		self.inter_wp_dist = 1  # Distance between waypoints considered (meters)
		self.max_route_deviation = 6  # Maximum allowed deviation to next waypoint (meters)
		self.target_wp_ahead = 5  # Distance to the next navigation waypoint along route
		self.track_width = 1.4  # Lane width in meters
		self.max_steering_angle = 70.0  # Max steering angle for the vehicle (degrees)
		self.max_steering_change = 9  # Max steering change per timestep (degrees)

		# World / simulation fixes
		self.load_new_world = False
		self.autopilotActions = False  # Use autopilot instead of actor
		self.fixed_delta_seconds = 0.1
		self.weather_change = int(2.5e4)
		self.useAutopilot = False  # Enable autopilot during exploration phase
		self.fps = 24

		# Scenario settings
		self.scenario = False  # Can be: "intersection", "traffic_low", "traffic_high", "tunnel", "roundabout", "highway", "crossing", etc.
		self.scenario_traffic = False  # Enable spawning traffic in scenarios
		self.evaluate_scenarios = False  # Evaluate agent on predefined scenarios

		# Reward parameters
		self.target_speed_perc = 0.8

		# Training parameters
		self.verbose = 0
		self.tm_port = 8000  # Traffic manager port; can be overwritten
		self.client_port = 2000  # CARLA client port; can be overwritten
		self.evaluate = False  # Train or evaluate the agent
		self.showPolicy = False  # Evaluate policy performance with multiple weight samples
		self.SPECATE = False  # Spectator follows the car; set False if rendering off-screen
		self.next_map = 'Town01'  # Initial map

		# Replay buffer
		self.REPLAY_BUFFER_SIZE = int(2e5)  # Large buffer; requires more memory

		# PDMORL-specific parameters
		self.key_steps = 0  # Number of steps for first phase # int(1e6)
		self.START_TIMESTEPS = int(2e3)
		self.delay_actor_timesteps = int(6e3)
		self.episodes_per_obj = 5
		self.eval_keys = int(5e3)
		self.num_rewards = 4  # Baseline reward included: Speed, Efficiency, Aggressiveness, Comfort
		self.psi = 1.5  # Actor angle weight
		self.zeta = 0.75  # Critic angle weight
		self.additionalPrefs = 2  # Extra sampled preferences for PDMORL replay buffer
		self.w_step_size = 1

		# Algorithm / TD3 parameters
		self.time_steps = self.key_steps + int(1e6)
		self.discount = 0.96
		self.policy_noise = 0.05
		self.noise_clip = 0.1
		self.policy_freq = 2
		self.tau = 0.01
		self.BATCH_SIZE = 256
		self.action_noise = 0.2
		self.lr_actor = 0.0001
		self.lr_critic = 0.0005
		self.weight_decay = 1e-8
		self.feature_dim = 126  # Must be divisible by 3

		self.mygroup = "None"  # WandB logging group

		# Helpers and flags (dont touch this)
		self.bestModelSave = False  # Saves best model during training
		self.bestScore = 0.05 # Performance evaluation threshold
		self.currentScore = 0 # Performance
		self.eval_spawnpoint = 0  # Spawn point for interpolator evaluation

	def to_dict(self):
		"""Convert all hyperparameters to a dictionary."""
		return self.__dict__
