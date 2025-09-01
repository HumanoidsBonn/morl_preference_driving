# Deaktiviert die GPU-Nutzung
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import os
import carla
import torch
import numpy as np
import time
import argparse

# from stable_baselines3 import TD3
# from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecMonitor
#from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList

# Importieren benutzerdefinierter Klassen
from policy_exection import CustomMultiInputPolicy
from pdmorl_train import PDMORL_TD3
from prefHer_replayBuffer import PDMORL_DictReplayBuffer
from custom_monitor import CustomMonitor
from custom_callback import MyCallback
from custom_checkpointCallback import CustomCheckpointCallback
from select_extractor import CustomCombinedExtractor

from custom_vectorEnvWrapper import DummyVecEnv
from carla_gym_env import CarlaEnv 
from config import Hyperparameters

# start carla server first, e.g in folder carlaSim.
# ./CarlaUE4.sh -RenderOffScreen -world-port=2004 -graphicsadapter=$GPU_ID > carla_output.txt 2>&1 &
# python td3_main.py --run=XXX --client_port=2004 --tm_port=8004

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Custom SB3 Model')

	# Add arguments for model run suffix and CARLA ports
	parser.add_argument('--run', type=str, default='', help='Suffix for the model prefix')
	parser.add_argument('--client_port', type=int, default=2000, help='CARLA client port')
	parser.add_argument('--tm_port', type=int, default=8000, help='Traffic manager port')

	# Parse the command-line arguments
	args = parser.parse_args()
	time.sleep(1)

	# Load hyperparameters
	config = Hyperparameters()
	time.sleep(1)
	config.client_port = args.client_port
	config.tm_port = args.tm_port

	# Determine whether to train or evaluate
	evaluate_model = config.evaluate
	TrainTD3 = not evaluate_model

	# Connect to the CARLA server
	localhost = 'localhost'
	client = carla.Client(localhost, config.client_port)
	client.set_timeout(100.0)
	time.sleep(1)

	# Create the CARLA environment
	env = CarlaEnv(client=client, config=config)
	time.sleep(1)

	# Set device to GPU if available, otherwise CPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	time.sleep(1)
	# print("Device:", device)

	if TrainTD3:
		# Wrap the environment with a monitor for logging
		env = CustomMonitor(env)
		# Wrap the environment to make it vectorized for SB3 compatibility
		env = DummyVecEnv([lambda: env])

		# Initialize the PD-MORL TD3 model
		model = PDMORL_TD3(
			CustomMultiInputPolicy,
			env,
			policy_kwargs=dict(
				net_arch=[250, 125],  # Network architecture
				# net_arch=[256, 128, 64],  # Alternative architecture
				features_extractor_class=CustomCombinedExtractor,
				features_extractor_kwargs={},  # Extra kwargs for the feature extractor
				normalize_images=True,  # Enable image normalization
				share_features_extractor=True,  # Share feature extractor between actor and critic
				# activation_fn=nn.LeakyReLU,  # Optional activation function
			),
			replay_buffer_class=PDMORL_DictReplayBuffer,
			buffer_size=config.REPLAY_BUFFER_SIZE,
			learning_starts=100,
			batch_size=config.BATCH_SIZE,
			tau=config.tau,
			gamma=config.discount,
			train_freq=4,
			action_noise=NormalActionNoise(
				mean=np.array([0, 0.3]),
				sigma=np.array([0.1, 0.2]),
				dtype=np.float32
			),  # Initial action noise, will be overwritten later
			target_policy_noise=config.policy_noise,
			target_noise_clip=config.noise_clip,
			verbose=config.verbose,
			_init_setup_model=True,
		)

		# Load configuration into the model
		model.load_config(config)

		# Base path to save model runs
		path = '../run/'

		# Construct a prefix for this run using the command-line argument
		prefix = f'pdmorl_{args.run}'
		print("Prefix of path: ", prefix)

		# Combine base path and prefix for saving/loading
		combined_path = f"{path}{prefix}"
		print("Combined path:", combined_path)

		# Set the path inside the model for saving outputs
		model.set_path(combined_path)

		# Print the architecture of the Actor network
		print("Actor network architecture:")
		print(model.actor)

		# Optionally, print the Critic network architecture
		# print("\nCritic network architecture:")
		# print(model.critic)

		# Initialize Weights & Biases logging with the given run name
		model.init_wandb_name(args.run)

		# Callback for periodic model saving
		checkpoint_callback = CustomCheckpointCallback(
			save_freq=int(1e5), 
			save_path=path, 
			name_prefix=prefix
		)
		checkpoint_callback.load_config(config)

		# Create a custom callback for environment updates / world resets / loading new towns
		my_callback = MyCallback(
			check_freq=int(0.5 * 1e5),
			env=env,
			config=config,
			save_freq=int(0.5 * 1e5),
			model=model,
			path=combined_path
		)

		# Start the learning process with both callbacks
		model.learn(
			total_timesteps=config.time_steps,
			callback=CallbackList([checkpoint_callback, my_callback])
		)

	if evaluate_model:
		# Path to the pretrained model for evaluation
		model_path = "../run/TODO.zip"

		# Small delays to ensure environment and resources are ready
		time.sleep(5)
		
		# Wrap the environment with DummyVecEnv for compatibility
		env = DummyVecEnv([lambda: env])
		time.sleep(5)
		print("Environment wrapped")

		# Print and track model name
		model_name = f'{args.run}'
		print("Model path: ", model_path)
		time.sleep(20)

		# Load the pretrained PDMORL_TD3 model
		print("Loading model")
		model = PDMORL_TD3.load(model_path)  # Optionally, custom_objects can be provided
		print("Model loaded")
		time.sleep(20)

		# Reset the environment for evaluation
		obs = env.reset()
		total_rewards = []

		# Optional adjustments for observation or visualization
		# if config.showPolicy:
		#     config.MAX_TICKS_PER_EPISODE = 1500
		#     config.max_waypoints = 500
		#     config.NUM_TRAFFIC_VEHICLES = 45
		#     config.NUM_TRAFFIC_VEHICLES = 0  # Alternative: disable traffic for testing

		import pandas as pd

		# Initialize logging structures for evaluation if not evaluating specific scenarios
		if not config.evaluate_scenarios:
			log_scores = {
				"episode": [],
				"driving_score": [],
				"pref_score": [],
			}

			print("Weights: Speed, Efficiency, Aggressiveness, Comfort")
			# Optionally generate stochastic evaluation weights
			# weights = model.generate_eval_weights(step_size=0.2, reward_size=config.num_rewards, samples=20)

			# For policy visualization, generate a deterministic batch of weights
			if config.showPolicy:
				weights = model.generate_weights_batch(step_size=0.5, reward_size=config.num_rewards)
				# Alternative: manually specify all single-objective weights
				# weights = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

			# Round weights for display
			weights = np.round(weights, 3)
			print(f"Evaluate {len(weights)} Episodes")

			# Reset environment before evaluation
			obs = env.reset()

			# Iterate through each weight vector for evaluation
			for episode, weight in enumerate(weights):
				done = False
				episode_rewards = np.zeros(config.num_rewards + 1, dtype=np.float32)
				prefScoreWeighted = 0
				steps = 0

				print("weights", weight)

				while not done:
					# Set the current preference weights in observation
					obs["pref_weights"] = weight

					# Predict action deterministically using the trained model
					action, _states = model.predict(obs, deterministic=True)
					action = np.round(action, 4)

					# Step environment with predicted action
					obs, rewards, done, info = env.step(action)

					# Unpack info dictionary from DummyVecEnv
					info = info[0]

					# Accumulate rewards
					episode_rewards += rewards[0]
					# Weighted sum of preference rewards (exclude core driving reward at index 0)
					prefScoreWeighted += np.sum(rewards[0][1:] * weight)
					steps += 1

					current_wp = float(info['current_wp_index'])

					# After episode ends, compute average preference score
					if current_wp >= 0 and done:
						prefScoreWeighted /= steps

						# Log evaluation scores
						log_scores["episode"].append(episode)
						log_scores["driving_score"].append(float(info['adjusted_driving_core']))
						log_scores["pref_score"].append(np.round(prefScoreWeighted, 4))

						print(f"Episode {episode} finished after {steps} steps "
							f"with driving score: {info['adjusted_driving_core']}, "
							f"pref score: {np.round(prefScoreWeighted, 4)}, "
							f"rewards: {np.round(episode_rewards, 4)}")

						break

				# Optional: save logs per episode or after all episodes
				save_path = "logs/csv/" + str(model_name)
				# df_log_scores = pd.DataFrame(log_scores)
				# df_log_scores.to_csv(save_path + "_scores.csv", index=False)
				# print("Scores data saved.")
				# time.sleep(1)

		if config.evaluate_scenarios:
			# Helper function: calculate mean and std, rounded to 3 decimals
			def calc_mean_std(data):
				mean = round(np.mean(data), 3)
				std = round(np.std(data), 3)
				return mean, std

			# Load existing evaluation results from CSV
			def load_existing_results(file_path):
				if os.path.exists(file_path):
					return pd.read_csv(file_path)
				else:
					return pd.DataFrame()

			# Save or append scenario results to CSV
			def save_scenario_results(file_path, scenario_results):
				if os.path.exists(file_path):
					existing_data = pd.read_csv(file_path)
					scenario_results = pd.concat([existing_data, scenario_results], ignore_index=True)
				scenario_results.to_csv(file_path, index=False)
				time.sleep(1)

			# Path to save evaluation results
			results_file_path = "table/" + model_name + ".csv"
			existing_results = load_existing_results(results_file_path)

			# Define scenarios for evaluation
			scenarios = ["intersection", "traffic_high", "traffic_low", "tunnel", "roundabout", "highway", "crossing"]

			# Filter out scenarios that have already been evaluated
			evaluated_scenarios = set(existing_results["scenario"]) if not existing_results.empty else set()
			scenarios_to_evaluate = [s for s in scenarios if s not in evaluated_scenarios]

			# Evaluate each new scenario
			for scenario in scenarios_to_evaluate:
				config.scenario = scenario
				print("Scenario:", scenario)

				# Map scenarios to specific CARLA maps
				if scenario == "intersection":
					config.next_map = "Town01"
				elif scenario in ["traffic_high", "traffic_low"]:
					config.next_map = "Town02"
				elif scenario in ["roundabout", "tunnel"]:
					config.next_map = "Town03"
				elif scenario in ["crossing", "highway"]:
					config.next_map = "Town04"

				time.sleep(1)

				# Load the new world in CARLA
				config.load_new_world = True
				obs = env.reset()
				done = False

				# Run the scenario until completion
				while not done:
					action, _ = model.predict(obs, deterministic=True)
					obs, rewards, done, info = env.step(action)

				# Generate evaluation weights for multiple episodes
				weights = np.round(
					model.generate_eval_weights(step_size=0.125, reward_size=config.num_rewards, samples=12), 3
				)
				print(f"Evaluating {len(weights)} episodes for scenario: {scenario}")

				# Metrics to accumulate results
				scenario_ds = []
				scenario_ps = []
				scenario_coll_env_rate = []
				scenario_coll_car_rate = []
				scenario_dist_to_center = []
				scenario_lane_invasion_rate = []
				scenario_speeding_rate = []
				scenario_route_completion = []
				scenario_episode_duration = []
				scenario_jerk_rate = []
				scenario_reward = []

				# Loop over all generated preference weight episodes
				for episode in range(len(weights)):
					done = False
					prefScoreWeighted = 0
					steps = 0
					obs = env.reset()
					total_reward = 0

					while not done:
						steps += 1
						# Set the current episode's preference weights
						obs["pref_weights"] = weights[episode]

						# Predict deterministic action using the trained model
						action, _states = model.predict(obs, deterministic=True)
						obs, rewards, done, info = env.step(action)

						# Unpack info and rewards for easier access
						info = info[0]
						rewards = rewards[0]

						# Accumulate weighted preference score (skip first core driving reward)
						prefScoreWeighted += np.sum(rewards[1:] * weights[episode])
						total_reward += np.sum(rewards)

						if done:
							# Normalize preference score by number of steps
							prefScoreWeighted /= steps
							driving_score = float(info['driving_score'])

							# Store scenario metrics
							scenario_route_completion.append(round(info["route_completion"], 4))
							scenario_ds.append(driving_score)
							scenario_ps.append(round(prefScoreWeighted, 4))
							scenario_reward.append(round(total_reward, 4))
							scenario_coll_env_rate.append(round(info["collision_cnt_env"] / steps * 100, 4))
							scenario_coll_car_rate.append(round(info["collision_cnt_car"] / steps * 100, 4))
							scenario_dist_to_center.append(round(info["dist_to_center"] / steps, 4))
							scenario_lane_invasion_rate.append(round(info["lane_invasion"] / steps * 10, 4))
							scenario_speeding_rate.append(round(info["speeding_cnt"] / steps * 100000, 4))
							scenario_episode_duration.append(round(info["episode_duration"], 4))
							break
				
				# Compute mean and standard deviation for all metrics in this scenario
				mean_ds, std_ds = calc_mean_std(scenario_ds)
				mean_ps, std_ps = calc_mean_std(scenario_ps)
				mean_coll_env_rate, std_coll_env_rate = calc_mean_std(scenario_coll_env_rate)
				mean_coll_car_rate, std_coll_car_rate = calc_mean_std(scenario_coll_car_rate)
				mean_dist_to_center, std_dist_to_center = calc_mean_std(scenario_dist_to_center)
				mean_lane_invasion_rate, std_lane_invasion_rate = calc_mean_std(scenario_lane_invasion_rate)
				mean_speeding_rate, std_speeding_rate = calc_mean_std(scenario_speeding_rate)
				mean_route_completion, std_route_completion = calc_mean_std(scenario_route_completion)
				mean_episode_duration, std_episode_duration = calc_mean_std(scenario_episode_duration)
				mean_reward, std_reward = calc_mean_std(scenario_reward)

				# Collect results into a DataFrame for this scenario
				scenario_results = pd.DataFrame([{
					"scenario": scenario,
					"driving_score": mean_ds,
					"pref_score": mean_ps,
					"reward": mean_reward,
					"collision_env_rate": mean_coll_env_rate,
					"collision_car_rate": mean_coll_car_rate,
					"dist_to_center": mean_dist_to_center,
					"lane_invasion_rate": mean_lane_invasion_rate,
					"speeding_rate": mean_speeding_rate,
					"route_completion": mean_route_completion,
					"episode_duration": mean_episode_duration,

					"driving_score_std": std_ds,
					"pref_score_std": std_ps,
					"reward_std": std_reward,
					"collision_env_rate_std": std_coll_env_rate,
					"collision_car_rate_std": std_coll_car_rate,
					"dist_to_center_std": std_dist_to_center,
					"lane_invasion_rate_std": std_lane_invasion_rate,
					"speeding_rate_std": std_speeding_rate,
					"route_completion_std": std_route_completion,
					"episode_duration_std": std_episode_duration,
				}])

				# Save the scenario results
				save_scenario_results(results_file_path, scenario_results)
				print(f"Results for scenario {scenario} saved.")
			
			
			# Compute overall mean and mean of standard deviations across all scenarios
			print("Evaluation completed.")
			time.sleep(1)

			metrics = [
				"route_completion", "driving_score", "pref_score", "reward", "collision_env_rate",
				"collision_car_rate", "dist_to_center", "car_response_rate", "lane_invasion_rate",
				"speeding_rate", "episode_duration",
			]

			existing_results = load_existing_results(results_file_path)

			# Calculate mean and mean of std for each metric
			mean_results = {"scenario": "mean"}
			for metric in metrics:
				mean_metric, _ = calc_mean_std(existing_results[metric].dropna())
				mean_results[metric] = mean_metric

				std_mean, _ = calc_mean_std(existing_results[f"{metric}_std"].dropna())
				mean_results[f"{metric}_std"] = std_mean

			# Convert to DataFrame and append to existing results
			df_mean_results = pd.DataFrame([mean_results])
			df_final_results = pd.concat([existing_results, df_mean_results], ignore_index=True)

			# Ensure proper column order for CSV
			columns_order = [
				"scenario", "route_completion", "driving_score", "pref_score", "reward",
				"collision_env_rate", "collision_car_rate", "dist_to_center", "car_response_rate",
				"lane_invasion_rate", "speeding_rate", "episode_duration", "jerk_rate",
				"route_completion_std", "driving_score_std", "pref_score_std", "reward_std",
				"collision_env_rate_std", "collision_car_rate_std", "dist_to_center_std", "car_response_rate_std",
				"lane_invasion_rate_std", "speeding_rate_std", "episode_duration_std", "jerk_rate_std"
			]
			df_final_results = df_final_results[columns_order]

			# Save final results with means included
			df_final_results.to_csv(results_file_path, index=False)
			print("Mean and mean-of-std results added to the CSV file.")
			exit(-1)
			