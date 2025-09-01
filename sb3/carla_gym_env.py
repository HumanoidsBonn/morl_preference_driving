import numpy as np
import random
import time
import math
import gymnasium as gym
from gymnasium import spaces
import carla
from carla_env_utils import CarlaEnvUtils
import imageio
import queue

class CarlaEnv(gym.Env):
	# gymnasium environment
	def __init__(self, client, config):
		# Initialize the CARLA environment
		super(CarlaEnv, self).__init__()

		self.client = client

		# Initialize the hyperparameters class
		self.config = config

		if config.showPolicy:
			# Time of the last tick
			self.last_tick_time = time.time()
			# Interval for X FPS
			self.tick_interval = 1 / self.config.fps

		# Observation space
		self.observation_space = spaces.Dict({
			'camera': spaces.Box(low=0, high=255, shape=self.config.camera_shape, dtype=np.uint8),

			# see params in get_vehicle_measurements(vehicle):
			'measurements': spaces.Box(
				low=np.array([0, -130 / 3.6, -130 / 3.6, 0, -130 / 3.6, -130 / 3.6, -100 / 3.6, -100 / 3.6, 0, -1, 0], dtype=np.float32),
				high=np.array([130 / 3.6, 130 / 3.6, 130 / 3.6, 200, 130 / 3.6, 130 / 3.6, 100 / 3.6, 100 / 3.6, 1, 1, 1], dtype=np.float32),
				dtype=np.float32),

			'pref_weights': spaces.Box(
				low=np.zeros(self.config.num_rewards, dtype=np.float32),
				high=np.ones(self.config.num_rewards, dtype=np.float32),
				dtype=np.float32),

			'n_action_hist': spaces.Box( # brake, steering, throttle, delta_steering x timesteps_action_hist
				low=np.array([0, -1, 0, -self.config.max_steering_change,] * 3, dtype=np.float32),
				high=np.array([1, 1, 1, self.config.max_steering_change,] * 3, dtype=np.float32),
				dtype=np.float32),

			'signs': spaces.Box(
				low=np.array([0], dtype=np.float32),
				high=np.array([130.1], dtype=np.float32),
				dtype=np.float32),  # 'speed limit'

			'next_wp': spaces.Box(
				low=np.array([-math.pi, 0], dtype=np.float32),
				high=np.array([math.pi, self.config.max_route_deviation], dtype=np.float32)
				, dtype=np.float32),
		})

		# Action space (Steering and throttle)
		self.action_space = spaces.Box( #limit the action space ? (e.g 3 degrees in steering change)
			low=np.array([-1.0, -1.0]),
			high=np.array([1.0, 1.0]),
			shape=(2,),
			dtype=np.float32
		)

		# Load the world and set weather
		self.world = self.client.load_world(self.config.next_map)
		self.world.set_weather(carla.WeatherParameters(
			cloudiness=20.0, 
			precipitation=0.0, 
			sun_altitude_angle=90.0
		))

		# Create new settings with synchronous mode enabled
		self.settings = carla.WorldSettings(
			no_rendering_mode=not self.config.SPECATE,  # Toggle rendering on/off
			synchronous_mode=True,  # Enable synchronous mode
			fixed_delta_seconds=0.1,  # Constant time interval for each simulation step (e.g., 20 FPS) 
			substepping = True,
			max_substep_delta_time = 0.02, #0.01 fixed_delta_seconds sollte immer kleiner oder gleich dem Produkt aus max_substep_delta_time und max_substeps sein
			max_substeps = 5, 
			actor_active_distance = 800, # Meter
		)

		self.world.apply_settings(self.settings)
		self.map = self.world.get_map()
		self.tm = self.client.get_trafficmanager(self.config.tm_port)
		self.tm.set_synchronous_mode(True)
		self.tm.set_respawn_dormant_vehicles(True)
		self.max_spawn_dist = 180  # Distance of traffic spawn in meters to the actor
		self.tm.set_boundaries_respawn_dormant_vehicles(25, self.max_spawn_dist)

		# Transformation of cameras
		self.transform_sensors = carla.Transform(carla.Location(z=1.535, x=1.955, y=0))
		self.depth_transform = carla.Transform(carla.Location(z=1.535, x=1.955, y=0), carla.Rotation(pitch=-30))
		self.cam_init_trans = carla.Transform(carla.Location(z=1.535, x=1.955, y=0), carla.Rotation(pitch=-30))

		if not self.config.evaluate:
			self.tm.set_hybrid_physics_mode(True)
			self.tm.set_hybrid_physics_radius(60)

		self.bp_lib = self.world.get_blueprint_library()
		self.spawn_points = self.map.get_spawn_points()
		self.vehicle_bp_ego = self.bp_lib.find('vehicle.mercedes.coupe_2020')
		self.vehicle_bp_ego.set_attribute('role_name', 'hero')
		self.transform = None  # Initialisierung der Transformation für den Spectator
		self.last_vehicle_location = None  # Initialisierung der Location für den Spectator
		self.list_spec_pos = []

		self.collision_hist = []
		self.sensor_list = []
		self.actor_list = []
		self.traffic_list = []
		self.car_liste = ['*vehicle.f*', '*vehicle.mer*', '*vehicle.n*', '*vehicle.a*', '*vehicle.t*', '*vehicle.do*', '*vehicle.bm*']

		self.lane_invasion_hist = []

		# Initialize camera observations
		self.camera_observation = np.zeros(self.config.camera_shape, dtype=np.uint8)  # segment and depth
		self.front_camera = np.zeros((self.config.NUM_CHANNELS, self.config.IM_HEIGHT, self.config.IM_WIDTH), dtype=np.uint8)  # segment and depth
		
		self.d_camera = np.zeros((self.config.IM_HEIGHT, self.config.IM_WIDTH), dtype=np.uint8)
		self.s_camera = np.zeros((self.config.IM_HEIGHT, self.config.IM_WIDTH), dtype=np.uint8)
		self.rgb_camera = np.zeros((3, self.config.IM_HEIGHT, self.config.IM_WIDTH), dtype=np.uint8)
		self.seg_rgb_camera = np.zeros((3, self.config.IM_HEIGHT, self.config.IM_WIDTH), dtype=np.uint8)

		self.seg_rgb_camera_queue = queue.Queue()
		self.rgb_camera_queue = queue.Queue()
		self.depth_camera_queue = queue.Queue()
		self.seg_camera_queue = queue.Queue()
		self.collision_queue = queue.Queue()
		self.lane_invasion_queue = queue.Queue()

		# Initialize preference weights
		self.pref_weights_round = np.zeros(self.config.num_rewards, dtype=np.float32)
		
		# Initialize loggers
		self.points_reached = 0

		self.lane_invasion_cnt = 0
		self.collision_cnt_log = 0
		self.collision_cnt_env_log = 0
		self.collision_cnt_car_log = 0
		self.speeding_cnt = 0
		self.collision_cnt = 0
		self.total_dist_to_center = 0
		self.distance_travelled = 0

		# Initialize lists for measurements and variables for state tracking
		self.measurements = []
		self.action_hist = []
		self.distances = []
		self.blocked = False
		self.reduced = False
		self.lane_invasion = False
		self.next_waypoint = []
		self.tf_light_state = 0
		self.episode_ticks = 0
		self.respawnen_wp = True
		self.timeout_ticks = 0
		self.timeout_index = 0
		self.timeout = 0

		# Initialize flags for route creation
		self.route = []
		self.current_index = 0
		self.prev_index = 0
		self.prev_location = None
		
		self.init_sensors()

	def init_sensors(self):
		# cameras
		self.seg_rgb_cam = self.bp_lib.find('sensor.camera.semantic_segmentation')
		self.seg_rgb_cam.set_attribute("image_size_x", f"{self.config.IM_WIDTH}")
		self.seg_rgb_cam.set_attribute("image_size_y", f"{self.config.IM_HEIGHT}")
		self.seg_rgb_cam.set_attribute("fov", f"110")

		self.rgb_cam = self.bp_lib.find('sensor.camera.rgb')
		self.rgb_cam.set_attribute("image_size_x", f"{self.config.IM_WIDTH}")
		self.rgb_cam.set_attribute("image_size_y", f"{self.config.IM_HEIGHT}")
		self.rgb_cam.set_attribute("fov", f"110")

		self.seg_cam = self.bp_lib.find('sensor.camera.semantic_segmentation')
		self.seg_cam.set_attribute("image_size_x", f"{self.config.IM_WIDTH}")
		self.seg_cam.set_attribute("image_size_y", f"{self.config.IM_HEIGHT}")
		self.seg_cam.set_attribute("fov", f"110")

		self.depth_cam_bp = self.bp_lib.find('sensor.camera.depth')
		self.depth_cam_bp.set_attribute("image_size_x", f"{self.config.IM_WIDTH}")
		self.depth_cam_bp.set_attribute("image_size_y", f"{self.config.IM_HEIGHT}")
		self.depth_cam_bp.set_attribute("fov", f"110")

		self.colsensor_bp = self.bp_lib.find("sensor.other.collision")
		self.lanesensor_bp = self.bp_lib.find('sensor.other.lane_invasion')

	def reload_map(self):
		self.destroy_all_actors()
		time.sleep(1)
		# print("Loading new world: ", self.config.next_map)
		self.client.set_timeout(300.0)
		self.world = self.client.load_world(self.config.next_map)
		time.sleep(3)	
		# print("World loaded")
		self.world.set_weather(carla.WeatherParameters(
			cloudiness=20.0, 
			precipitation=0.0, 
			sun_altitude_angle=90.0
		))
		self.world.apply_settings(self.settings)
		time.sleep(3)
		self.map = self.world.get_map()
		self.client.set_timeout(180.0)
		time.sleep(3)
		# print("Map loaded")
		if self.config.SPECATE:
			self.spectator = self.world.get_spectator() 

		self.bp_lib = self.world.get_blueprint_library()
		self.init_sensors()
		self.spawn_points = self.map.get_spawn_points()
		self.reset_spawn_points()
		self.config.load_new_world = False
		# print("Done")

	def reset(self, seed=None, options=None):
		info = {}  # Add useful information if needed
		successful_created_route = False
		while not successful_created_route:
			# Destroy all existing actors and clear the list
			self.destroy_actors()

			# Reset the spawn points
			self.reset_spawn_points()

			# Alle lebenden Akteure abrufen
			all_actors = self.world.get_actors()

			# Create the vehicle and traffic
			if self.config.scenario:
				scenarios = [
				{
					"name": "intersection",
					"map": "Town01",
					"seed": 8924,
					"tm_seed": 8924,
					"max_waypoints": 250,
					"spawnpoint": 8,
					"traffic_spawnpoints": (4, 6, 7, 9, 10, 11, 14, 17, 28, 58, 60, 61, 66, 69, 70, 73, 74, 76, 93, 94, 96, 98, 99, 100, 102, 107, 109, 112)
				},
				{
					"name": "higher_traffic",
					"map": "Town02",
					"seed": 123457,
					"tm_seed": 123457,
					"max_waypoints": 300,
					"spawnpoint": 57,
					"traffic_spawnpoints": (2, 3, 6, 7, 8, 9, 12, 13, 16, 18, 19, 22, 28, 33, 36, 37, 46, 47, 49, 50, 51, 54, 56, 59, 61, 64, 69, 70, 72, 73, 76, 84, 86, 88, 90, 97, 98)
				},
				{
					"name": "lower_traffic",
					"map": "Town02",
					"seed": 123457,
					"tm_seed": 123457,
					"max_waypoints": 300,
					"spawnpoint": 57,
					"traffic_spawnpoints": (2, 3, 7, 12, 18, 19, 28, 33, 36, 47, 51, 59, 70, 72, 73, 76, 84, 88, 90, 97)
				},
				{
					"name": "tunnel",
					"map": "Town03",
					"seed": 234567,
					"tm_seed": 1234,
					"max_waypoints": 320,
					"spawnpoint": 78,
					"traffic_spawnpoints": (1, 2, 13, 14, 18, 19, 20, 21, 26, 32, 34, 35, 38, 41, 53, 56, 58, 63, 79, 83, 89, 95, 99, 100, 104, 105, 108, 109, 111, 113, 116, 119, 120, 121)
				},
				{
					"name": "roundabout",
					"map": "Town03",
					"seed": 582013,
					"tm_seed": 582013,
					"max_waypoints": 300,
					"spawnpoint": 0,
					"traffic_spawnpoints": (2, 4, 6, 7, 8, 9, 12, 13, 19, 22, 37, 43, 46, 47, 49, 50, 54, 56, 61, 64, 69, 70, 72, 84, 86, 98, 104, 105, 108, 111, 114, 120, 121, 123)
				},
				{
					"name": "highway",
					"map": "Town04",
					"seed": 582013,
					"tm_seed": 582013, 
					"max_waypoints": 800,
					"spawnpoint": 9,
					"traffic_spawnpoints": (2, 6, 7, 8, 12, 13, 19, 22, 37, 43, 46, 47, 49, 50, 54, 56, 61, 64, 70, 72, 84, 86, 98, 104, 105, 108, 111, 114, 120, 121, 123, 207)
				},			
				{
					"name": "crossing",
					"map": "Town04",
					"seed": 58206,
					"tm_seed": 582013,
					"max_waypoints": 300,
					"spawnpoint": 166,
					"traffic_spawnpoints": (251, 252, 250, 254, 247, 248, 249, 230, 231, 232, 233, 182, 183, 162, 164, 177, 178, 170, 195, 192, 261, 192)
				},
				]

				# Processing the scenario
				found_scenario = False
				for scenario in scenarios:
					if self.config.scenario == scenario["name"]:
						found_scenario = True
						if self.config.next_map != scenario["map"]:
							print(f"Wrong map loaded for {scenario['name']} scenario with {self.config.next_map}")
							exit(-1)
						self.config.seed = scenario["seed"]
						self.config.max_waypoints = scenario["max_waypoints"]
						self.tm.set_random_device_seed(scenario["tm_seed"])
						self.create_vehicle(spawnpoint=scenario["spawnpoint"])
						if self.config.scenario_traffic:
							self.create_traffic_scenario(scenario["traffic_spawnpoints"])
						break

				if not found_scenario:
					print("Scenario not found!")
					exit(-1)
			else:
				# normal training
				self.config.seed = np.random.randint(1, 100000)
				self.create_vehicle(spawnpoint=-1) 

				self.create_traffic()

			self.route, successful_created_route = CarlaEnvUtils.create_route(self.world, self.spawn_point, self.config, self.map)

		# Configure the camera and collision sensor
		# With more than multiple cameras, the simulator crashes at training start if the CPU is overloaded
		
		# self.configure_camera()  # RGB
		self.configure_SegRGB()  # Seg-RGB
		# self.configure_depthcamera() # Depth
		# self.configure_segcamera()  # Segmentation 1-channel
		self.configure_collision_sensor()
		self.configure_lane_invasion_sensor()

		# Wait for the car to be ready
		self.wait_for_car()  # Since the car falls from the sky when spawning            

		# Wait for the camera to be ready
		self.wait_for_camera()  # Camera is faster than the car, only needed if standalone

		if self.config.SPECATE:
			self.spectator = self.world.get_spectator() 
					
		# Start the NPCs
		for npcs in self.world.get_actors().filter('*vehicle.*'):
			self.tm.ignore_lights_percentage(npcs, 90)  # Ignore red lights
			if npcs.id != self.vehicle.id:
				npcs.set_autopilot(True, self.config.tm_port)
				
		self.episode_ticks = 0
		self.respawnen_wp = True
		
		self.prev_steering = 0.0
		self.prev_throttle = 0.0
		self.current_steering_angle = 0.0  # Aktueller Lenkwinkel in Carla-Einheiten (-1 bis 1)
		self.steering = 0.0  # Aktion des Agenten von -1 bis 1
		self.throttle = 0.125  # Aktion des Agenten von 0 bis 1

		self.prev_velo = 0.0
		self.prev_yaw = 0.0
		self.get_pref_weights()
		self.collision_cnt = 0
		self.points_reached = 0
		self.lane_invasion_cnt = 0
		self.collision_cnt_log = 0
		self.collision_cnt_env_log = 0
		self.collision_cnt_car_log = 0
		self.total_dist_to_center = 0
		self.speeding_cnt = 0
		self.current_index = 0
		self.prev_index = 0
		self.prev_location = self.vehicle.get_location()
		self.prev_acc_indivduals = 0
		self.distance_travelled = 0
		self.timeout_ticks = 0
		self.timeout_index = 0
		self.timeout = 0
		self.prev_acc_indivduals = 0

		self.measurements = []
		self.action_hist = np.zeros((3*4)).astype(np.float32)
		self.distances = []
		self.blocked = False
		self.reduced = False
		self.lane_invasion = False

		# Create a new axis at the end of self.front_camera to make it 4D
		self.get_cams()
		self.current_index, self.next_waypoint = CarlaEnvUtils.update_position_in_route(self.vehicle, self.route, self.current_index, self.config)
		self.prev_index = self.current_index
		if self.config.useAutopilot:
			self.vehicle.set_autopilot(True, self.config.tm_port)
		else:
			self.vehicle.set_autopilot(False, self.config.tm_port)
		
		expanded_camera = np.expand_dims(self.front_camera, axis=1)  # In the time dimension

		# Repeat the image along the new axis as needed
		self.camera_observation = np.tile(expanded_camera, (1, self.config.state_frames, 1, 1))

		# Set the start time of the episode and return the camera image
		self.episode_start = time.time()
		
		info["reset"] = "done"
		return self.get_observation(), info

	def get_cams(self):		
		# Important to configure cams, by initializing them (einkommentieren in setup) und anzahl channels in config richit setzen 

		# self.front_camera[:3,:,:] = self.rgb_camera		
		# self.front_camera[:3,:,:] = self.seg_rgb_camera
		# self.front_camera[3,:,:] = self.d_camera

		self.front_camera = self.seg_rgb_camera
		# self.front_camera = self.rgb_camera

	def destroy_all_actors(self):
		self.destroy_actors()

		# Alle noch lebenden Akteure in der Welt abrufen
		all_actors = self.world.get_actors()
		batch_destory = []

		# Durch alle Akteure iterieren und sie zerstören
		for actor in all_actors:
			if actor is not None and actor.is_alive :
				batch_destory.append(actor)

		response = self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in batch_destory])
		
		# Sicherstellen, dass der Simulator den Zustand nach dem Zerstören aktualisiert
		self.world.tick()

	def destroy_actors(self):
		batch_destory = []
		for sensor in self.sensor_list:
			if sensor is not None and sensor.is_alive:
				sensor.stop()  # Stoppt den Sensor-Listener
				batch_destory.append(sensor)
			elif sensor is not None:
				batch_destory.append(sensor)

		for actor in self.actor_list:
				if actor is not None:
					if actor.is_alive:
						batch_destory.append(actor)

		# Sicherstellen, dass Sensoren gestoppt werden
		self.world.tick()

		# Verwende Batch API, um alle Akteure und Sensoren in einem Schritt zu zerstören
		response = self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in batch_destory])
		# print("Response: ", response)

		# Sicherstellen, dass der Simulator den Zustand nach dem Zerstören aktualisiert
		self.world.tick()

		# Optionale Bereinigung von Listen, falls notwendig
		self.lane_invasion_hist = []
		self.collision_hist = []
		self.traffic_list = []
		self.actor_list = []
		self.sensor_list = []

	def reset_spawn_points(self):
		self.available_spawn_points = list(self.spawn_points)

	def filter_spawn_points(self):
		ego_location = self.spawn_point.location
		filtered_spawn_points = [point for point in self.spawn_points if self.distance(ego_location, point.location) <= self.max_spawn_dist]
		return filtered_spawn_points

	def distance(self, location1, location2):
		return math.sqrt((location1.x - location2.x)**2 + (location1.y - location2.y)**2 + (location1.z - location2.z)**2)
	
	def create_vehicle(self, spawnpoint):
		self.vehicle = None
		while (not self.available_spawn_points) or self.available_spawn_points == []:
			time.sleep(2)  
			self.destroy_actors()
			self.reset_spawn_points()
			print("Error spawn Point list")
		while self.vehicle is None:
			if self.available_spawn_points:

				# used only when evaluating interpolator
				if self.config.eval_spawnpoint > 0:
					self.spawn_point = self.available_spawn_points[self.config.eval_spawnpoint % len(self.available_spawn_points)]
					
				elif (self.config.evaluate and not self.config.showPolicy) or self.config.scenario:
					self.spawn_point = self.available_spawn_points[spawnpoint%len(self.available_spawn_points)] 
				else:
					# normal case with random spawn point
					self.spawn_point = random.choice(self.available_spawn_points)

				# spawn_index = self.available_spawn_points.index(self.spawn_point)
				# print("Selected spawn point index:", spawn_index)

				if not self.config.scenario :
					self.available_spawn_points.remove(self.spawn_point)
			
			self.vehicle = self.world.try_spawn_actor(self.vehicle_bp_ego, self.spawn_point)
		self.vehicle.set_simulate_physics(True)
		self.actor_list.append(self.vehicle)
		
	def create_traffic(self):
		number_of_vehicles = self.config.NUM_TRAFFIC_VEHICLES
		available_spawn_points = self.filter_spawn_points()  # Verwende gefilterte Spawn-Punkte
		
		# Liste für die Batch-Befehle zum Spawnen von Fahrzeugen
		batch = []

		# Fahrzeug-Blueprints und Spawn-Punkte für den Batch sammeln
		for x in range(number_of_vehicles):
			if available_spawn_points:
				self.rdm_spawn = random.choice(available_spawn_points)
				if not self.config.scenario:
					available_spawn_points.remove(self.rdm_spawn)
				
				# Wähle ein zufälliges Fahrzeug aus der Blueprint-Liste
				self.vehicle_bp = random.choice(self.bp_lib.filter(random.choice(self.car_liste)))
				
				# Füge das Spawn-Kommando für das Fahrzeug zur Batch-Liste hinzu
				batch.append(carla.command.SpawnActor(self.vehicle_bp, self.rdm_spawn))

		# Fahrzeuge im Batch spawnen
		responses = self.client.apply_batch_sync(batch, True)  # 'True' sorgt für synchrones Spawnen

		# Liste für die erfolgreich gespawnten NPC-Fahrzeuge
		npc_vehicles = []

		# Verarbeite die Antworten des Batch-Spawns
		for response in responses:
			if not response.error:
				npc = self.world.get_actor(response.actor_id)  # Hol das Fahrzeug mit der zurückgegebenen ID
				npc_vehicles.append(npc)  # Füge das Fahrzeug der Liste hinzu
				self.tm.auto_lane_change(npc, True)
				self.tm.set_global_distance_to_leading_vehicle(2.5)
				self.actor_list.append(npc)  # Füge das Fahrzeug zur Actor-Liste hinzu
				self.traffic_list.append(npc)  # Füge das Fahrzeug zur Verkehrsliste hinzu

		# Geschwindigkeitsanpassung
		rdm_speed = np.random.normal(self.config.target_speed_perc - 0.05, 0.2)
		npc_speed_perc = np.clip(rdm_speed, self.config.target_speed_perc - 0.2, self.config.target_speed_perc + 0.1)
		
		if self.config.evaluate:
			npc_speed_perc = self.config.target_speed_perc - 0.05
		
		self.tm.global_percentage_speed_difference((1 - npc_speed_perc) * 100)  # Geschwindigkeit anpassen
		
	def create_traffic_scenario(self,spawnpoints, stop=False):
		available_spawn_points = self.available_spawn_points # Use filtered spawn points
		blueprint = self.bp_lib.find('vehicle.ford.mustang') # set fix blueprint here for easier visualization
		for position in spawnpoints:
			self.npc = None
			if available_spawn_points:
				pos = int(position)%len(available_spawn_points)
				self.other_spawn_point = available_spawn_points[pos]
				self.npc = self.world.try_spawn_actor(blueprint, self.other_spawn_point)
				if self.npc is not None:
					self.tm.auto_lane_change(self.npc, True)
					self.tm.set_global_distance_to_leading_vehicle(2.5)
					self.actor_list.append(self.npc)
					self.traffic_list.append(self.npc)
		npc_speed_perc = self.config.target_speed_perc - 0.05
		self.tm.global_percentage_speed_difference((1 - npc_speed_perc) * 100)  # Geschwindigkeit anpassen
		if stop:
			self.tm.global_percentage_speed_difference(90) #stehen

	def get_image_from_queue(self):
		# Überprüfen, ob Bilder in der Queue vorhanden sind, bevor du sie verarbeitest
		# Nimm das nächste Bild aus der Queue und Verarbeite das Bild und speichere es in self.seg_rgb_camera
		if not self.seg_rgb_camera_queue.empty():
			self.process_seg_rgb(self.seg_rgb_camera_queue.get())

		if not self.rgb_camera_queue.empty():
			self.process_img(self.rgb_camera_queue.get())
		
		if not self.depth_camera_queue.empty():
			self.process_depth(self.depth_camera_queue.get())
		
		if not self.seg_camera_queue.empty():
			self.process_seg(self.seg_camera_queue.get())
		
		if not self.collision_queue.empty():
			self.collision_data(self.collision_queue.get())

		if not self.lane_invasion_queue.empty():
			self.lane_invasion_callback(self.lane_invasion_queue.get(), 'middle')

	def configure_camera(self):
		self.camera = self.world.spawn_actor(self.rgb_cam, self.cam_init_trans, attach_to=self.vehicle)
		self.sensor_list.append(self.camera)
		# self.camera.listen(lambda image: self.process_img(image))

		while not self.rgb_camera_queue.empty(): self.rgb_camera_queue.get()
		self.camera.listen(self.rgb_camera_queue.put)

	def configure_SegRGB(self):
		self.seg_rgb_camera = self.world.spawn_actor(self.seg_rgb_cam, self.cam_init_trans, attach_to=self.vehicle)
		self.sensor_list.append(self.seg_rgb_camera)
		# self.seg_rgb_camera.listen(lambda imageSRGB: self.process_seg_rgb(imageSRGB))

		while not self.seg_rgb_camera_queue.empty(): self.seg_rgb_camera_queue.get()
		self.seg_rgb_camera.listen(self.seg_rgb_camera_queue.put)
	
	def configure_segcamera(self):
		self.segcamera = self.world.spawn_actor(self.seg_cam, self.cam_init_trans, attach_to=self.vehicle)
		self.sensor_list.append(self.segcamera)
		# self.segcamera.listen(lambda imageS: self.process_seg(imageS))

		while not self.seg_camera_queue.empty(): self.seg_camera_queue.get()
		self.segcamera.listen(self.seg_camera_queue.put)
	
	def configure_depthcamera(self):
		self.depth_camera = self.world.spawn_actor(self.depth_cam_bp, self.depth_transform, attach_to=self.vehicle)
		self.sensor_list.append(self.depth_camera)
		# self.depth_camera.listen(lambda imageD: self.process_depth(imageD))

		while not self.depth_camera_queue.empty(): self.depth_camera_queue.get()
		self.depth_camera.listen(self.depth_camera_queue.put)

	def configure_collision_sensor(self):
		self.colsensor = self.world.spawn_actor(self.colsensor_bp, self.transform_sensors, attach_to=self.vehicle)
		self.sensor_list.append(self.colsensor)
		# self.colsensor.listen(lambda event: self.collision_data(event))

		while not self.collision_queue.empty(): self.collision_queue.get()
		self.colsensor.listen(self.collision_queue.put)

		# Lane invasion sensor configuration

	def configure_lane_invasion_sensor(self):
		self.lanesensor = self.world.spawn_actor(self.lanesensor_bp, self.transform_sensors, attach_to=self.vehicle) #transformation does not affeckt the sensor
		self.sensor_list.append(self.lanesensor)
		# self.lanesensor.listen(lambda invasion: self.lane_invasion_callback(invasion, 'middle'))

		while not self.lane_invasion_queue.empty(): self.lane_invasion_queue.get()
		self.lanesensor.listen(self.lane_invasion_queue.put)

	def get_offset_positions(self, main_vehicle_transform):
		vehicle_location = main_vehicle_transform.location
		vehicle_rotation = main_vehicle_transform.rotation

		# Calculate offsets relative to the vehicle's orientation
		offset_distance_y = 1.0  # Side offset relative
		offset_distance_x = 1.6  # Forward offset fixed

		# Convert yaw to radians
		yaw_rad = math.radians(vehicle_rotation.yaw)

		# Calculate new positions
		right_x = vehicle_location.x + offset_distance_x * math.cos(yaw_rad) - offset_distance_y * math.sin(yaw_rad)
		right_y = vehicle_location.y + offset_distance_x * math.sin(yaw_rad) + offset_distance_y * math.cos(yaw_rad)
		left_x = vehicle_location.x + offset_distance_x * math.cos(yaw_rad) + offset_distance_y * math.sin(yaw_rad)
		left_y = vehicle_location.y + offset_distance_x * math.sin(yaw_rad) - offset_distance_y * math.cos(yaw_rad)

		left_position = carla.Location(x=left_x, y=left_y, z=vehicle_location.z)
		right_position = carla.Location(x=right_x, y=right_y, z=vehicle_location.z)

		return left_position, right_position

	def check_lane_invasion(self):
		main_vehicle_transform = self.vehicle.get_transform()

		# Calculate positions to the left and right of the vehicle
		left_position, right_position = self.get_offset_positions(main_vehicle_transform)

		# vehicle_lane_id = vehicle_waypoint.lane_id if vehicle_waypoint else None
		# vehicle_road_id = vehicle_waypoint.road_id if vehicle_waypoint else None
		# vehicle_lane_type = vehicle_waypoint.lane_type if vehicle_waypoint else None
		
		# Get lane information at the positions
		left_lane_id, left_road_id, left_lane_type = self.get_waypoint_info(left_position)
		right_lane_id, right_road_id, right_lane_type = self.get_waypoint_info(right_position)

		if not left_road_id:
			self.lane_invasion_hist.append((left_lane_id, 'left'))

		if not right_road_id:
			self.lane_invasion_hist.append((right_lane_id, 'right'))

	def get_waypoint_info(self, location):
		waypoint = self.map.get_waypoint(location, project_to_road=False)
		if waypoint is not None:
			return waypoint.lane_id, waypoint.road_id, waypoint.lane_type
		return None, None, None

	def lane_invasion_callback(self, invasion, side):
		# Callback function that is called when a lane invasion is detected.
		self.lane_invasion_hist.append((invasion, side))
	
	def tick_the_world(self):
		self.world.tick()  # Ensure the world state is updated

		self.get_image_from_queue()
		self.check_lane_invasion()

	def wait_for_car(self):
		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1, steer=0.0))
		self.tick_the_world()
		self.check_lane_invasion()
		while self.vehicle.get_velocity().z != 0.0:
			self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.2, steer=0.0))
			self.tick_the_world()
			
	def wait_for_camera(self):
		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
		while np.all(self.d_camera == 0) and np.all(self.s_camera == 0) and np.all(self.rgb_camera == 0) and np.all(self.seg_rgb_camera == 0):
			print("d", self.d_camera.any())
			print("s", self.s_camera.any())
			print("r", self.rgb_camera.any())
			print("sr", self.seg_rgb_camera.any())
			self.tick_the_world()

	def process_img(self, image):
		array = np.frombuffer(image.raw_data, dtype=np.uint8)  # Array of BGRA 32-bit pixels
		# copied_array = array.copy()
		data = array.reshape((self.config.IM_HEIGHT, self.config.IM_WIDTH, 4))

		bgr_image = data[:, :, :3]  # Array of BGRA 32-bit pixels
		# Convert BGR to RGB
		rgb_image = bgr_image[:, :, ::-1]
		# Convert shape to [channels, height, width]
		rgb_image_t = np.transpose(rgb_image, (2, 0, 1))
		
		self.rgb_camera = rgb_image_t
		#imageio.imwrite('./img/saved_image.png', rgb_image)

	def process_seg_rgb(self, imageSRGB):
		# Konvertiert das Bild in das RGB-Format mit der CityScapes-Palette
		imageSRGB.convert(carla.ColorConverter.CityScapesPalette)
		
		# Rohdaten des Bildes als Array konvertieren
		array = np.frombuffer(imageSRGB.raw_data, dtype=np.uint8)
		
		# In Form eines Bildes mit den Dimensionen (Höhe, Breite, Channels) umformen
		data = array.reshape((self.config.IM_HEIGHT, self.config.IM_WIDTH, 4))  # BGRA format

		# Der letzte Channel (Alpha) ist nicht notwendig, daher ignorieren wir ihn
		bgr_image = data[:, :, :3]  # BGR Format, Alpha wird entfernt

		# Falls nötig, konvertieren Sie das Bild von BGR zu RGB
		rgb_image = bgr_image[:, :, ::-1]  # Konvertieren von BGR zu RGB

		# Konvertiert die Form zu [channels, height, width] für PyTorch-kompatiblen Tensor
		image_data = np.transpose(rgb_image, (2, 0, 1))
		
		# Speichern des resultierenden RGB-Bildes
		self.seg_rgb_camera = image_data

		# Optional: Bild speichern, falls Sie es überprüfen möchten
		# imageio.imwrite('./img/saved_image.png', rgb_image)

	def process_depth(self, imageD):
		# Convert CARLA image to a NumPy array of BGRA 32-bit pixels
		array = np.frombuffer(imageD.raw_data, dtype=np.uint8)
		copied_array = array.copy()
		data = copied_array.reshape((self.config.IM_HEIGHT, self.config.IM_WIDTH, 4))

		# Extract B, G, R channels; ignore alpha channel
		B = data[:, :, 0].astype(np.float32)
		G = data[:, :, 1].astype(np.float32)
		R = data[:, :, 2].astype(np.float32)
		
		# Calculate depth values in meters using the BGRA 32-bit pixel information
		depth = (R + G*256 + B*256*256) / (256**3 - 1) * 1000  # Scale to meters
		
		max_dist = 22
		# Clip all values greater than max_dist meters to 30 meters
		clipped_in_meters = np.clip(depth, 0, max_dist)
		
		# Normalize to the range 0 to 255
		inverted_depth_image_uint8 = np.uint8(((max_dist - clipped_in_meters) / max_dist) * 255)

		# Return the processed image
		self.d_camera = inverted_depth_image_uint8
		# imageio.imwrite('./img/saved_image.png', inverted_depth_image_uint8)

	def process_seg(self, imageS):
		array = np.frombuffer(imageS.raw_data, dtype=np.uint8)
		copied_array = array.copy()
		data = copied_array.reshape((self.config.IM_HEIGHT, self.config.IM_WIDTH, 4))  # BGRA format

		# Extract the red channel for semantic segmentation
		seg_image = data[:, :, 2]  # The red channel contains the tag information

		# Constants for masking
		road = 1
		sideWalk = 2
		cars = 14
		truck = 15
		bicycle = 19
		
		zeros = np.zeros_like(seg_image)

		# Compare with tolerance
		# Only use the mask if you want to initally easy the training, by allowing for some classes only

		road_mask = np.where(road == seg_image, seg_image, zeros)
		sidewalk_mask = np.where(sideWalk == seg_image, seg_image, zeros)
		cars_mask = np.where(cars == seg_image, seg_image, zeros)
		truck_mask = np.where(truck == seg_image, seg_image, zeros)
		bicycle_mask = np.where(bicycle == seg_image, seg_image, zeros)

		# Combine masks
		valid_classes = road_mask | cars_mask | truck_mask | bicycle_mask

		# Apply masks
		seg_image = np.where(valid_classes, seg_image, zeros)

		# Scale the image
		scaled_image = np.uint8(seg_image * 255.0)
		
		self.s_camera = scaled_image
		# imageio.imwrite('./img/saved_image.png', scaled_image)
		# e.g. non blocking view using sxiv 

	def collision_data(self, event):
		self.collision_hist.append(event)

	def get_pref_weights(self):
		"""
		Theses functions are kind of doubled in replay buffer...
		"""
		# Generate preference weights for a round
		weights = np.random.uniform(low=0.0, high=1.0, size=(self.config.num_rewards,)).astype(np.float32)
		weights = np.square(weights)

		# With a small probability, set one random weight to 1 and the rest to 0
		if random.random() < 0.01:
			weights = np.zeros_like(weights)
			random_index = random.randrange(self.config.num_rewards)
			weights[random_index] = 1.0
		
		# Calculate the sum of weights and add a small epsilon to prevent division by zero
		sum_weights = np.sum(weights).astype(np.float32)
		epsilon = 1e-8  # A small value to prevent division by zero
		if sum_weights < epsilon:
			normalized_weights = np.ones_like(weights) / len(weights)
		else:
			# Perform normalization ensuring the result remains float32 and avoiding division by zero
			normalized_weights = weights / sum_weights

		self.pref_weights_round = normalized_weights

	def get_pref_weights_step(self):
		"""
		Theses functions are kind of doubled in replay buffer...
		"""
		local_w = np.random.normal(loc=0, scale=0.05, size=(self.config.num_rewards,)).astype(np.float32)
		step_weights = self.pref_weights_round + local_w
		step_weights = np.round(step_weights, 4)
		step_weights = np.clip(step_weights, 0, 1).astype(np.float32)
		
		# Normalize step_weights
		normalized_step_weights = step_weights / np.sum(step_weights).astype(np.float32)
		
		return np.round(normalized_step_weights, 3)

	def get_observation(self):
		self.get_cams()

		# Verschiebe alle vorhandenen Beobachtungen eine Position nach hinten in der time dimension, 
		# nur notwendig wenn mehre timeslots verwenden werden, aktuell t = 1

		# Shift all existing observations one position backward in the time dimension
		# self.camera_observation[:,1:,:,:] = self.camera_observation[:,:-1,:,:]

		# kein time shift notwendig, einfach daten schreiben
		self.camera_observation[:, 0, :, :] = self.front_camera

		self.measurements = CarlaEnvUtils.get_vehicle_measurements(self.vehicle)

		pref_weights = self.get_pref_weights_step()
		
		# When you want to assume, that traffic lights information is provided
		# tf_light = self.get_traffic_light() #todo this function
		# signs = np.array([tf_light, speed_limit], dtype=np.float32)

		speed_limit = CarlaEnvUtils.get_speed_limit_ms(self.vehicle)
		signs = np.array([speed_limit], dtype=np.float32)

		if self.respawnen_wp:
			self.current_index, self.next_waypoint = CarlaEnvUtils.update_position_in_route(self.vehicle, self.route, self.current_index, self.config)
			self.prev_index = self.current_index

		# Angle, Distance only at intersections
		angle = 0.0
		distance = 0.0

		angle = np.round(CarlaEnvUtils.get_relative_direction_orientation(self.vehicle.get_transform(), self.next_waypoint),5)
		distance = self.vehicle.get_transform().location.distance(self.next_waypoint.transform.location)
		next_wp = np.round(np.array([angle, distance]).astype(np.float32),4)
	
		# print("angle", np.round(np.rad2deg(angle),2), f"distance: {distance:.2f}")
		# next_wp = np.round(np.array([angle]).astype(np.float32),4)

		# Get the closest car and its distance, for reward and safety computation
		self.distances, self.blocked, self.reduced = CarlaEnvUtils.closest_car(self.vehicle, self.traffic_list)

		# Write the observation into a dictionary
		observation = {
			'camera': self.camera_observation,
			'measurements': self.measurements,
			'n_action_hist': self.action_hist,
			'pref_weights': pref_weights,
			'signs': signs,
			'next_wp': next_wp,
		}

		return observation

	def apply_action(self, action, info):
		# Process and set the action
		action = CarlaEnvUtils.process_action(action)
		self.steering = float(action[0])
		self.throttle = float(action[1])

		# Flag to apply the autopilot action
		if self.config.useAutopilot or self.config.autopilotActions:
			self.vehicle.set_autopilot(True, self.config.tm_port)
			brake, self.steering, self.throttle = self.measurements[-3:]
			self.vehicle.apply_control(carla.VehicleControl(throttle=float(self.throttle), steer=float(self.steering)))
			info["steering"] = f"{self.steering:.3f}"
			info["throttle"] = f"{self.throttle:.3f}"
			info["brake"] = f"{brake:.3f}"

			self.current_steering_delta = 0 

		else: # Apply the control to the vehicle, by only allowing a small change in the steering angle
			
			# Calculate the desired steering change based on the agent's action
			self.current_steering_delta = self.config.max_steering_change * self.steering  # Max 3 degrees per tick (scale from -1 to 1 to degrees)
			new_steering_angle_in_degrees = self.current_steering_angle + self.current_steering_delta
			
			# Limit the new steering angle to the maximum range
			if new_steering_angle_in_degrees > self.config.max_steering_angle:
				new_steering_angle_in_degrees = self.config.max_steering_angle
			elif new_steering_angle_in_degrees < -self.config.max_steering_angle:
				new_steering_angle_in_degrees = -self.config.max_steering_angle
			

			# Update delta based on the limited change
			self.current_steering_delta = new_steering_angle_in_degrees - self.current_steering_angle #sinnvoll??
			self.current_steering_delta = np.round(np.clip(self.current_steering_delta, -self.config.max_steering_change, self.config.max_steering_change),5)
			
			# Update the current steering angle
			self.current_steering_angle = new_steering_angle_in_degrees
			
			# Scale back to -1 to 1 for Carla
			steering_in_carla_unit = self.current_steering_angle / self.config.max_steering_angle
			
			# Store the steering value in the info dictionary
			info["steering"] = f"{steering_in_carla_unit:.3f}"

			# Threshold for converting network output into brake action
			if self.throttle < 0.125:
				braking = 0.5 - self.throttle
				self.vehicle.apply_control(carla.VehicleControl(brake=braking, steer=steering_in_carla_unit))

				# Store the throttle value in the info dictionary
				info["throttle"] = 0
				info["brake"] = f"{braking:.3f}"
			else:
				# Apply the control to the vehicle
				scaled_throttle = np.clip((self.throttle), 0, 1)
				self.vehicle.apply_control(carla.VehicleControl(throttle=scaled_throttle, steer=steering_in_carla_unit))
				info["throttle"] = f"{scaled_throttle:.3f}"
				info["brake"] = 0
				self.throttle = scaled_throttle

				# Slightly smooth the actions?

		return info
	
	def step(self, action):
		done = False
		multi_reward = np.zeros(self.config.num_rewards + 1, dtype=np.float32)
		baseline_reward = 0.0
		info = {}
		self.prev_acc_indivduals = self.vehicle.get_acceleration()
		_, self.prev_acc_longitudinal = CarlaEnvUtils.calculate_lateral_longitudinal_acceleration(self.vehicle)

		self.prev_velo = self.measurements[0]
		self.prev_yaw = self.measurements[7]
		self.prev_location = self.vehicle.get_location()

		info = self.apply_action(action, info)

		# Set the spectator position to the vehicle's position, unterschiedliche Optionen für die Kameraposition
		if self.config.SPECATE:
			# vehicle_transform = self.vehicle.get_transform()
			# # ego view
			# self.transform = carla.Transform(vehicle_transform.transform(carla.Location(x=-6,z=3.5)), vehicle_transform.rotation)		
			
			# Set the BEV (Bird's Eye View) position 17 meters above the vehicle
			# self.transform = carla.Transform(vehicle_transform.transform(carla.Location(x=0,z=17)),  carla.Rotation(pitch=-90))

			# self.transform = carla.Transform(
			# 		carla.Location(x=132.5, y=11.2, z=29.5),
			# 		carla.Rotation(pitch=-90.0, yaw=0.0, roll=-90.0)
			# )

			self.transform = carla.Transform(
				carla.Location(x=-16.3, y=-0.8, z=44.9),
				carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
			)	

			self.spectator.set_transform(self.transform)
			
		# schreibe logging daten wenn evaluiert wird in die env info
		if self.config.evaluate:
			location = self.vehicle.get_transform().location
			speed_limit = CarlaEnvUtils.get_speed_limit_ms(self.vehicle)

			# Used for the preference evaluation
			current_vel = self.measurements[0]
			current_acc = self.measurements[3]
			info["current_vel"] = current_vel
			info["current_acc"] = current_acc

			# Evaluate speed limit adherence
			speed_limit = CarlaEnvUtils.get_speed_limit_ms(self.vehicle) * self.config.target_speed_perc
			speed_limit_adherence = 1 - (abs(speed_limit - current_vel) / speed_limit)
			info["speed"] = f"{speed_limit_adherence:.3f}"
			
			# BeV maps of pref evaluation
			info["pos_x"] = location.x
			info["pos_y"] = location.y
			info["pos_z"] = location.z
				
		# Tick the world in the same client for synchronous mode
		if self.config.showPolicy:
			current_time = time.time()
			elapsed_time = current_time - self.last_tick_time
			self.last_tick_time = time.time()
			
			if elapsed_time < self.tick_interval:
				time.sleep(self.tick_interval - elapsed_time)  # Wait until the interval is reached

		self.tick_the_world()
		self.current_index, self.next_waypoint = CarlaEnvUtils.update_position_in_route(self.vehicle, self.route, self.current_index, self.config)
		self.measurements = CarlaEnvUtils.get_vehicle_measurements(self.vehicle)
		self.episode_ticks += 1

		# fail safe
		if self.vehicle is None:
			done = True
		
		# calculate the rewards,
		else:
			# Lateral error and heading error
			feedback, info = CarlaEnvUtils.error_baseline(self.vehicle, self.next_waypoint, info)
			baseline_reward += feedback

			# Distance to lane center (bevor waypoint logic)
			feedback, distance = CarlaEnvUtils.distance_to_lane_center(self.vehicle, self.world, self.route, self.current_index)
			self.total_dist_to_center += distance
			baseline_reward += feedback

			# Waypoint logic
			feedback, done, info = CarlaEnvUtils.waypoint_logic(self.vehicle, self.measurements, self.route, self.respawnen_wp, self.current_index, self.prev_index, self.prev_location, distance, self.config, done, info)
			self.prev_index = self.current_index
			baseline_reward += feedback

			# Baseline reward regarding velocity
			feedback, info = CarlaEnvUtils.velocity_baseline(self.vehicle, self.measurements, self.steering, self.throttle, self.prev_steering, self.prev_throttle, self.blocked, self.config, info)
			baseline_reward += feedback
			
			# Lane invasion detected
			feedback, self.lane_invasion, info = CarlaEnvUtils.lane_invasion_baseline(self.lane_invasion_hist, self.lane_invasion, info)
			baseline_reward += feedback

			# Collision detected
			feedback, self.collision_cnt, done, info = CarlaEnvUtils.collision_baseline(self.collision_hist, self.measurements, self.collision_cnt, done, info)
			baseline_reward += feedback
			self.collision_hist = []

			# Off road
			feedback = CarlaEnvUtils.off_road(self.vehicle, self.map)
			baseline_reward += feedback

			# Vehicle distance
			feedback, info = CarlaEnvUtils.vehicle_dist(self.vehicle, self.measurements, self.traffic_list, self.prev_steering, self.steering, self.distances, self.blocked, self.reduced, info)
			baseline_reward += feedback

			baseline_reward = round(baseline_reward, 5)

			# reward for moving
			goal_speed = self.config.target_speed_perc

			normal = CarlaEnvUtils.speed_reward(self.vehicle, self.measurements, goal_speed, self.blocked, self.reduced)
			multi_reward[0] = baseline_reward + normal

			jerk_vec, jerk_magnitude = CarlaEnvUtils.get_jerk(self.config.fixed_delta_seconds, self.vehicle.get_acceleration(), self.prev_acc_indivduals)
			
			#######
			# Preference-Objectives-Rewards
			#######

			# Speed
			goal_speed = self.config.target_speed_perc + 0.2
			fast = CarlaEnvUtils.speed_reward(self.vehicle, self.measurements, goal_speed, self.blocked, self.reduced) * 0.5
			multi_reward[1] = fast 

			# Efficiency
			goal_speed = self.config.target_speed_perc - 0.1
			# slow = CarlaEnvUtils.speed_reward(self.vehicle, self.measurements, goal_speed, self.blocked, self.reduced)
			efficiency = CarlaEnvUtils.efficiency(self.vehicle, self.measurements, self.throttle)
			multi_reward[2] = efficiency 

			# Aggressiveness
			acc = CarlaEnvUtils.aggressiveness(self.vehicle, self.measurements, self.collision_cnt, self.prev_acc_longitudinal, done)
			multi_reward[3] = acc

			# Comfort
			comfort = CarlaEnvUtils.comfort(self.vehicle, self.measurements, self.prev_steering, self.prev_throttle, self.steering, self.throttle, self.prev_acc_longitudinal, self.prev_velo, self.prev_yaw, jerk_magnitude, self.config)
			multi_reward[4] = comfort

			self.respawnen_wp = False
			self.prev_steering, self.prev_throttle = self.steering, self.throttle
			next_state = self.get_observation()

			if self.episode_ticks + 1 > self.config.MAX_TICKS_PER_EPISODE:
				done = True

		# Update the action history in state space
		self.action_hist[4:] = self.action_hist[:-4] # shift all by 4 back
		self.action_hist[0:3] = np.round(self.measurements[-3:], 4) # brake, steer, throttle
		self.action_hist[3] = self.current_steering_delta
		
		# Check for scenario timeout
		self.timeout = float(0)
		if self.timeout_index != self.current_index:
			self.timeout_index = self.current_index
			self.timeout_ticks = 0
		else:
			self.timeout_ticks += 1
			# If no new waypoint is reached within one third of the episode length, timeout
			if self.timeout_ticks > int(self.config.MAX_TICKS_PER_EPISODE / 3) or self.timeout_ticks > 200:
				done = True
				self.timeout = float(1)

		# Add evaluation information and more logging calculations into info dictionary
		info["baseline_reward"] = f"{baseline_reward:.3f}"
		info["done"] = done
		info["current_wp_index"] = self.current_index

		if info["waypoint_logic_value"] == "reached_next_wp":
			self.points_reached += 1

		if info["collision"] == "collided":
			self.collision_cnt_log += 1

		if info["collision_type"] == "environment":
			self.collision_cnt_env_log += 1
		
		if info["collision_type"] == "vehicle":
			self.collision_cnt_car_log += 1

		if not info["lane_invasion"] == "":
			self.lane_invasion_cnt += 1

		if info["speeding"] == "over_limit":
			self.speeding_cnt += 1

		self.distance_travelled += CarlaEnvUtils.update_distance_travelled(self.vehicle.get_transform().location, self.prev_location)
		info["distance_travelled"] = float(self.distance_travelled)
		
		lateral_acceleration, longitudinal_acceleration = CarlaEnvUtils.calculate_lateral_longitudinal_acceleration(self.vehicle)
		info["lateral_acc"] = float(lateral_acceleration)
		info["longitudinal_acc"] = float(longitudinal_acceleration)
		
		# Log final results when episode finished
		if done:
			info["points_reached"] = float(self.points_reached)
			info["collision_cnt"] = float(self.collision_cnt_log)
			info["collision_cnt_env"] = float(self.collision_cnt_env_log)
			info["collision_cnt_car"] = float(self.collision_cnt_car_log)
			info["lane_invasion"] = float(self.lane_invasion_cnt)
			info["speeding_cnt"] = float(self.speeding_cnt)
			info["dist_to_center"] = float(self.total_dist_to_center)
			avg_dist = self.total_dist_to_center / (self.episode_ticks + 1)
			route_completion = self.current_index / (len(self.route) -1)
			info["route_completion"] = float(route_completion)
			info["episode_duration"] = float(self.episode_ticks)
			info["ep_steps"] = float(self.episode_ticks) #doppelt
			info["timeout"] = float(self.timeout)

			# driving_score calculation, Careful here not exactly the same calulation as done in carla leaderboard
			driving_score = np.round(1 *
				np.power(0.98, avg_dist) *
				np.power(0.65, self.collision_cnt_env_log) *
				np.power(0.75, self.collision_cnt_car_log) *
				np.power(0.995, self.lane_invasion_cnt) *
				np.power(0.95, self.speeding_cnt) *
				np.power(0.75, self.timeout) *
				
				np.power(route_completion, 1.2),
			2)
			driving_score = np.round(np.mean(driving_score),3)
			info["adjusted_driving_core"] = float(driving_score)
			
		# Stop the exploration with the autopilot
		if done:
			self.vehicle.set_autopilot(False, self.config.tm_port)
			self.config.useAutopilot = False
		
		if done and self.config.load_new_world:
			self.reload_map()

		multi_reward = np.round(np.clip(multi_reward, -10, 10), 5)
		# print("multi_reward", multi_reward)

		return next_state, multi_reward, done, False, info
