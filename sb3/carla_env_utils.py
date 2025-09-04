import random
import time
import math
import carla
import numpy as np

"""
This class provides functions for calculating driving rewards, including the core driving reward 
and preference-based objectives. It also contains utility functions for route generation, 
action processing, and measurement retrieval.
"""

class CarlaEnvUtils:

	def process_action(action):
		"""
		Processes the given action by adjusting the steering and throttle values.

		Args:
			action (np.array): Array containing steering and throttle values.

		Returns:
			np.array: Adjusted steering and throttle values in the range [-1, 1] for steering and [0, 1] for throttle.
		"""
		
		steering, throttle = action  # Extract steering and throttle values from the input action array

		# Clamp the steering value to the range [-1, 1]
		steering = np.clip(steering, -1, 1).astype(np.float32)

		# Adjust throttle to a range of [0, 1] (no brake allowed)
		throttle = np.clip(((throttle + 1) / 2), 0, 1).astype(np.float32)  # Scale throttle from [-1, 1] to [0, 1]

		return np.array([steering, throttle], dtype=np.float32)

	def get_speed_limit_ms(vehicle): 
		"""
		Returns the speed limit in meters per second (m/s).

		Args:
			vehicle (carla.Vehicle): The vehicle object.

		Returns:
			float: Speed limit in m/s, rounded to four decimal places.
		"""
		speed_limit = vehicle.get_speed_limit()  # Speed limit in km/h

		# Cap speed limit at 130 km/h
		if speed_limit > 130:  
			speed_limit = 130  # Correct the speed limit if it exceeds 130 km/h

		# Convert speed limit from km/h to m/s
		speed_limit = speed_limit / 3.6 + 0.0001  # Add a small value to avoid division by zero

		return np.round(speed_limit, 4).astype(np.float32)

	def calculate_lateral_longitudinal_acceleration(vehicle):
		"""
		Calculates the lateral and longitudinal acceleration of the vehicle.

		Args:
			vehicle (carla.Vehicle): The vehicle object.

		Returns:
			tuple: Lateral and longitudinal accelerations in the vehicle's local coordinate system.
		"""
		# Get the current acceleration and transformation of the vehicle
		acceleration = vehicle.get_acceleration()
		transform = vehicle.get_transform()

		# Extract vehicle's rotation (orientation)
		rotation = transform.rotation

		# Convert the acceleration to a NumPy array
		acceleration_vector = np.array([acceleration.x, acceleration.y, acceleration.z])

		# Compute direction cosine matrices for the vehicle's rotation
		roll = np.radians(rotation.roll)
		pitch = np.radians(rotation.pitch)
		yaw = np.radians(rotation.yaw)

		# Rotation matrix for yaw (around the Z-axis)
		R_yaw = np.array([
			[np.cos(yaw), -np.sin(yaw), 0],
			[np.sin(yaw), np.cos(yaw), 0],
			[0, 0, 1]
		])

		# Rotation matrix for pitch (around the Y-axis)
		R_pitch = np.array([
			[np.cos(pitch), 0, np.sin(pitch)],
			[0, 1, 0],
			[-np.sin(pitch), 0, np.cos(pitch)]
		])

		# Rotation matrix for roll (around the X-axis)
		R_roll = np.array([
			[1, 0, 0],
			[0, np.cos(roll), -np.sin(roll)],
			[0, np.sin(roll), np.cos(roll)]
		])

		# Combine the rotation matrices
		R = np.dot(np.dot(R_yaw, R_pitch), R_roll)

		# Transform the acceleration into the vehicle's local coordinate system
		local_acceleration = np.dot(R.T, acceleration_vector)

		# Extract lateral (Y) and longitudinal (X) accelerations
		lateral_acceleration = local_acceleration[1]  # Y-component (lateral)
		longitudinal_acceleration = local_acceleration[0]  # X-component (longitudinal)

		return lateral_acceleration, longitudinal_acceleration

	def calculate_lateral_longitudinal_velocity(vehicle):
		"""
		Calculates the lateral and longitudinal velocity of the vehicle.

		Args:
			vehicle (carla.Vehicle): The vehicle object.

		Returns:
			tuple: Lateral and longitudinal velocities in the vehicle's local coordinate system.
		"""
		# Get the current velocity and transformation of the vehicle
		velocity = vehicle.get_velocity()
		transform = vehicle.get_transform()

		# Extract vehicle's rotation (orientation)
		rotation = transform.rotation

		# Convert the velocity to a NumPy array
		velocity_vector = np.array([velocity.x, velocity.y, velocity.z])

		# Compute direction cosine matrices for the vehicle's rotation
		yaw = np.radians(rotation.yaw)
		pitch = np.radians(rotation.pitch)
		roll = np.radians(rotation.roll)

		# Rotation matrix for yaw (around the Z-axis)
		R_yaw = np.array([
			[np.cos(yaw), -np.sin(yaw), 0],
			[np.sin(yaw), np.cos(yaw), 0],
			[0, 0, 1]
		])

		# Rotation matrix for pitch (around the Y-axis)
		R_pitch = np.array([
			[np.cos(pitch), 0, np.sin(pitch)],
			[0, 1, 0],
			[-np.sin(pitch), 0, np.cos(pitch)]
		])

		# Rotation matrix for roll (around the X-axis)
		R_roll = np.array([
			[1, 0, 0],
			[0, np.cos(roll), -np.sin(roll)],
			[0, np.sin(roll), np.cos(roll)]
		])

		# Combine the rotation matrices
		R = np.dot(np.dot(R_yaw, R_pitch), R_roll)

		# Transform the velocity into the vehicle's local coordinate system
		local_velocity = np.dot(R.T, velocity_vector)

		# Extract lateral (Y) and longitudinal (X) velocities
		longitudinal_velocity = local_velocity[0]  # X-component (longitudinal)
		lateral_velocity = local_velocity[1]  # Y-component (lateral)

		return lateral_velocity, longitudinal_velocity

	def get_vehicle_measurements(vehicle):
		"""
		Retrieves and returns various vehicle measurements.

		Args:
			vehicle (carla.Vehicle): The vehicle object.

		Returns:
			np.array: A NumPy array containing the vehicle's speed, acceleration, angular velocity, and control inputs.
		"""
		# Get the velocity, acceleration, angular velocity, and control of the vehicle
		velocity = vehicle.get_velocity()
		acceleration = vehicle.get_acceleration()
		angular_velocity = vehicle.get_angular_velocity()
		control = vehicle.get_control()

		# Calculate speed magnitude in m/s
		speed_magnitude = np.float32(math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
		speed_magnitude = np.clip(speed_magnitude, 0, 130 / 3.6)

		# Calculate lateral and longitudinal velocities using the helper function
		lateral_velocity, longitudinal_velocity = CarlaEnvUtils.calculate_lateral_longitudinal_velocity(vehicle)
		lateral_velocity = np.clip(lateral_velocity, -130 / 3.6, 130 / 3.6)
		longitudinal_velocity = np.clip(longitudinal_velocity, -130 / 3.6, 130 / 3.6)  # Allowing negative longitudinal velocity

		# Calculate acceleration magnitude in m/s²
		acceleration_magnitude = math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)
		acceleration_magnitude = np.clip(acceleration_magnitude, 0, 200)

		# Calculate lateral and longitudinal acceleration
		lateral_acceleration, longitudinal_acceleration = CarlaEnvUtils.calculate_lateral_longitudinal_acceleration(vehicle)
		lateral_acceleration = np.clip(lateral_acceleration, -130 / 3.6, 130 / 3.6)
		longitudinal_acceleration = np.clip(longitudinal_acceleration, -130 / 3.6, 130 / 3.6)

		# Angular velocities (pitch and yaw rates)
		pitch_rate = np.deg2rad(angular_velocity.y)  # Rotation around Y-axis in rad/s
		yaw_rate = np.deg2rad(angular_velocity.z)  # Rotation around Z-axis in rad/s

		pitch_rate = np.clip(pitch_rate, -math.pi, math.pi)
		yaw_rate = np.clip(yaw_rate, -math.pi, math.pi)

		# Get control inputs (steering, throttle, brake)
		steer = control.steer
		throttle = control.throttle
		brake = control.brake
		# gear = control.gear
		# hand_brake = control.hand_brake
		# clutch = control.clutch
		# reverse = control.reverse
		
		# Construct the measurements array
		measurements = np.array([
			speed_magnitude,                #0: Speed magnitude (m/s)
			lateral_velocity,               #1: Lateral velocity (m/s)
			longitudinal_velocity,          #2: Longitudinal velocity (m/s)
			acceleration_magnitude,         #3: Acceleration magnitude (m/s²)
			lateral_acceleration,           #4: Lateral acceleration (m/s²)
			longitudinal_acceleration,      #5: Longitudinal acceleration (m/s²)
			pitch_rate,                     #6: Pitch rate (rad/s)
			yaw_rate,                       #7: Yaw rate (rad/s)
			brake,                          #8: Brake input (0-1)
			steer,                          #9: Steering input (-1 to 1)
			throttle,                       #10: Throttle input (0-1)
		], dtype=np.float32)

		# Print the measurements
		# print(f"Speed Magnitude: {speed_magnitude} m/s")
		# print(f"Lateral Velocity: {lateral_velocity} m/s")
		# print(f"Longitudinal Velocity: {longitudinal_velocity} m/s")
		# print(f"Acceleration Magnitude: {acceleration_magnitude} m/s²")
		# print(f"Lateral Acceleration: {lateral_acceleration} m/s²")
		# print(f"Longitudinal Acceleration: {longitudinal_acceleration} m/s²")
		# print(f"Pitch Rate: {pitch_rate} rad/s")
		# print(f"Yaw Rate: {yaw_rate} rad/s")
		# print(f"Brake: {brake}")
		# print(f"Steer: {steer}")
		# print(f"Throttle: {throttle}")

		# Return the measurements, rounded to five decimal places
		return np.round(measurements, 5)

	def get_jerk(time_step, current_acc, prev_acc):
		"""
		Calculates the jerk vector and its magnitude, which represents the rate of change of acceleration.

		Args:
			time_step (float): The time difference between the current and previous accelerations.
			current_acc (carla.Vector3D): The current acceleration vector of the vehicle.
			prev_acc (carla.Vector3D): The previous acceleration vector of the vehicle.

		Returns:
			tuple: The jerk vector and its magnitude.
		"""
		# Convert Vector3D objects into NumPy arrays for easier computation
		current_acc = np.array([current_acc.x, current_acc.y, current_acc.z])
		prev_acc = np.array([prev_acc.x, prev_acc.y, prev_acc.z])
		
		# Calculate the jerk vector as the change in acceleration divided by the time difference
		jerk_vector = (current_acc - prev_acc) / time_step
		
		# Calculate the magnitude (norm) of the jerk vector
		jerk_magnitude = np.linalg.norm(jerk_vector)
		
		return jerk_vector, jerk_magnitude
	
	def update_distance_travelled(current_location, prev_location):
		"""
		Computes the distance traveled by the vehicle between two locations.

		Args:
			current_location (carla.Location): The current location of the vehicle.
			prev_location (carla.Location): The previous location of the vehicle.

		Returns:
			float: The Euclidean distance between the two locations.
		"""
		# Convert Location objects into NumPy arrays
		current_location_array = np.array([current_location.x, current_location.y, current_location.z])
		prev_location_array = np.array([prev_location.x, prev_location.y, prev_location.z])
		
		# Calculate the Euclidean distance between the two points
		distance = np.linalg.norm(current_location_array - prev_location_array)
		
		return distance

	def transform_point_with_inverse_matrix(inv_matrix, point):
		"""
		Transforms a point from global coordinates to the local coordinate system using the inverse of a transformation matrix.

		Args:
			inv_matrix (np.array): The inverse transformation matrix.
			point (carla.Location): The point in global coordinates to be transformed.

		Returns:
			carla.Location: The transformed point in the vehicle's local coordinate system.
		"""
		# Convert the point to homogeneous coordinates (x, y, z, 1)
		point_homogeneous = np.array([point.x, point.y, point.z, 1])
		
		# Apply the inverse transformation matrix to the point
		transformed_point = np.dot(inv_matrix, point_homogeneous)
		
		# Return the transformed point in Cartesian coordinates as a carla.Location object
		return carla.Location(x=transformed_point[0], y=transformed_point[1], z=transformed_point[2])
	
	def get_relative_waypoint_position(vehicle, next_wp):
		"""
		Computes the position of the next waypoint relative to the vehicle's local coordinate system.

		Args:
			vehicle (carla.Vehicle): The vehicle object.
			next_wp (carla.Waypoint): The next waypoint.

		Returns:
			np.array: The relative position of the waypoint in the vehicle's local coordinates.
		"""
		# Get the global position of the waypoint
		wp_global_position = next_wp.transform.location

		# Get the vehicle's transform (position and orientation)
		vehicle_transform = vehicle.get_transform()
		inv_matrix = np.array(vehicle_transform.get_inverse_matrix())
		
		# Transform the waypoint's global position to the vehicle's local coordinate system
		wp_relative_position = CarlaEnvUtils.transform_point_with_inverse_matrix(inv_matrix, wp_global_position)

		# Sum of the x and y components of the relative position (could be useful for debugging or analysis)
		norm_sum = wp_relative_position.x + wp_relative_position.y

		# Return the relative position as a NumPy array
		position_array = np.array([wp_relative_position.x, wp_relative_position.y], dtype=np.float32)

		return position_array

	def xy_to_polar(next_wp):
		"""
		Converts the relative position of the next waypoint (x, y) to polar coordinates (angle, distance).

		Args:
			next_wp (np.array): The relative coordinates of the waypoint (x, y).

		Returns:
			np.array: A NumPy array containing the angle and distance in polar coordinates.
		"""
		try:
			# Calculate the distance (Euclidean norm)
			distance = np.linalg.norm(next_wp)
			
			# Calculate the angle in radians using arctan2
			angle = np.arctan2(next_wp[1], next_wp[0])
			
			# Return the angle and distance as a NumPy array
			polar_array = np.array([angle, distance], dtype=np.float32)

		except Exception as e:
			# In case of an error, return default values and print the error
			polar_array = np.array([0.0, 0.0], dtype=np.float32)

		return polar_array
	
	def get_relative_direction_orientation(transform, next_waypoint):
		"""
		Computes the relative yaw orientation (direction) between the vehicle and the next waypoint.

		Args:
			transform (carla.Transform): The current transform (orientation) of the vehicle.
			next_waypoint (carla.Waypoint): The next waypoint.

		Returns:
			float: The relative yaw angle in radians between the vehicle's orientation and the next waypoint's orientation.
		"""
		# Get the yaw values (orientation) of both the vehicle and the next waypoint
		current_yaw = transform.rotation.yaw
		next_yaw = next_waypoint.transform.rotation.yaw

		# Calculate the difference in yaw angles
		relative_yaw = next_yaw - current_yaw

		# Normalize the angle to be within the range [-180, 180]
		while relative_yaw > 180:
			relative_yaw -= 360
		while relative_yaw < -180:
			relative_yaw += 360

		# Return the relative yaw in radians
		return np.deg2rad(relative_yaw)

	def create_route(world, start_location, config, map):
		"""
		Generates a route of waypoints starting from a given location and returns the associated angles at intersections.

		Args:
			world (carla.World): The Carla world object.
			start_location (carla.Location): The starting location for the route.
			config (object): Configuration object containing route parameters (e.g., max waypoints, distance between waypoints).
			map (carla.Map): The Carla map object used to get waypoints.

		Returns:
			tuple: A tuple containing:
				- list(carla.Waypoint): List of waypoints in the route.
				- list(float): List of angles at intersections (0 for non-intersection points).
		"""
		successful_created_route = True

		# Get the starting waypoint based on the given start location
		waypoint_start = map.get_waypoint(start_location.location, project_to_road=True, lane_type=(carla.LaneType.Driving))
		route = [waypoint_start]  # Initialize the route with the starting waypoint
		current_waypoint = waypoint_start

		# Loop to generate a route with the specified number of waypoints
		for i in range(config.max_waypoints + config.target_wp_ahead):
			direction_wp = None
			# Get the next waypoints at the defined distance
			next_waypoints = current_waypoint.next(config.inter_wp_dist)
			if not next_waypoints:
				successful_created_route = False
				break  # No further waypoints available

			# If evaluation is enabled, use a deterministic random selection based on the seed
			if config.evaluate:
				random.seed(config.seed + i)

			# Select a random waypoint from the available next waypoints
			selected_next_waypoint = random.choice(next_waypoints)
			route.append(selected_next_waypoint)
			current_waypoint = selected_next_waypoint  # Move to the selected waypoint for the next iteration

		return route, successful_created_route

	def update_position_in_route(vehicle, route, current_index, config):
		"""
		Updates the current position of the vehicle in the route and determines the next target waypoint.

		Args:
			vehicle (carla.Vehicle): The vehicle object.
			route (list): The list of waypoints representing the route.
			current_index (int): The current index of the waypoint in the route.
			config (object): Configuration object with parameters like max waypoints and target waypoint ahead.

		Returns:
			tuple: The updated current index and the target waypoint the vehicle is heading towards.
		"""
		current_location = vehicle.get_location()

		# Update the current waypoint index based on the vehicle's location in the route
		while current_index < config.max_waypoints - 1 and (current_location.distance(route[current_index].transform.location) > current_location.distance(route[current_index + 1].transform.location)):
			current_index += 1  # Move to the next waypoint in the route if the vehicle is closer to it

		# Set the target waypoint to be ahead of the current waypoint, with a maximum index limit
		target_index = min(current_index + config.target_wp_ahead, len(route) - 1)
		current_target_wp = route[target_index]  # Get the target waypoint based on the updated index
		
		return current_index, current_target_wp

	def waypoint_logic(vehicle, measurements, route, respawnen_wp, current_index, prev_index, prev_location, distance_to_center, config, done, info):
		"""
		Logic for waypoint navigation and handling state changes in the environment.

		Args:
			vehicle (carla.Vehicle): The vehicle object.
			measurements (numpy.ndarray): Vehicle measurements (e.g., speed, acceleration).
			route (list): The list of waypoints defining the route.
			respawnen_wp (bool): Whether the vehicle needs to respawn at a waypoint.
			current_index (int): Current waypoint index in the route.
			prev_index (int): Previous waypoint index in the route.
			prev_location (carla.Location): Previous location of the vehicle.
			distance_to_center (float): Distance from the center of the route or track.
			config (object): Configuration parameters (e.g., maximum allowed deviation, track width).
			done (bool): Status whether the episode is done or not.
			info (dict): Additional information (e.g., feedback messages).

		Returns:
			tuple: A tuple containing:
				- feedback (float): The reward or penalty based on the current state.
				- done (bool): Whether the episode is completed.
				- info (dict): Updated information with the waypoint logic state.
		"""
		feedback = 0  # Initialize feedback value
		info["waypoint_logic_value"] = ""  # Initialize information key

		# Get the vehicle's current location
		current_location = vehicle.get_location()
		cloesest_wp = route[current_index + 1]  # Get the next waypoint
		distance_cloest_wp = current_location.distance(cloesest_wp.transform.location)  # Distance to the next waypoint

		# If the vehicle needs to respawn at a waypoint
		if respawnen_wp:
			info["waypoint_logic_value"] = "respawn"

		# If the vehicle reaches the last waypoint, mark the episode as done
		elif current_index == config.max_waypoints - 1:
			info["waypoint_logic_value"] = "goal"
			done = True
			feedback = 20  # High reward for reaching the goal

		# If the vehicle deviates too far from the route, penalize it
		elif distance_cloest_wp > config.max_route_deviation:
			done = True
			feedback = -5  # Negative reward for straying too far
			info["waypoint_logic_value"] = "over_maximum_wp_distance"

		# If the vehicle is within the track's width
		elif distance_to_center < config.track_width:
			current_location = vehicle.get_location()
			index = current_index + config.target_wp_ahead - 1
			target_location = route[index].transform.location
			prev_distance = prev_location.distance(target_location)
			current_distance = current_location.distance(target_location)
			progress = (prev_distance / current_distance) - 1  # Calculate progress towards the target waypoint
			feedback = np.clip(np.round(progress, 5), -1, 2)  # Feedback based on progress

			# If the vehicle has reached a new waypoint, update the feedback
			if not (prev_index == current_index):
				info["waypoint_logic_value"] = "reached_next_wp"
				current_vel = measurements[0]  # Vehicle's current speed
				scaling_factor = max(0.4, min(1.0, 0.4 + (1 / current_vel + 0.001)))  # Speed-based scaling
				feedback += 0.25 * scaling_factor  # Add some positive feedback based on speed
				
				# Add extra reward if the vehicle reaches a junction
				if cloesest_wp.is_junction:
					feedback += 0.1
			else:
				info["waypoint_logic_value"] = "progress"  # Update progress state

		else:  # Vehicle is outside of the track's width
			info["waypoint_logic_value"] = "within_distance_threshold"
			feedback = -0.35  # Penalize if the vehicle goes off the track

			# If the vehicle missed a waypoint, deduct more points
			if not (prev_index == current_index):
				feedback -= 1

		return feedback, done, info  # Return the feedback, done status, and info dictionary

	def off_road(vehicle, map):
		"""
		Penalizes the vehicle if it is off the road by giving a negative reward.

		Args:
			vehicle (carla.Vehicle): The vehicle object.
			map (carla.Map): The map object used to check the vehicle's location on the road.

		Returns:
			feedback (float): Negative reward if the vehicle is off the road.
		"""
		feedback = 0
		wp = map.get_waypoint(vehicle.get_location(), project_to_road=False)  # Get waypoint without projecting to the road
		if wp is None:  # If no valid waypoint exists (off-road)
			feedback -= 1.2  # Penalize the vehicle for being off-road
		return feedback  # Return the negative feedback

	def distance_to_lane_center(vehicle, world, route, current_index):
		"""
		Berechnet die Entfernung vom Fahrzeug zum Zentrum der Fahrbahn entlang der angegebenen Route.
		Stellt sicher, dass der Wegpunkt auf der richtigen Fahrspur liegt.

		Parameter:
		vehicle (carla.Vehicle): Das Fahrzeugobjekt.
		world (carla.World): Die CARLA-Welt.
		route (list): Liste der Wegpunkte, die die Route definieren.
		current_index (int): Der aktuelle Index des Wegpunkts, der dem Fahrzeug am nächsten liegt.

		Rückgabewerte:
		feedback (float): Feedback-Wert basierend auf der Entfernung zum Fahrbahnzentrum.
		distance (float): Die berechnete Entfernung zum Fahrbahnzentrum.
		"""
		feedback = 0
		route_match = False  # Flag, um anzuzeigen, ob der nächstgelegene Wegpunkt zur Route passt.

		# Fahrzeugposition holen
		vehicle_location = vehicle.get_location()

		# Finde den nächsten Wegpunkt auf der Route
		min_distance = float('inf')
		closest_index = current_index + 1

		# Durchlaufe die Wegpunkte, um den nächsten auf der Route zu finden
		for i in range(closest_index, len(route) - 1):
			wp_location = route[i].transform.location
			distance = vehicle_location.distance(wp_location)
			if distance < min_distance:
				min_distance = distance
				closest_index = i
			# Frühzeitiger Abbruch, wenn die Entfernung zunimmt
			if distance > min_distance + 8:
				break

		closest_waypoint_location_route = route[closest_index].transform.location

		# Finde den nächstgelegenen Wegpunkt auf der Straße
		closest_waypoint = world.get_map().get_waypoint(vehicle_location, project_to_road=True)

		# Überprüfe, ob der nächstgelegene Wegpunkt auf derselben Fahrspur liegt
		if closest_waypoint.lane_id != world.get_map().get_waypoint(closest_waypoint_location_route).lane_id:
			closest_waypoint = world.get_map().get_waypoint(closest_waypoint_location_route)
			route_match = True  # Markiere die Route als abgeglichen

		# Berechne die euklidische Entfernung zwischen Fahrzeug und Fahrbahnmitte
		lane_center_location = closest_waypoint.transform.location
		distance = math.sqrt(
			(vehicle_location.x - lane_center_location.x) ** 2 +
			(vehicle_location.y - lane_center_location.y) ** 2 +
			(vehicle_location.z - lane_center_location.z) ** 2
		)

		# Berechne Feedback basierend auf der Entfernung
		feedback -= np.clip(distance, 0, 2.5) * 0.15

		if route_match:
			feedback *= 0.7  # Wenn die Route übereinstimmt, wird das Feedback reduziert

		# Endgültiges Feedback
		feedback *= 0.7
		return feedback, distance

	def velocity_baseline(vehicle, measurements, steering, throttle, prev_steering, prev_throttle, blocked, config, info):
		"""
		Bewertet die Fahrzeugbewegung im Vergleich zum Geschwindigkeitslimit und den Steuerungsaktionen.

		Parameter:
		vehicle (carla.Vehicle): Das Fahrzeugobjekt.
		measurements (numpy.ndarray): Fahrzeugmessungen (z.B. Geschwindigkeit, Beschleunigung).
		steering (float): Aktueller Lenkwinkel.
		throttle (float): Aktueller Gaspedalwert.
		prev_steering (float): Vorheriger Lenkwinkel.
		prev_throttle (float): Vorheriger Gaspedalwert.
		blocked (bool): Flag, ob das Fahrzeug blockiert ist.
		config (object): Konfigurationsparameter.
		info (dict): Zusätzliche Informationen.

		Rückgabewerte:
		tuple: Ein Tuple bestehend aus:
			- feedback (float): Feedback-Wert basierend auf den Steuerungsaktionen und der Geschwindigkeit.
			- info (dict): Aktualisierte Informationen.
		"""
		feedback = 0

		# Holen des aktuellen Geschwindigkeitslimits
		speed_limit = CarlaEnvUtils.get_speed_limit_ms(vehicle)
		current_vel = measurements[0]
		info["speeding"] = ""  # Initialisiere das Feld "speeding" im Info-Dictionary

		# Vermeide Überschreitung des Geschwindigkeitslimits
		if current_vel > speed_limit:
			feedback -= (1 + 0.3 * (current_vel - speed_limit))  # Negative Belohnung für Geschwindigkeitsüberschreitung
			info["speeding"] = "over_limit"

		# Bewertung von Lenkwinkeländerungen
		if abs(prev_steering - steering) > 0.1:
			feedback -= (current_vel * 0.07 + 0.6) * abs(prev_steering - steering)

		# Bewertung von plötzlichen Gaspedaländerungen
		if abs(prev_throttle - throttle) > 0.2:
			feedback -= 0.4 * abs(prev_throttle - throttle)

		# Bewertung von Lenken bei nahezu Stillstand
		if current_vel == 0 and abs(steering) > 0.1:
			feedback -= 0.25

		# Bewertung von Vorzeichenänderungen beim Lenken (kann auf Schwingungen hinweisen)
		if steering * prev_steering < 0:
			feedback -= 0.2

		# Negatives Feedback bei Stillstand
		if current_vel == 0:
			feedback -= 0.4

		# Lenkwinkel > 0.05 erzeugt negative Belohnung basierend auf der Größe des Lenkwinkels
		if abs(steering) > 0.05:
			feedback -= abs(steering) * 0.05

		# Strafe für Fahrzeug, das wartet, obwohl es fahren könnte
		if current_vel == 0 and not blocked:
			feedback -= 3.5

		return feedback, info

	def lane_invasion_baseline(lane_invasion_hist, prev_invasion, info):
		"""
		Evaluates lane marking violations based on the history of lane invasions.

		Args:
			lane_invasion_hist (list): History of lane invasions.
			prev_invasion (bool): Indicates if there was a previous lane invasion.
			info (dict): Additional information to be updated.

		Returns:
			tuple: A tuple containing:
				- feedback (float): Feedback value based on lane invasions.
				- prev_invasion (bool): Updated previous invasion status.
				- info (dict): Updated information with lane invasion details.
		"""
		feedback = 0
		info["lane_invasion"] = ""  # Initialize lane invasion info
		
		# Set to track which sides of the lane have been processed
		processed_sides = set()
		markings_set = set()
		middle_invasion = False  # Flag to check if middle lane was invaded

		# Process the lane invasion history
		for invasion, side in lane_invasion_hist:
			if side not in processed_sides:
				if side == "middle":
					# Collect the types of crossed lane markings in the middle
					markings = [str(x.type) for x in invasion.crossed_lane_markings]
					markings_set.update(markings)
					middle_invasion = True
					processed_sides.add(side)
				elif side in ["right", "left"]:
					# Small penalty for crossing the left or right lane
					feedback -= 1.0
					processed_sides.add(side)

		# Clear the lane invasion history for future evaluations
		lane_invasion_hist.clear()

		# If middle lane was invaded or there was a previous invasion, apply penalty
		if middle_invasion or prev_invasion:
			info["lane_invasion"] = list(markings_set)  # Update the info with markings
			adj = 1.5 if 'Broken' in markings_set else 1  # Harder penalty for crossing broken lines
			feedback -= 2 * adj  # Apply penalty for lane invasion
			prev_invasion = not prev_invasion  # Toggle the previous invasion state

		return feedback, prev_invasion, info

	def collision_baseline(collision_hist, measurements, collision_cnt, done, info):
		"""
		Evaluates collisions and updates the state and information accordingly.

		Args:
			collision_hist (list): History of collisions.
			measurements (list): Vehicle measurements, including acceleration.
			collision_cnt (int): Count of recent collisions.
			done (bool): Status indicating if the episode is finished.
			info (dict): Additional information to be updated.

		Returns:
			tuple: A tuple containing:
				- feedback (float): Feedback value based on collision status.
				- collision_cnt (int): Updated collision count.
				- done (bool): Updated episode status.
				- info (dict): Updated collision-related information.
		"""
		feedback = 0
		info["collision"] = ""
		info["collision_type"] = ""

		# Check if a collision has been registered
		if collision_hist:
			info["collision"] = "collided"  # Set collision info
			collision_cnt += 1  # Increment the collision count
		elif collision_cnt > 0:
			collision_cnt -= 1  # Decrement the collision count if no new collision

		# Apply penalty if collisions occurred recently
		if collision_cnt:
			# If there have been too many collisions, reset the environment
			if collision_cnt == 3:
				done = True  # Set the episode as done
				collision_cnt = 0  # Reset collision count

			feedback -= 5  # Apply a general penalty for collisions
			current_acc = measurements[3]  # Get the current acceleration
			feedback -= current_acc * 0.1  # Apply additional penalty based on acceleration

			# Check the type of collision and apply penalties accordingly
			if collision_hist:
				event = collision_hist[0]
				if 'vehicle' in event.other_actor.type_id:
					info["collision_type"] = "vehicle"  # If a vehicle is involved
				else:
					# Environment collisions are penalized more heavily
					info["collision_type"] = "environment"
					feedback *= 1.7  # Increase penalty for environment collisions

		return feedback, collision_cnt, done, info

	def heading_errors(vehicle, waypoint):
		"""
		Computes the lateral and heading errors of the vehicle relative to a waypoint.

		Args:
			vehicle (carla.Vehicle): The vehicle actor.
			waypoint (carla.Waypoint): The target waypoint.

		Returns:
			tuple: A tuple containing:
				- lateral_error (float): The lateral error between the vehicle and the waypoint.
				- heading_error (float): The heading error between the vehicle and the waypoint.
		"""
		# Get the vehicle's transform (location and rotation)
		vehicle_transform = vehicle.get_transform()
		vehicle_location = vehicle_transform.location
		waypoint_location = waypoint.transform.location

		# Vector from the vehicle to the waypoint (ignoring the Z-axis)
		direction_vector = carla.Location(
			waypoint_location.x - vehicle_location.x,
			waypoint_location.y - vehicle_location.y,
			0  # Ignoring the Z-component as we're working in 2D
		)

		# Calculate lateral error: the perpendicular distance from the vehicle to the waypoint
		vehicle_forward_vector = vehicle_transform.get_forward_vector()
		lateral_error = math.sqrt(
			(direction_vector.y * vehicle_forward_vector.x - direction_vector.x * vehicle_forward_vector.y) ** 2
		)

		# Calculate heading error: the angle difference between the vehicle's heading and the waypoint's heading
		vehicle_yaw_rad = math.radians(vehicle_transform.rotation.yaw)
		waypoint_yaw_rad = math.radians(waypoint.transform.rotation.yaw)
		heading_error = math.degrees(
			math.atan2(math.sin(waypoint_yaw_rad - vehicle_yaw_rad),
					math.cos(waypoint_yaw_rad - vehicle_yaw_rad))
		)

		return lateral_error, heading_error
		
	def calculate_angle(vec1, vec2):
		"""
		Computes the angle in degrees between two vectors.

		Args:
			vec1 (carla.Location): The first vector.
			vec2 (carla.Location): The second vector.

		Returns:
			float: The angle in degrees between the two vectors.
		"""
		# Compute the dot product of the two vectors
		dot_product = vec1.x * vec2.x + vec1.y * vec2.y
		# Compute the magnitude (norm) of each vector
		norm_product = math.sqrt(vec1.x**2 + vec1.y**2) * math.sqrt(vec2.x**2 + vec2.y**2)
		# Compute the cosine of the angle
		cos_angle = dot_product / norm_product

		# Clip the cosine value to avoid numerical issues (e.g., due to floating point precision)
		if cos_angle is None:
			cos_angle = 1
		clipped_cos_angle = np.clip(cos_angle, -1, 1)

		# Calculate the angle in radians and convert it to degrees
		angle = math.acos(clipped_cos_angle)

		return math.degrees(angle)

	def closest_car(vehicle, traffic_list):
		"""
		Finds the closest car in the traffic list and determines if the vehicle is blocked or needs to reduce speed.

		Args:
			vehicle (carla.Vehicle): The vehicle actor.
			traffic_list (list): A list of other vehicle actors in the traffic.

		Returns:
			tuple: A tuple containing:
				- distances (list): A list of distances to the closest vehicles within the detection range.
				- blocked (bool): Whether the vehicle is blocked (i.e., collision is imminent).
				- reduced (bool): Whether the vehicle needs to reduce speed (i.e., another vehicle is in close proximity).
		"""
		blocked = False  # Flag to indicate if the vehicle is blocked (i.e., a collision is imminent)
		reduced = False  # Flag to indicate if the vehicle needs to reduce its velocity due to nearby cars
		detection_range = 10  # The range in which the vehicle detects other cars (in meters)

		# Get the current position and orientation (direction) of the vehicle
		my_pos = vehicle.get_location()
		my_transform = vehicle.get_transform()
		my_direction = carla.Vector3D(math.cos(math.radians(my_transform.rotation.yaw)), 
									math.sin(math.radians(my_transform.rotation.yaw)), 
									0)

		# Initialize variables for minimum distance tracking
		distances = []
		min_dist = 30
		min_dist2 = 30

		# If there are other vehicles in the traffic list, process each one
		if traffic_list:
			for car in traffic_list:
				# Get the position, transform, and velocity of the other car
				other_pos = car.get_location()
				other_transform = car.get_transform()
				velocity = car.get_velocity()
				speed_magnitude = np.float32(math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

				# Direction vector from the vehicle to the other car
				direction_to_car = carla.Vector3D(other_pos.x - my_pos.x, other_pos.y - my_pos.y, 0)

				# Direction vector of the other car
				other_direction = carla.Vector3D(math.cos(math.radians(other_transform.rotation.yaw)), 
												math.sin(math.radians(other_transform.rotation.yaw)), 
												0)

				# Calculate the angle to the other car and the angle between the vehicle's direction and the other car's direction
				angle_to_car = CarlaEnvUtils.calculate_angle(my_direction, direction_to_car)
				angle_between_directions = CarlaEnvUtils.calculate_angle(my_direction, other_direction)

				# Calculate the Euclidean distance to the other car
				dist = math.sqrt((my_pos.x - other_pos.x)**2 + (my_pos.y - other_pos.y)**2 + (my_pos.z - other_pos.z) ** 2)

				# Check if the other car is within the detection range and in a similar direction
				if (angle_to_car < 42 and dist < detection_range + 3.5 and angle_between_directions < 20):  
					# Car is in front and within the detection range (with some margin)
					distances.append(dist)
					reduced = True  # The vehicle should reduce speed because the car is close

				# If the vehicle is very close to another car, it is blocked and might collide soon
				if (angle_to_car < 18 and dist < detection_range):  
					# Car is too close in front of the vehicle and collision is imminent
					distances.append(dist)
					blocked = True  # Set blocked flag to True as a collision is imminent

		return distances, blocked, reduced

	def vehicle_dist(vehicle, measurements, traffic_list, prev_steering, steering, distances, blocked, reduced, info):
		"""
		Provides feedback based on the proximity of other vehicles, whether the vehicle is blocked, and if the speed needs to be reduced.

		Args:
			vehicle (carla.Vehicle): The vehicle actor.
			measurements (list): List of measurements including the current speed and brake status.
			traffic_list (list): A list of vehicles in the surrounding traffic.
			prev_steering (float): The previous steering angle of the vehicle.
			steering (float): The current steering angle of the vehicle.
			distances (list): List of distances to vehicles that may affect the vehicle's behavior.
			blocked (bool): Flag indicating whether the vehicle is blocked by another vehicle.
			reduced (bool): Flag indicating whether the vehicle should reduce speed.
			info (dict): Dictionary to store additional information, such as "car_ahead".

		Returns:
			tuple: A tuple containing:
				- feedback (float): The feedback value based on the vehicle's situation.
				- info (dict): The updated information dictionary with the "car_ahead" status.
		"""
		feedback = 0
		detection_range = 10  # The detection range for vehicles in front
		info["car_ahead"] = "None"  # Default value for the "car_ahead" status

		# If there are any vehicles detected in the front
		if distances:
			closest = min(distances)  # Get the closest vehicle
			brake = measurements[8]  # Brake status (0 if not braking, 1 if braking)
			current_vel = measurements[0]  # Current vehicle velocity

			# Case when the vehicle is blocked (potential collision situation)
			if blocked:
				if brake == 0:  # If not braking, potential crash situation
					info["car_ahead"] = "crashed"
					feedback = -2 * (1 - np.clip(closest / detection_range, 0, 1))  # Negative feedback for crash scenario
				else:  # If braking, but blocked, detected in the front
					info["car_ahead"] = "detected"
					feedback = 2.75  # Positive feedback for safe detection

				# When the vehicle is blocked but still moving forward
				if current_vel > 0.01:
					feedback *= 2.5  # Increase feedback to encourage stopping

			else:  # Case when the vehicle is not blocked but needs to reduce speed
				if brake == 0:  # If not braking, negative feedback for lack of response
					feedback = -1.5
				else:
					feedback = 1.5  # Positive feedback for braking action

				# Adjust feedback based on steering behavior (sharp steering or sudden changes)
				if abs(prev_steering - steering) > 0.1:
					feedback -= 0.15  # Negative feedback for sharp steering changes
				if abs(steering) > 0.05:
					feedback -= abs(steering) * 0.15  # More penalty for excessive steering

		return feedback, info

	def error_baseline(vehicle, next_waypoint, info):
		"""
		Evaluates the deviation of the vehicle from a waypoint and updates the information.

		Args:
			vehicle (carla.Vehicle): The vehicle object.
			next_waypoint (carla.Waypoint): The next waypoint to compare the vehicle's position with.
			info (dict): Dictionary to store additional information like heading and lateral errors.

		Returns:
			tuple: A tuple containing:
				- feedback (float): A feedback value based on the vehicle's deviation from the waypoint.
				- info (dict): The updated information dictionary with "heading_error" and "lateral_error".
		"""
		# Calculate lateral and heading errors using the CarlaEnvUtils.heading_errors method
		lateral_error, heading_error = CarlaEnvUtils.heading_errors(vehicle, next_waypoint)

		# Update the 'info' dictionary with the calculated errors
		info["heading_error"] = f"{heading_error:.3f}"  # Format heading error to 3 decimal places
		info["lateral_error"] = f"{lateral_error:.3f}"  # Format lateral error to 3 decimal places
		
		# Scale and limit the feedback values based on the errors
		# Feedback based on lateral error (maximum 3 meters deviation)
		feedback = np.clip(lateral_error, 0, 3) * -1 / 3 * 0.1

		# Feedback based on heading error (maximum 90 degrees deviation)
		feedback += np.clip(np.abs(heading_error), 0, 90) * -1 / 90 * 0.6

		# Return feedback value and updated information
		return feedback, info

	def speed_reward(vehicle, measurements, target_speed_perc, blocked, reduced):
		"""
		Calculates the speed reward based on the vehicle's current velocity, the target speed,
		and whether the vehicle is blocked or in a reduced-speed state.

		Args:
			vehicle (carla.Vehicle): The vehicle object.
			measurements (list): A list of measurements that includes the current velocity and brake status.
			target_speed_perc (float): The target speed percentage (e.g., 0.8 for 80% of speed limit).
			blocked (bool): Flag indicating whether the vehicle is blocked by another vehicle.
			reduced (bool): Flag indicating whether the vehicle is in a reduced-speed state (e.g., due to a nearby vehicle).

		Returns:
			float: A speed reward value based on the vehicle's behavior.
		"""
		# Initialize variables
		speed_reward = 0
		current_vel = measurements[0]  # Current velocity of the vehicle
		brake = measurements[8]        # Brake status (True/False)

		# Constants for speed reward calculation
		PENALTY_BLOCKED = -2.25  # Penalty when the vehicle is stopped (blocked)
		PENALTY_REDUCED = -1     # Penalty when the vehicle is in reduced speed state
		PENALTY_NORMAL = -1.75   # Penalty for driving at normal speed
		BRAKE_THRESHOLD = 0.8    # Brake penalty applied if speed is below 80% of the target speed
		REDUCED_SPEED_MULTIPLIER = 0.25  # Reduced speed multiplier
		NORMAL_SPEED_MULTIPLIER = 1.75   # Normal speed multiplier

		# Get the speed limit of the current road segment
		speed_limit = CarlaEnvUtils.get_speed_limit_ms(vehicle)

		# If the vehicle is blocked, apply a significant penalty based on how far it is from 0 m/s
		if blocked:
			# The penalty is higher the further the vehicle is from stopping (0 speed)
			speed_reward = (abs(0.0 - current_vel) / speed_limit) * PENALTY_BLOCKED
			return speed_reward

		# If the vehicle is in a reduced state (i.e., there is a car ahead), adjust the target speed accordingly
		if reduced:
			target_speed_perc *= REDUCED_SPEED_MULTIPLIER  # Scale the target speed down
			target_speed = target_speed_perc * speed_limit  # Calculate the target speed
			# Reward based on how close the vehicle is to the target speed in the reduced state
			speed_reward = (abs(target_speed - current_vel) / target_speed) * PENALTY_REDUCED + 1
			return speed_reward

		# Calculate the normal target speed and apply the speed reward formula for normal driving
		target_speed = target_speed_perc * speed_limit
		speed_reward = (abs(target_speed - current_vel) / target_speed) * PENALTY_NORMAL + NORMAL_SPEED_MULTIPLIER

		# Apply a penalty for unnecessary braking when the vehicle speed is less than 80% of the target
		if (not reduced) and (not blocked) and brake and (current_vel < target_speed * BRAKE_THRESHOLD):
			speed_reward -= 1  # Penalty for random braking

		return speed_reward

	def efficiency(vehicle, measurements, throttle):
		"""
		Evaluates the efficiency of the vehicle based on its current velocity, acceleration, and throttle input.

		Args:
			vehicle (carla.Vehicle): The vehicle object.
			measurements (list): A list of measurements including the current velocity and acceleration.
			throttle (float): Throttle input of the vehicle (value between 0 and 1).

		Returns:
			float: A scaled efficiency score between 0 and 1.
		"""
		# Extract vehicle measurements: current velocity (m/s) and acceleration (m/s^2)
		current_vel = measurements[0]  # Current velocity in m/s
		current_acc = measurements[3]  # Current acceleration (m/s^2)

		# Constants for efficiency calculation
		MAX_SPEED_KMH = 130  # Maximum speed in km/h
		MAX_SPEED_MS = MAX_SPEED_KMH / 3.6  # Maximum speed in m/s
		MAX_ACCELERATION = 200  # Maximum acceleration (used for normalization)
		MIN_THROTTLE_THRESHOLD = 0.125  # Minimum throttle value to calculate efficiency
		EFFICIENCY_SCALE_FACTOR = 12  # Scaling factor for efficiency calculation

		# Limit the current velocity and throttle within valid ranges
		current_vel = np.clip(current_vel, 0, MAX_SPEED_MS)  # Clip velocity to max speed
		throttle = np.clip(throttle, 0, 1)  # Clip throttle to range [0, 1]

		# Normalize velocity and acceleration for efficiency calculation
		normalized_acc = current_acc / MAX_ACCELERATION  # Normalize acceleration (scaled to max value)
		normalized_vel = current_vel / MAX_SPEED_MS  # Normalize velocity (scaled to max speed)

		# Initialize the efficiency score
		scaled_efficiency = 0

		# Calculate efficiency if throttle exceeds the minimum threshold
		if throttle >= MIN_THROTTLE_THRESHOLD:
			# Efficiency is higher when throttle is low and velocity and acceleration are within limits
			efficiency = (1.0 - throttle) * normalized_vel * (1 - normalized_acc)
			scaled_efficiency = np.round(np.clip(efficiency * EFFICIENCY_SCALE_FACTOR, 0, 1), 4)

		return scaled_efficiency

	def aggressiveness(vehicle, measurements, collision_cnt, prev_acc_longitudinal, done):
		"""
		Calculates the aggressiveness score based on the vehicle's measurements, including 
		longitudinal and lateral acceleration, yaw rate, and changes in acceleration.

		Args:
			vehicle (carla.Vehicle): The vehicle object.
			measurements (list): A list of measurements including acceleration and yaw rate.
			collision_cnt (int): The number of collisions the vehicle has encountered.
			prev_acc_longitudinal (float): The previous longitudinal acceleration.
			done (bool): Flag indicating whether the episode is finished.

		Returns:
			float: A score representing the vehicle's aggressiveness.
		"""
		# Initialize feedback score
		feedback = 0.0

		# Extract relevant measurements
		yaw = measurements[7]  # Yaw rate (angular velocity)
		lateral_acceleration = measurements[4]  # Lateral acceleration (side-to-side)
		longitudinal_acceleration = measurements[5]  # Longitudinal acceleration (forward/backward)

		# Constants for reward calculation
		MAX_ACCELERATION = 11  # Maximum value for acceleration (used for clipping)
		LONGITUDINAL_REWARD_FACTOR = 0.1  # Scaling factor for longitudinal acceleration reward
		LATERAL_REWARD_FACTOR = 0.1  # Scaling factor for lateral acceleration reward
		YAW_REWARD_FACTOR = 0.3  # Scaling factor for yaw reward
		MAX_YAW_RATE = 0.25  # Maximum yaw rate for reward calculation
		ACCELERATION_CHANGE_PENALTY_FACTOR = 0.02  # Penalty factor for changes in acceleration

		# Only calculate rewards if the vehicle is not done (episode not finished) and no collision occurred
		if not done and collision_cnt == 0:
			
			# Reward high longitudinal acceleration (positive and negative)
			longitudinal_reward = np.clip(np.abs(longitudinal_acceleration), 0, MAX_ACCELERATION) * LONGITUDINAL_REWARD_FACTOR
			feedback += longitudinal_reward

			# Reward high lateral acceleration
			lateral_reward = np.clip(np.abs(lateral_acceleration), 0, MAX_ACCELERATION) * LATERAL_REWARD_FACTOR
			feedback += lateral_reward

			# Reward fast changes in direction (yaw rate)
			angular_reward = np.clip(np.abs(yaw), 0, MAX_YAW_RATE) * YAW_REWARD_FACTOR
			feedback += angular_reward

			# Penalize frequent changes in longitudinal acceleration (sign change indicates direction reversal)
			if prev_acc_longitudinal * longitudinal_acceleration < 0:
				accel_change_penalty = np.clip(np.abs(prev_acc_longitudinal - longitudinal_acceleration), 0, MAX_ACCELERATION) * ACCELERATION_CHANGE_PENALTY_FACTOR
				feedback -= accel_change_penalty

		return feedback

	def comfort(vehicle, measurements, prev_steering, prev_throttle, steering, throttle, prev_acc_longitudinal, prev_velo, prev_yaw, jerk_magnitude, config):
		"""
		Rewards smooth changes in speed, acceleration, and steering, promoting a comfortable driving experience.

		Args:
			vehicle (carla.Vehicle): The vehicle object.
			measurements (list): A list of measurements that includes velocity, yaw, and acceleration.
			prev_steering (float): The steering angle from the previous step.
			prev_throttle (float): The throttle input from the previous step.
			steering (float): The current steering angle.
			throttle (float): The current throttle input.
			prev_acc_longitudinal (float): The previous longitudinal acceleration.
			prev_velo (float): The previous velocity of the vehicle.
			prev_yaw (float): The previous yaw rate (angular velocity).
			jerk_magnitude (float): The magnitude of the jerk (rate of change of acceleration).
			config (object): Configuration object that holds various parameters like `THROTTLE_CHANGE`.

		Returns:
			float: A score representing the comfort of the driving experience.
		"""
		# Initialize feedback score
		feedback = 0.0

		# Extract relevant measurements from the vehicle's current state
		current_vel = measurements[0]  # Current velocity of the vehicle (in m/s)
		yaw = measurements[7]  # Current yaw rate (angular velocity, in rad/s)
		longitudinal_acceleration = measurements[5]  # Current longitudinal acceleration (in m/s^2)

		# Constants for comfort reward calculation
		JERK_PENALTY_FACTOR = 0.01  # Factor to penalize jerk (rate of change of acceleration)
		STEERING_SMOOTHNESS_PENALTY = 0.1  # Penalty factor for sudden steering changes
		THROTTLE_SMOOTHNESS_PENALTY = 0.05  # Penalty for sudden throttle changes
		ACCELERATION_DIFF_PENALTY_FACTOR = 0.03  # Penalty for large differences in longitudinal acceleration
		VELOCITY_DIFF_PENALTY_LIMIT = 0.25  # Maximum penalty for speed changes (limit to avoid too much penalization)
		ANGULAR_VELOCITY_PENALTY_LIMIT = 0.3  # Limit for penalizing angular velocity (yaw)
		ANGULAR_VELOCITY_PENALTY_FACTOR = 0.3  # Factor for penalizing angular velocity (yaw)
		VELOCITY_PENALTY_FACTOR = 0.3  # Factor for penalizing velocity changes

		# Penalize sudden throttle changes
		if abs(prev_throttle - throttle) > 0.2:
			feedback -= THROTTLE_SMOOTHNESS_PENALTY

		# Penalize abrupt steering changes
		feedback -= STEERING_SMOOTHNESS_PENALTY * abs(prev_steering - steering)

		# Penalize large differences in velocity (abrupt speed changes)
		vel_diff_penalty = np.clip(abs(current_vel - prev_velo), 0, VELOCITY_DIFF_PENALTY_LIMIT) * VELOCITY_PENALTY_FACTOR
		feedback -= vel_diff_penalty

		# Penalize large changes in yaw rate (angular velocity)
		angular_velocity_penalty = np.clip(abs(prev_yaw - yaw), 0, ANGULAR_VELOCITY_PENALTY_LIMIT) * ANGULAR_VELOCITY_PENALTY_FACTOR
		feedback -= angular_velocity_penalty
		
		# Penalize large changes in longitudinal acceleration
		acc_diff_penalty = np.clip(abs(longitudinal_acceleration - prev_acc_longitudinal), 0, 5) * ACCELERATION_DIFF_PENALTY_FACTOR
		feedback -= acc_diff_penalty

		# Reward based on low jerk (smooth acceleration changes)
		feedback += 1.2 - np.clip(jerk_magnitude * JERK_PENALTY_FACTOR, 0, 1)

		return feedback
