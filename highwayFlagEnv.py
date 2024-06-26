import math 
import time 
import numpy as np 
import gymnasium as gym 
from gymnasium import spaces 
import carla 
import cv2

SECONDS_PER_EPISODE = 50 

N_CHANNELS = 3 
HEIGHT = 300 
WIDTH = 300 

FIXED_DELTA_SECONDS = 0.2 

SHOW_PREVIEW = True 

class HighwayFlagEnv(gym.Env):
    SHOW_CAM = SHOW_PREVIEW 
    imageWidth = WIDTH 
    imageHeight = HEIGHT 
    frontCamera = None 
    CAMERA_POS_Z = 1.3 
    CAMERA_POS_X = 1.4
    
    def __init__(self, world_name = "Town04"): 
        super(HighwayFlagEnv, self).__init__()
        
        # Two possible inputs for reverse (THROTTLE CONSTANT)
        self.action_space = spaces.Discrete(2, seed = 42)
        
        self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape = (HEIGHT, WIDTH, N_CHANNELS), dtype = np.uint8)
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(150.0)
        self.world = self.client.load_world(world_name)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        
        self.settings = self.world.get_settings() 
        self.settings.no_rendering_mode = True 
        self.settings.synchronous_mode = False 
        self.settings.fixed_deta_seconds = FIXED_DELTA_SECONDS 
        self.world.apply_settings(self.settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.model3 = self.blueprint_library.filter("model3")[0]
        
    def cleanup(self):
        for sensor in self.world.get_actors().filter("*sensor*"): 
            sensor.destroy() 
        for actor in self.world.get_actors().filter("*vehicle*"):
            actor.destroy()
            
    def render(self, mode = "human"): 
        pass 
    
    def reset(self, seed = None): 
        self.collision_hist = [] 
        self.actor_list = [] 
        self.lane_invasion_hist = [] 
        truncated = False 
        
        self.spawn_index = 350 
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = None 
        self.flag_location = carla.Location(x = -300.865570, y=33.573753, z=0.281942)
        
        
        while self.vehicle is None: 
            try:
                self.vehicle = self.world.try_spawn_actor(self.model3, self.spawn_points[self.spawn_index])
            except: 
                pass 
            
        self.actor_list.append(self.vehicle)
        self.initial_location = self.vehicle.get_location()
        self.spawn_location = self.spawn_points[self.spawn_index].location 
        
        self.semantic_camera = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        self.semantic_camera.set_attribute("image_size_x", f"{self.imageWidth}")
        self.semantic_camera.set_attribute("image_size_y", f"{self.imageHeight}")
        self.semantic_camera.set_attribute("fov", f"90")
        
        camera_init_trans = carla.Transform(carla.Location(z = self.CAMERA_POS_Z, x= self.CAMERA_POS_X))
        self.sensor = self.world.spawn_actor(self.semantic_camera, camera_init_trans, attach_to = self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        time.sleep(2)
        
        # Showing camera at the spawn point 
        if self.SHOW_CAM: 
            cv2.namedWindow("Initial Spawn Location", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Initial Spawn Location", self.frontCamera)
            cv2.waitKey(1)
            
        # Adding collision sensor 
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to = self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        
        # TODO: SHOULD I ADD A LANE INVASION SENSOR? 
        # lane_sensor = self.blueprint_library.find("sensor.other.lane_invasion")
        # lane_trans = carla.Transform(carla.Location(z = 0, x = 0, y = 0))
        # self.lane_sensor = self.world.try_spawn_actor(lane_sensor, lane_trans, attach_to = self.vehicle)
        # self.lane_sensor.listen(lambda event: self.lane_invasion_data(event))
        
        
        
        while self.frontCamera is None: 
            time.sleep(0.01)
        
        self.episode_start = time.time() 
        self.flag_collected = False 
        self.step_counter = 0 
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        
        info = {} 
        return self.frontCamera / 255.0, info 
        
    def step(self, action): 
        self.step_counter += 1 
        self.vehicle_location = self.vehicle.get_location()
        self.distance_from_spawn = self.vehicle_location.distance(self.spawn_location)
        self.distance_from_flag = self.vehicle_location.distance(self.flag_location)
        
        reverse = action
        
        
        THROTTLE = 0.5
        
        # Mapping VEHICLE CONTROL 
        if reverse == 0: 
            self.vehicle.apply_control(carla.VehicleControl(throttle = THROTTLE, reverse = False))
        elif reverse == 1: 
            self.vehicle.apply_control(carla.VehicleControl(throttle = THROTTLE, reverse = True))

        v = self.vehicle.get_velocity() # Getting the velocity of the vehicle 
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        distance_travelled = self.spawn_location.distance(self.vehicle.get_location())
        
        # Printing throttle and reverse every 100 steps 
        if self.step_counter % 100 == 0: 
            print("Distance:", int(distance_travelled), "Velocity:", kmh, "Throttle:", THROTTLE, "Reverse: ", reverse, "Flag collected:", self.flag_collected, 
                  "\nDistance from SPAWN:", int(self.distance_from_spawn), "Distance from FLAG:", int(self.distance_from_flag))
            
        camera = self.frontCamera 
        
        if self.SHOW_CAM: 
            cv2.imshow("Camera Footage", camera)
            cv2.waitKey(1)
            
        # Rewards 
        reward = 0 
        terminated = False 
        truncated = False 
        
        # Punishing for being too slow 
        if kmh < 10: 
            reward -= 1 
        elif kmh > 10:
            reward += 1
        elif kmh > 20: 
            reward += 2
        
        # Punishing for colliding
        if len(self.collision_hist) != 0: 
            terminated = True 
            reward = reward - 100 
            self.cleanup() 
        
        if self.distance_from_flag >= 0 and self.distance_from_spawn >= 65: 
            self.flag_collected = True 
        
        # IF THE FLAG IS NOT COLLECTED    
        if not self.flag_collected: 
            # Reward for getting closer to the the flag 
            if self.distance_from_flag < 65: 
                reward += 1 
            elif self.distance_from_flag < int(65 / 2): 
                reward += 2 
            elif self.distance_from_flag < 30: 
                reward += 3
            elif self.distance_from_flag < 10: 
                reward += 4 
            # PUNISH HEAVILY FOR GETTING AWAY FROM THE FLAG 
            elif self.distance_from_spawn > 65: 
                reward -= 10
            elif self.distance_from_flag == 0: 
                self.flag_collected = True 
                reward += 100 
        
        # IF THE FLAG IS COLLECTED 
        elif self.flag_collected: 
            # Reward for getting closer to the spawn location 
            if self.distance_from_spawn < 65: 
                reward += 2 
            elif self.distance_from_spawn < int(65 / 2): 
                reward += 3 
            elif self.distance_from_spawn < 30: 
                reward += 4 
            elif self.distance_from_spawn < 10: 
                reward += 5
            elif self.distance_from_spawn == 0: 
                reward += 200 
                terminated = True 
            # Punish for getting further from the spawn distance 
            if self.distance_from_flag >= 65 or self.distance_from_spawn > 65: 
                reward -= 10
                
        # Check for episode duration 
        if self.episode_start + SECONDS_PER_EPISODE < time.time(): 
            terminated = True 
            self.cleanup()
            
        observation = camera / 255.0 
        
        return observation, reward, terminated, truncated, {} 
    
    def process_img(self, image): 
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.imageHeight, self.imageWidth, 4))[:, :, :3] 
        self.frontCamera = i
        
    def collision_data(self, event): 
        self.collision_hist.append(event)
        
    # TODO: THINK ABOUT LANE INVASION! DO YOU NEED IT? 
    def lane_invasion_data(self, event): 
        self.lane_invasion_hist.append(event)
    
    def seed(self, seed): 
        pass
               