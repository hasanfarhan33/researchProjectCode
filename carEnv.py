import random 
import time 
import numpy as np 
import math 
import gym 
from gym import spaces
import carla 
import cv2

SECONDS_PER_EPISODE = 30 

N_CHANNELS = 3 
HEIGHT = 300 
WIDTH = 300 

FIXED_DELTA_SECONDS = 0.2 

SHOW_PREVIEW = True 

class CarEnv(gym.Env):
    SHOW_CAM = SHOW_PREVIEW 
    STEER_AMT = 1.0
    imageWidth = WIDTH 
    imageHeight = HEIGHT 
    frontCamera = None 
    CAMERA_POS_Z = 1.3 
    CAMERA_POS_X = 1.4 
    
    def __init__(self, world_name = "Town05"):
        super(CarEnv, self).__init__() 
        # 9 possible inputs for steering, and 4 inputs for braking and throttle 
        self.action_space = spaces.MultiDiscrete([9, 4])
        self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype = np.float32)
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(150.0)
        self.client.load_world(world_name)
        self.world = self.client.get_world() 
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        
        self.settings = self.world.get_settings() 
        self.settings.no_rendering_mode = True 
        self.settings.synchronous_mode = False 
        self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS 
        self.world.apply_settings(self.settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.model3 = self.blueprint_library.filter("model3")[0]
        
    def cleanup(self):
        for sensor in self.world.get_actors().filter("*sensor*"):
            sensor.destroy() 
            
        for actor in self.world.get_actors().filter("*vehicle*"): 
            actor.destroy() 
 
 
    def reset(self): 
        self.collision_hist = [] 
        self.actor_list = [] 
        self.randomSpawn = False 
        
        if self.randomSpawn: 
            self.transform = random.choice(self.world.get_map().get_spawn_points()) 
            self.vehicle = None 
            while self.vehicle is None: 
                try:
                    self.vehicle = self.world.spawn_actor(self.model3, self.transform)
                except:
                    pass 
        else:
            self.spawn_index = 176
            self.spawn_points = self.world.get_map().get_spawn_points() 
            self.vehicle = None 
            while self.vehicle is None: 
                try: 
                    self.vehicle = self.world.spawn_actor(self.model3, self.spawn_points[self.spawn_index])
                except:
                    pass 
        
        self.actor_list.append(self.vehicle)
        self.initial_location = self.vehicle.get_location() 
        
        
        # TODO: CHANGE TO NORMAL CAMERA
        self.semantic_camera = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        self.semantic_camera.set_attribute("image_size_x", f"{self.imageWidth}")        
        self.semantic_camera.set_attribute("image_size_y", f"{self.imageHeight}")        
        self.semantic_camera.set_attribute("fov", f"90")
        
        camera_init_trans = carla.Transform(carla.Location(z = self.CAMERA_POS_Z, x = self.CAMERA_POS_X))
        self.sensor = self.world.spawn_actor(self.semantic_camera, camera_init_trans, attach_to = self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        time.sleep(2)
        
        # Showing camera at the spawn point 
        if self.SHOW_CAM: 
            cv2.namedWindow("Semantic Camera Footage", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Semantic Camera Footage", self.frontCamera)
            cv2.waitKey(1)
        
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to = self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        
        while self.frontCamera is None: 
            time.sleep(0.01)
            
        self.episode_start = time.time()
        self.steering_lock = False 
        self.steering_lock_start = None 
        self.step_counter = 0 
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        
        return self.frontCamera/255.0
            
    def step(self, action): 
        self.step_counter += 1 
        steer = action[0] 
        throttle = action[1] 
        
        # Mapping steering actions 
        if steer == 0: 
            steer = -0.9 
        elif steer == 1: 
            steer = -0.25 
        elif steer == 2: 
            steer = -0.1 
        elif steer == 3: 
            steer = 0.05
        elif steer == 4: 
            steer = 0
        elif steer == 5: 
            steer = 0.05 
        elif steer == 6: 
            steer = 0.1 
        elif steer == 7:
            steer == 0.25 
        elif steer == 8: 
            steer = 0.9 
            
        # Mapping throttle 
        if throttle == 0: 
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, steer = steer, brake = 1.0))
        elif throttle == 1: 
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0.3, steer = steer, brake = 0.0))
        elif throttle == 2: 
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0.7, steer = steer, brake = 0.0))
        elif throttle == 3: 
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer = steer, brake = 0.0))
            
        # Printing steer and throttle every 50 steps 
        if self.step_counter % 50 == 0: 
            print("Steer input from model: ", steer, "Throttle: ", throttle)
            
        
        v = self.vehicle.get_velocity() 
        kmh = int(3.6 * math.sqrt(v.x**2, v.y**2, v.z**2))
        
        distance_travelled = self.initial_location.distance(self.vehicle.get_location())
        
        # Storing camera to return at the end in case the clean-up function destroys it 
        cam = self.font_camera 
        
        if self.SHOW_CAM: 
            cv2.imshow("Semantic Camera", cam)
            cv2.waitKey(1)
            
        # Track steering lock duration to prevent tail chasing 
        lockDuration = 0 
        if self.steering_lock == False: 
            if steer <-0.6 or steer > 0.6: 
                self.steering_lock = True 
                self.steering_lock_start = time.time() 
                
        else:
            if steer < -0.6 or steer > 0.6: 
                lockDuration = time.time() - self.steering_lock_start
                
        # Rewards 
        reward = 0 
        done = False 
        
        # Punish for collision 
        if len(self.collision_hist) != 0: 
            done = True 
            reward = reward - 300 
            self.cleanup() 
            
        # Punish for locking wheels 
        if lockDuration > 3: 
            reward = reward - 150 
            done = True 
            self.cleanup() 
        elif lockDuration > 1: 
            reward = reward - 20 
        
        # Reward for acceleration 
        if kmh < 10: 
            reward = reward - 3
        elif kmh < 15: 
            reward = reward - 1
        elif kmh > 40: 
            reward = reward  - 10 # Punish for going too fast 
        else: 
            reward = reward + 1 
            
        # Reward for making distance 
        if distance_travelled < 30: 
            reward = reward - 1
        elif distance_travelled < 50: 
            reward = reward + 1
        else: 
            reward = reward + 2 
            
        # Check for episode duration 
        if self.episode_start + SECONDS_PER_EPISODE < time.time(): 
            done = True 
            self.cleanup() 
        
        return cam/255.0, reward, done, {} 
    
    def process_img(self, image): 
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.imageHeight, self.imageWidth, 5))[:, :, :3] 
        self.frontCamera = i 
        
    def collision_data(self, event): 
        self.collision_hist.append(event)
        
    def seed(self, seed):
        if not seed:
            seed = 7
        random.seed(seed)
        self._np_random = np.random.RandomState(seed) 
        return seed