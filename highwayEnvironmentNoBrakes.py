import math
import random 
import time 
import numpy as np 
import gymnasium as gym 
from gymnasium import spaces 
import carla 
import cv2 

SECONDS_PER_EPISODE = 30 

N_CHANNELS = 3 
HEIGHT = 300 
WIDTH = 300 

FIXED_DELTA_SECONDS = 0.2 

SHOW_PREVIEW = True 

class HighwayEnvironmentNoBrakes(gym.Env): 
    SHOW_CAM = SHOW_PREVIEW 
    STEER_AMT = 1.0 
    imageWidth = WIDTH 
    imageHeight = HEIGHT 
    frontCamera = None 
    CAMERA_POS_Z = 1.3 
    CAMERA_POS_X = 1.4 
    
    def __init__(self, world_name = "Town04"): 
        super(HighwayEnvironmentNoBrakes, self).__init__() 
        
        # 3 possible inputs for steering, 2 for throttle
        # self.action_space = spaces.MultiDiscrete([3, 2])
        
        # 3 possible inputs for steering (THROTTLE CONSTANT)
        self.action_space = spaces.MultiDiscrete([3])
        
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
        self.lane_invasion_hist = [] 
        self.actor_list = [] 
        truncated = False 
        
        self.spawn_index = 350
        self.spawn_points = self.world.get_map().get_spawn_points() 
        self.vehicle = None 
        while self.vehicle is None: 
            try: 
                self.vehicle = self.world.spawn_actor(self.model3, self.spawn_points[self.spawn_index])
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
        self.colsensor.listen(lambda event: self.collision_data(event))
        
        # Adding lane invasion sensor 
        lane_sensor = self.blueprint_library.find("sensor.other.lane_invasion")
        lane_trans = carla.Transform(carla.Location(z = 0, x = 0, y = 0))
        self.lane_sensor = self.world.spawn_actor(lane_sensor, lane_trans, attach_to = self.vehicle)
        self.lane_sensor.listen(lambda event: self.lane_invasion_data(event))
        
        while self.frontCamera is None: 
            time.sleep(0.01)
            
        self.episode_start = time.time()
        # self.steering_lock = False 
        # self.steering_lock_start = None 
        self.step_counter = 0 
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        
        info = {}
        return self.frontCamera / 255.0, info 
    
    # Function to maintain the speed of the vehicle 
    def maintain_speed(self, s, preferred_speed, speed_threshold): 
        if s >= preferred_speed: 
            return 0 
        elif s < preferred_speed - speed_threshold: 
            return 0.7 
        else: 
            return 0.5 
    
        
    def step(self, action): 
        self.step_counter += 1
        steer = action[0] 
        
        # COMMENT THIS OUT IF YOU WANT TO KEEP THROTTLE CONSTANT
        # throttle = action[1] 
        # THROTTLE = 0.5
        v = self.vehicle.get_velocity() 
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        estimated_throttle = self.maintain_speed(kmh, 60, 50)
        
        # ACKERMANN SPEEDS
        EASY_SPEED = 5.55556 # 20 kph 
        MEDIUM_SPEED= 11.1111 # 40 kph
        HARD_SPEED = 22.2222 # 80 kph
        
        # MAPPING VEHICLECONTROL
        # Mapping steering actions
        if steer == 0:
            steer = -0.025
            self.vehicle.apply_control(carla.VehicleControl(throttle = estimated_throttle, steer = float(steer)))
        elif steer == 1: 
            steer = 0.0 
            self.vehicle.apply_control(carla.VehicleControl(throttle = estimated_throttle, steer = float(steer)))
        elif steer == 2: 
            steer = 0.025
            self.vehicle.apply_control(carla.VehicleControl(throttle = estimated_throttle, steer = float(steer)))

        
        distance_travelled = self.spawn_location.distance(self.vehicle.get_location())
        
        # Printing steer, throttle and brake every 50 steps 
        if self.step_counter % 50 == 0: 
            print("Steer: ", steer, "Distance: ", int(distance_travelled), "Velocity: ", kmh, "Throttle: ", estimated_throttle)          
            # print("Steer: ", steer, "Distance: ", int(distance_travelled), "Velocity: ", kmh)          
        camera = self.frontCamera 
        
        if self.SHOW_CAM: 
            cv2.imshow("Camera Footage", camera)
            cv2.waitKey(1)
            
            
        # Track steering lock duration to prevent tail chasing 
        # lockDuration = 0
        # if self.steering_lock == False: 
        #     if steer < -0.6 or steer > 0.6:
        #         self.steering_lock = True 
        #         self.steering_lock_start = time.time() 
                
        #     else: 
        #         if steer < -0.6 or steer > 0.6:
        #             lockDuration = time.time() - self.steering_lock_start
        
        # Rewards 
        reward = 0 
        terminated = False 
        truncated = False 
        
        # Punish for collision and lane invasion
        if len(self.collision_hist)!= 0 or len(self.lane_invasion_hist) != 0: 
            terminated = True 
            reward = reward - 200 
            self.cleanup()
        
        # if len(self.collision_hist) != 0: 
        #     reward = reward - 100
        #     terminated = True 
        #     self.cleanup()  
            
        # Reward for making distance
        EASY_DISTANCE = 50 
        MEDIUM_DISTANCE = 100 
        HARD_DISTANCE = 200
        
        # SIMPLE REWARD SYSTEM 
        # if int(distance_travelled) < 10: 
        #     reward = reward - 1
        # elif int(distance_travelled) < 30: 
        #     reward = reward - 1 
        # elif int(distance_travelled) >= EASY_DISTANCE and int(distance_travelled) < MEDIUM_DISTANCE: 
        #     if(len(self.lane_invasion_hist) != 0):
        #         reward = (reward + EASY_DISTANCE) // len(self.lane_invasion_hist)
        #     else:
        #         reward = reward + EASY_DISTANCE
        #     print("REACHED EASY DISTANCE")
        # # elif distance_travelled == EASY_DISTANCE + 25: 
        # #     reward = reward + 60
        # elif int(distance_travelled) >= MEDIUM_DISTANCE and int(distance_travelled) < HARD_DISTANCE: 
        #     if(len(self.lane_invasion_hist) != 0):
        #         reward = (reward + MEDIUM_DISTANCE) // len(self.lane_invasion_hist)
        #     else: 
        #         reward = reward + MEDIUM_DISTANCE
        #     print("REACHED MEDIUM DISTANCE") 
        # elif int(distance_travelled) >= HARD_DISTANCE:
        #     if(len(self.lane_invasion_hist) != 0): 
        #         reward = (reward + HARD_DISTANCE) // len(self.lane_invasion_hist)
        #     else: 
        #         reward = reward + HARD_DISTANCE
        #     print("REACHED HARD DISTANCE")
        
        # HARSH REWARD SYSTEM 
        if int(distance_travelled) < 30: 
            reward = reward - 1 
        elif int(distance_travelled) >= EASY_DISTANCE and int(distance_travelled) < MEDIUM_DISTANCE: 
            print("REACHED EASY DISTANCE")
            reward = reward + EASY_DISTANCE 
        elif int(distance_travelled) >= MEDIUM_DISTANCE and int(distance_travelled) < HARD_DISTANCE: 
            print("REACHED MEDIUM DISTANCE")
            reward = reward + MEDIUM_DISTANCE
        elif int(distance_travelled) >= HARD_DISTANCE: 
            print("REACHED HARD DISTANCE")
            reward = reward + HARD_DISTANCE 
        
        
            
        # Check for episode duration 
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            terminated = True 
            self.cleanup() 
            
        observation = camera/255.0 
        
        return observation, reward, terminated, truncated, {} 
    
    def process_img(self, image): 
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.imageHeight, self.imageWidth, 4))[:, :, :3] 
        self.frontCamera = i 
        
    def collision_data(self, event):
        self.collision_hist.append(event)
        
    def lane_invasion_data(self, event): 
        self.lane_invasion_hist.append(event)
        
    def seed(self, seed): 
        pass 