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

FIRST_PED_LOCATION = carla.Transform(carla.Location(x=-325.865570, y=33.573753, z=0.281942), carla.Rotation(yaw = 180))
SECOND_PED_LOCATION = carla.Transform(carla.Location(x=-300.865570, y=30.573753, z=0.781942), carla.Rotation(yaw = 180))
THIRD_PED_LOCATION = carla.Transform(carla.Location(x = -275.865570, y=36.573753, z = 1.781942), carla.Rotation(yaw = 180))
FOURTH_PED_LOCATION = carla.Transform(carla.Location(x = -250.865570, y=26.573753, z = 2.781942), carla.Rotation(yaw = 180))

PUNISHMENT_VALUE = 0 

class HighwayEnvMultiPedestrian(gym.Env): 
    SHOW_CAM = SHOW_PREVIEW 
    imageWidth = WIDTH 
    imageHeight = HEIGHT 
    frontCamera = None 
    CAMERA_POS_Z = 1.3 
    CAMERA_POS_X = 1.4
    
    def __init__(self, world_name = "Town04"): 
        super(HighwayEnvMultiPedestrian, self).__init__() 
        
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
        for actor in self.world.get_actors().filter("*pedestrian*"): 
            actor.destroy()
                
    def render(self, mode = "human"): 
        pass 
        
    def reset(self, seed = None): 
        self.collision_hist = []
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
        
        camera_init_trans = carla.Transform(carla.Location(z = self.CAMERA_POS_Z, x = self.CAMERA_POS_X))
        self.sensor = self.world.spawn_actor(self.semantic_camera, camera_init_trans, attach_to = self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        time.sleep(2)
        
        if self.SHOW_CAM: 
            cv2.namedWindow("Initial Spawn Location", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Front Camera", self.frontCamera)
            cv2.waitKey(1)
            
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to = self.vehicle)
        self.colsensor.listen(lambda event: self.collision_data(event))
        
        while self.frontCamera is None: 
            time.sleep(0.01)
        
        self.firstPedestrian = None 
        self.secondPedestrian = None 
        self.thirdPedestrian = None 
        self.fourthPedestrian = None 
        
        firstPed = self.blueprint_library.find("walker.pedestrian.0030")
        secondPed = self.blueprint_library.find("walker.pedestrian.0032")
        thirdPed = self.blueprint_library.find("walker.pedestrian.0002") 
        fourthPed = self.blueprint_library.find("walker.pedestrian.0034") 
        
        while self.firstPedestrian is None: 
            try: 
                self.firstPedestrian = self.world.try_spawn_actor(firstPed, FIRST_PED_LOCATION)
            except: 
                pass 
        
        while self.secondPedestrian is None: 
            try: 
                self.secondPedestrian = self.world.try_spawn_actor(secondPed, SECOND_PED_LOCATION)
            except:
                pass
        
        while self.thirdPedestrian is None: 
            try: 
                self.thirdPedestrian = self.world.try_spawn_actor(thirdPed, THIRD_PED_LOCATION)
            except:
                pass
        
        while self.fourthPedestrian is None: 
            try: 
                self.fourthPedestrian = self.world.try_spawn_actor(fourthPed, FOURTH_PED_LOCATION)
            except:
                pass 
            
        self.actor_list.append(self.firstPedestrian)
        self.actor_list.append(self.secondPedestrian)
        self.actor_list.append(self.thirdPedestrian)
        self.actor_list.append(self.fourthPedestrian)
        
        self.episode_start = time.time() 
        self.step_counter = 0 
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        
        info = {} 
        return self.frontCamera / 255.0, info 
        
    def step(self, action): 
        self.step_counter += 1 
        steer = action[0] 
        
        THROTTLE = 0.3 
        
        if steer == 0: 
            steer = -0.05 
            self.vehicle.apply_control(carla.VehicleControl(throttle = THROTTLE, steer = float(steer)))
        elif steer == 1: 
            steer = 0.0 
            self.vehicle.apply_control(carla.VehicleControl(throttle = THROTTLE, steer = float(steer)))
        elif steer == 2: 
            steer = 0.05 
            self.vehicle.apply_control(carla.VehicleControl(throttle = THROTTLE, steer = float(steer)))
            
        # Getting distance between vehicle and pedestrian 
        vehicle_location = self.vehicle.get_location()
        firstPedLocation = self.firstPedestrian.get_location()
        secondPedLocation = self.secondPedestrian.get_location()
        thirdPedLocation = self.thirdPedestrian.get_location()
        fourthPedLocation = self.fourthPedestrian.get_location()
        
        
        firstPedVehicleDistance = int(vehicle_location.distance(firstPedLocation))
        secondPedVehicleDistance = int(vehicle_location.distance(secondPedLocation))
        thirdPedVehicleDistance = int(vehicle_location.distance(thirdPedLocation))
        fourthPedVehicleDistance = int(vehicle_location.distance(fourthPedLocation))
        
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        distance_travelled = self.spawn_location.distance(self.vehicle.get_location())
        
        # Printing steer, throttle and brake every 100 steps 
        if self.step_counter % 100 == 0: 
            print("Steer: ", steer, "Distance from SPAWN: ", int(distance_travelled), "Velocity: ", kmh, "Throttle: ", THROTTLE, "\nFirst Ped Distance: ", firstPedVehicleDistance, "Second Ped Distance: ", secondPedVehicleDistance, "Third Ped Distance: ", thirdPedVehicleDistance, "Fourth Ped Distance: ", fourthPedVehicleDistance)
        
        camera = self.frontCamera 
        
        if self.SHOW_CAM: 
            cv2.imshow("Camera Footage", camera)
            cv2.waitKey(1)
        
        # REWARDS 
        #TODO: COME BACK TO FIGURE OUT THE REWARDING SYSTEM 
        reward = 0 
        terminated = False 
        truncated = False 
        
        if len(self.collision_hist) != 0: 
            print("PUNISHMENT VALUE: ", PUNISHMENT_VALUE)
            terminated = True 
            reward = reward - PUNISHMENT_VALUE
            self.cleanup()
            
        if firstPedVehicleDistance > 40: 
            reward += 1 
        if secondPedVehicleDistance > 65: 
            reward += 2
        if thirdPedVehicleDistance > 90: 
            reward += 3 
        if fourthPedVehicleDistance > 115:
            reward += 4 
            
            
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
        actor_we_collide_against = event.other_actor.type_id
        global PUNISHMENT_VALUE
        if "pedestrian" in actor_we_collide_against: 
            print("HIT A PEDESTERIAN. HE DEAD ~~")
            PUNISHMENT_VALUE = 200
            self.collision_hist.append(event)
        else:
            # print("hit something else") 
            PUNISHMENT_VALUE = 100
            self.collision_hist.append(event)