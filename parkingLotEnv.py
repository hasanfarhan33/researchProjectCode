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

# Location of all the corners 
BOTTOM_LEFT_PL = carla.Transform(carla.Location(x = 15, y = -12.5, z = 2))
TOP_LEFT_PL = carla.Transform(carla.Location(x = -35, y = -12.5, z = 2))
TOP_RIGHT_PL = carla.Transform(carla.Location(x = -35, y = -47, z = 2))
BOTTOM_RIGHT_PL = carla.Transform(carla.Location(x = 15, y = -47, z = 2))

npcVehicleLocOne = carla.Transform(carla.Location(x= -40, y= -25.75, z= 4), carla.Rotation(yaw = -90))
npcVehicleLocTwo = carla.Transform(carla.Location(x = -40, y= -29.5, z = 4), carla.Rotation(yaw = -90)) 
npcVehicleLocThree = carla.Transform(carla.Location(x = -40, y = -33.5, z = 4), carla.Rotation(yaw = -90))

npcVehicleLocFour = carla.Transform(carla.Location(x= 20, y= -32.75, z= 4), carla.Rotation(yaw = -90))
npcVehicleLocFive = carla.Transform(carla.Location(x= 20, y= -26.5, z= 4), carla.Rotation(yaw = -90))

class ParkingLotEnv(gym.Env): 
    SHOW_CAM = SHOW_PREVIEW 
    imageWidth = WIDTH 
    imageHeight = HEIGHT 
    frontCamera = None 
    CAMERA_POS_Z = 1.3 
    CAMERA_POS_X = 1.4 
    
    def __init__(self, world_name = "Town05_Opt"): 
        super(ParkingLotEnv, self).__init__() 
        
        self.action_space = spaces.MultiDiscrete([3])
        
        self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape = (HEIGHT, WIDTH, N_CHANNELS), dtype = np.uint8)
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(150.0)
        self.world = self.client.load_world(world_name)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        
        # Removing the junk 
        self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        # self.world.unload_map_layer(carla.MapLayer.Walls)
        
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
            
    def render(self, mode = "human"): 
        pass 
    
    def maintain_speed(self, s, preferred_speed, speed_threshold): 
        if s >= preferred_speed: 
            return 0 
        elif s < preferred_speed - speed_threshold: 
            return 0.2 
        else: 
            return 0.1 
    
    def spawn_npc(self, blueprint_lib, vehicle_id, location): 
        init_loc = location
        vehicle_bp = blueprint_lib.find(vehicle_id)
        npc_vehicle = self.world.try_spawn_actor(vehicle_bp, init_loc)
        if npc_vehicle is not None: 
            self.actor_list.append(npc_vehicle)
            # print("Vehicle added")
        else: 
            return 
    
    def reset(self, seed = None): 
        self.collision_hist = [] 
        self.actor_list = [] 
        
        truncated = False 
        
        self.spawn_location = carla.Transform(carla.Location(x = 15, y = -12.5, z = 2), carla.Rotation(yaw = 180))
        self.vehicle = None
        while self.vehicle is None: 
            try:
                self.vehicle = self.world.spawn_actor(self.model3, self.spawn_location)
            except: 
                pass 
            
        self.actor_list.append(self.vehicle)
        
        # Adding NPC Vehicles 
        self.spawn_npc(self.blueprint_library, "vehicle.mini.cooper_s", npcVehicleLocOne)
        self.spawn_npc(self.blueprint_library, "vehicle.mini.cooper_s", npcVehicleLocTwo)
        self.spawn_npc(self.blueprint_library, "vehicle.mini.cooper_s", npcVehicleLocThree)
        self.spawn_npc(self.blueprint_library, "vehicle.mini.cooper_s", npcVehicleLocFour)
        self.spawn_npc(self.blueprint_library, "vehicle.mini.cooper_s", npcVehicleLocFive)
        
        
        self.initial_location = self.vehicle.get_location()

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
            cv2.namedWindow("Initial Spawn Location", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Initial Spawn Location", self.frontCamera)
            cv2.waitKey(1)
            
        # Adding a collision sensor 
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to = self.vehicle)
        self.colsensor.listen(lambda event: self.collision_data(event))
        
        while self.frontCamera is None: 
            time.sleep(0.01)
            
        self.episode_start = time.time() 
        self.step_counter = 0 
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        
        info = {} 
        return self.frontCamera / 255.0, info 
    
    def step(self, action): 
        self.step_counter += 1 
        steer = action[0]
        
        # TODO: FIGURE OUT PROPER STEERING VALUES
        if steer == 0: 
            steer = 0 
        elif steer == 1: 
            steer = 0.3 
        elif steer == 2: 
            steer = -0.3
            
        v = self.vehicle.get_velocity() 
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        estimated_throttle = self.maintain_speed(kmh, 2, 1)
        
        # Map throttle and apply steer and throttle 
        self.vehicle.apply_control(carla.VehicleControl(throttle = estimated_throttle, steer = steer, brake = 0.0))
        
        # distance_travelled = self.spawn_location.distance(self.vehicle.get_location())
            
        camera = self.frontCamera
        if self.SHOW_CAM: 
            cv2.imshow("Camera Footage", camera)
            cv2.waitKey(1)
            
        # Rewards 
        reward = 0
        terminated = False 
        truncated = False 
        
        # Punish for colliding
        if len(self.collision_hist) != 0: 
            terminated = True 
            reward -= 100 
            self.cleanup()
        
        maxReward = 400
        BL_Bool = False 
        TL_Bool = False 
        TR_Bool = False 
        BR_Bool = False 
        
        cur_location = self.vehicle.get_location()
        
        distance_from_TL = int(cur_location.distance(TOP_LEFT_PL.location))
        distance_from_TR = int(cur_location.distance(TOP_RIGHT_PL.location))
        distance_from_BR = int(cur_location.distance(BOTTOM_RIGHT_PL.location))
        distance_from_BL = int(cur_location.distance(BOTTOM_LEFT_PL.location))
        
        # OLD REWARD SYSTEM
        # PART ONE
        if TL_Bool == False and TR_Bool == False and BR_Bool == False: 
            # print("In PART ONE")
            if distance_from_TL > 50: 
                reward -= 5 
            elif distance_from_TL < 50: 
                reward += 5 
            # Make sure the car doesn't reach other points before TL 
            elif int(distance_from_TR) == 4 or int(distance_from_BR) == 4: 
                reward -= 50 
                terminated = True
                self.cleanup() 
            elif int(distance_from_TL) == 4: 
                reward += maxReward // 4 
                TL_Bool = True   
        
        # PART TWO 
        elif TL_Bool == True and TR_Bool == False: 
            # print("In PART TWO")
            if distance_from_TR > 34: 
                reward -= 10 
            elif distance_from_TR < 34: 
                reward += 10
            # Make sure the car doesn't reach other points before TR 
            elif int(distance_from_BR) == 4 or int(distance_from_BL) == 4: 
                reward -= 50 
                terminated = True
                self.cleanup() 
            elif int(distance_from_TR) == 4: 
                reward += maxReward // 3 
                TR_Bool = True 
        
        # PART THREE 
        elif TL_Bool and TR_Bool and BR_Bool == False: 
            # print("In PART THREE")
            if distance_from_BR > 50: 
                reward -= 15 
            elif distance_from_BR < 50: 
                reward += 15 
            elif int(distance_from_TL) == 4 or int(distance_from_BL) == 4: 
                reward -= 50
                terminated = True
                self.cleanup() 
            elif int(distance_from_BR) == 4: 
                reward += maxReward // 2 
                BR_Bool = True 
        
        # PART FOUR 
        elif TL_Bool and TR_Bool and BR_Bool and BL_Bool == False: 
            # print("In PART FOUR")
            if distance_from_BL > 34: 
                reward -= 20 
            elif distance_from_BL < 34: 
                reward += 20 
            elif int(distance_from_TL) == 4 or int(distance_from_TR) == 4: 
                reward -= 50 
                terminated = True 
                self.cleanup()
            elif int(distance_from_BL) == 1: 
                reward += maxReward 
                BL_Bool = True 

        
        # NEW REWARD SYSTEM 
        # if TL_Bool == False and TR_Bool == False and BR_Bool == False: 
        #     if self.step_counter % 100 == 0:
        #         print("In part 1")
        #     if distance_from_TL > 50: 
        #         reward -= 5 
        #     elif distance_from_TL < 50: 
        #         reward += 5 
        #     elif distance_from_TL < 25: 
        #         reward += 10 
        #     elif distance_from_TL < 10: 
        #         reward += 15 
        #     elif distance_from_TL <= 8: 
        #         reward += maxReward 
        #         TL_Bool = True 
        #     elif distance_from_TR <= 8: 
        #         reward += maxReward // 2 
        #     elif distance_from_BR <= 8: 
        #         reward += maxReward // 3
        
        # elif TL_Bool == True and TR_Bool == False and BR_Bool == False and BL_Bool == False: 
        #     if self.step_counter % 100 == 0:
        #         print("In part 2")
        #     if distance_from_TR > 34: 
        #         reward -= 10 
        #     elif distance_from_TR < 34: 
        #         reward += 10 
        #     elif distance_from_TR < 34 // 2: 
        #         reward += 15 
        #     elif distance_from_TR < 10: 
        #         reward += 20 
        #     elif distance_from_TR <= 8: 
        #         reward += maxReward  
        #         TR_Bool = True 
        #     elif distance_from_BR <= 8: 
        #         reward += maxReward // 3 
        #     elif distance_from_BL <= 8: 
        #         reward += maxReward // 4 
        
        # elif TL_Bool == True and TR_Bool == True and BR_Bool == False and BL_Bool == False: 
        #     if self.step_counter % 100 == 0:
        #         print("In part 3")
        #     if distance_from_BR > 50: 
        #         reward -= 15 
        #     elif distance_from_BR < 50: 
        #         reward += 10 
        #     elif distance_from_BR < 25: 
        #         reward += 15 
        #     elif distance_from_BR < 10: 
        #         reward += 20 
        #     elif distance_from_BR <= 8: 
        #         reward += maxReward  
        #         BR_Bool = True 
        #     elif distance_from_BL <= 8: 
        #         reward += maxReward // 3 
        #     elif distance_from_TL <= 8: 
        #         reward += maxReward // 4
        
        # elif TL_Bool == True and TR_Bool == True and BR_Bool == True and BL_Bool == False:
        #     if self.step_counter % 100 == 0:
        #         print("In part 4") 
        #     if distance_from_BL > 34: 
        #         reward -= 20 
        #     elif distance_from_BL < 34: 
        #         reward += 20 
        #     elif distance_from_BL <= 34 // 2: 
        #         reward += 25 
        #     elif distance_from_BL < 10: 
        #         reward += 30 
        #     elif distance_from_BL <= 8: 
        #         reward += maxReward * 2
        #         BL_Bool = True 
            
        
        if self.step_counter % 100 == 0: 
            print("\nSteer: ", steer, "Velocity: ", kmh, "Throttle: ", estimated_throttle, "\nBL, TL, TR, BR Bools: ", BL_Bool, TL_Bool, TR_Bool, BR_Bool, "Speed: ", kmh)
            print("Distance from TL:",distance_from_TL)
            print("Distance from TR:",distance_from_TR)
            print("Distance from BR:",distance_from_BR)
            print("Distance from BL:",distance_from_BL)  
            print("Reward: ", reward)          
        
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
        
    def seed(self, seed): 
        pass 
    
    
        
        