import time
import carla 
import random 
import numpy as np
import cv2 
import math
import gymnasium as gym 
from gymnasium import spaces
from setup import setup
from queue import Queue
import atexit 
import os 
import signal 
import sys 
from absl import logging 
import pygame 
import graphics

IM_WIDTH = 400
IM_HEIGHT = 400
    
SHOW_PREVIEW = False
SECONDS_PER_EPISODE = 10

class CarEnvironment: 
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_WIDTH = IM_WIDTH 
    im_HEIGHT = IM_HEIGHT 
    rear_camera = None 
    metadata = {"render.modes":["human"], "render_fps":60}
    
    def __init__(self, town, fps, repeat_action, start_transform_type, sensors, action_type, enable_preview, steps_per_episode, playing = False, timeout = 60):
        self.client, self.world, self.frame, self.server = setup(town = town, fps = fps, client_timeout = timeout)
        
        self.client.set_timeout(50.0)
        self.world = self.world().get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model3 = blueprint_library.filter("model3")[0]
        self.repeat_action = repeat_action
        self.action_type = action_type
        self.start_transform_type = start_transform_type
        self.sensors = sensors 
        self.actor_list = [] 
        self.preview_camera = None 
        self.steps_per_episode = steps_per_episode
        self.playing = playing 
        self.preview_camera_enabled = enable_preview
        
    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation spec of the sensor"""
        return gym.spaces.Box(low = 0.0, high = 255.0, shape=(IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8)
    
    @property 
    def action_space(self):
        """Returns the expected action passed to the 'step' method"""
        if self.action_type == "continuous":
            return gym.spaces.Box(low=np.array([-1.0, -1.0]), high = np.array([1.0, 1.0]))
        elif self.action_type == "discrete":
            return gym.spaces.MultiDiscrete([4, 9])
    
    #TODO: Might not need this        
    def seed(self, seed): 
        if not seed: 
            seed = 7 
        random.seed(seed)
        self._np_random = np.random.RandomState(seed)
        return seed
    
    def reset(self):
        self.destroy_agents()
        self.collision_history = []
        self.lane_invasion_hist = [] 
        self.actor_list = [] 
        self.frame_step = 0 
        self.out_of_loop = 0 
        self.dist_from_start = 0
        
        self.chase_cam_queue = Queue() 
        self.preview_image_queue = Queue()
        
        # When CARLA breaks or spawn point is already occupied 
        spawn_start = time.time()
        while True: 
            try:
                self.random_spawn = False 
                
                # Spawning at a random location
                if self.random_spawn:
                    self.start_transform = random.choice(self.world.get_map().get_spawn_points())
                    self.vehicle = self.world.spawn_actor(self.model3, self.start_transform)
                # Spawning at a certain location
                else:
                    self.spawn_points = self.world.get_map().get_spawn_points() 
                    self.spawn_index = 176
                    self.vehicle = self.world.spawn_actor(self.model3, self.spawn_points[self.spawn_index])
                    
                break 
            except Exception as e:
                logging.error("Error carla 141 {}".format(str(e)))
                time.sleep(0.01)
        
            if time.time() > spawn_start + 3: 
                raise Exception("Can't spawn a car!")
        
        
        
        self.actor_list.append(self.vehicle)
        
        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_WIDTH}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_HEIGHT}")
        self.rgb_cam.set_attribute("fov", "90")
        
        transform = carla.Transform(carla.Location(x = -5, z = 2))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to = self.vehicle)
        self.sensor.listen(self.chase_cam_queue.put)
        self.actor_list.append(self.sensor)
        
        
        
        # Top down camera 
        if self.preview_camera_enabled:
            self.preview_cam = self.world.get_blueprint_library().find("sensor.camera.rgb")
            self.preview_cam.set_attribute("image_size_x", 400)
            self.preview_cam.set_attribute("image_size_y", 400)
            self.preview_cam.set_attribute("fov", 100)
            transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
            self.preview_sensor = self.world.spawn_actor(self.preview_cam, transform, attach_to = self.vehicle, attachment_type=carla.AttachmentType.SpringArm)
            self.preview_sensor.listen(self.preview_image_queue.put)
            self.actor_list.append(self.preview_sensor)
            
            
        
        self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, brake = 1.0))
        time.sleep(4)
        
        # Collision History 
        self.collision_hist = [] 
        self.lane_invasion_hist = [] 
        
        # Collision sensor
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to = self.vehicle)
        self.actor_list.append(self.colsensor)
        
        # Lane sensor 
        lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lanesensor = self.world.spawn_actor(lanesensor, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.lanesensor)
        
        self.world.tick() 
        
        # Wait for a camera to send first image 
        while self.chase_cam_queue.empty():
            logging.debug("Waiting for camera to be ready")
            time.sleep(0.01)
            self.world.tick()
        
        # Disengage brakes 
        self.vehicle.apply_control(carla.VehicleControl(brake = 0.0))
        
        image = self.chase_cam_queue.get() 
        image = np.array(image.raw_data)
        image = image.reshape((IM_HEIGHT, IM_WIDTH, -1))
        image = image[:, :, :3]
        
        
        return image
    
    def step(self, action):
        total_reward = 0 
        for _ in range(self.repeat_action):
            obs, rew, done, info = self._step(action)
            total_reward += rew 
            if done: 
                break 
        return obs, total_reward, done, info
    
    # Steps Environment 
    def _step(self, action):
        self.world.tick()
        self.render()
        
        self.frame_step += 1
        
        # Apply control to the vehicle based on an action 
        if self.action_type == "continuous":
            if action[0] > 0: 
                action = carla.VehicleControl(throttle = float(action[0]), brake = 0)
            else: 
                action = carla.VehicleControl(throttle = 0, brake = -float(action[0]))
        elif self.action_type == "discrete": 
            if action[0] == 0: 
                action = carla.VehicleControl(throttle = 0, brake = 1)
            else: 
                action = carla.VehicleControl(throttle = float((action[0])/2), brake = 0)
        else: 
            raise NotImplementedError()
        
        logging.debug('{}, {}'.format(action.throttle, action.brake))
        self.vehicle.apply_control(action)
        
        # Calculating the speed of the vehicle (kmh)
        v = self.vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        loc = self.vehicle.get_location()
        new_dist_from_start = loc.distance(self.start_transform.location)
        square_dist_diff = new_dist_from_start ** 2 - self.dist_from_start ** 2 
        self.dist_from_start = new_dist_from_start 
        
        image = self.chase_cam_queue.get()
        image = np.arry(image.raw_data)
        image = image.reshape((IM_HEIGHT, IM_WIDTH, -1))
        
        if 'rgb' in self.sensors: 
            image = image[:, :, :3]
        
        
        done = False 
        reward = 0 
        info = dict() 
        
        # If the car collides - end the episode and send back a penalty
        # TODO: Add lane invasion history later
        if len(self.collision_hist) != 0: 
            done = True 
            reward += -100 
            self.collision_hist = [] 
            
        # Reward for speed and distance 
        reward += 0.1 * kmh 
        reward += square_dist_diff
        
        if self.frame_step >= self.steps_per_episode: 
            done = True 
            
        if done: 
            logging.debug("Env lasts {} steps, restarting ...".format(self.frame_step))
            self.destroy_agents()
            
        return image, reward, done, info
            
    
    def close(self): 
        logging.info("Closes the CARLA server with process PID {}".format(self.server.pid))
        os.killpg(self.server.pid, signal.SIGKILL)
        atexit.unregister(lambda: os.killpg(self.server.pid, signal.SIGKILL))
        
    def render(self, mode = "human"):
        if self.preview_camera_enabled:
            self.display, self._clock, self._font = graphics.setup(width = IM_WIDTH, height = IM_HEIGHT, render=(mode=="human"))
            
            preview_img = self.preview_image_queue.get() 
            preview_img = np.array(preview_img.raw_data)
            preview_img = preview_img.reshape(IM_WIDTH, IM_HEIGHT, -1)
            graphics.make_dashboard(
                display = self.display, 
                font = self._font, 
                clock = self.clock, 
                observations = {"preview_camera":preview_img}, 
            )
            
            if mode == "human": 
                pygame.display.flip()
            else: 
                raise NotImplementedError() 
            
    def _destroy_agents(self): 
        for actor in self.actor_list: 
            # If it has a callback attached, remove it first 
            if hasattr(actor, "is_listening") and actor.is_listening: 
                actor.stop()
                
            # If it is still alive destroy it 
            if actor.is_alive: 
                actor.destroy() 
        self.actor_list = [] 
    
    def _collision_data(self, event): 
        collision_actor_id = event.other_actor.type_id 
        collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)
        
        # Add collision 
        self.collision_hist.append(event)
    
    # TODO: Use it later
    def _lane_invasion_data(self, event): 
        self.lane_invasion_hist.append(event)
        
        
    # Processing image from camera sensor 
    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_HEIGHT, self.im_WIDTH, 4))
        i3 = i2[:, :, :3]
        
        if self.SHOW_CAM: 
            cv2.imshow("", i3)
            cv2.waitKey(1)
        
        self.front_camera = i3
        
        
        
def spawn_traffic(models, maxVehicles):
    manage_traffic()
    blueprints = [] 
    for vehicle in blueprint_library.filter("*vehicle*"):
        if any(model in vehicle.id for model in models):
            blueprints.append(vehicle)
            
    max_vehicles = maxVehicles
    max_vehicles = min([max_vehicles, len(spawn_points)])
    vehicles = [] 
    
    for i, spawn_point in enumerate(random.sample(spawn_points, max_vehicles)): 
        temp = world.try_spawn_actor(random.choice(blueprints), spawn_point)
        if temp is not None:
            vehicles.append(temp)
            
    # Setting autopilot on spawned vehicles 
    for vehicle in vehicles:
        vehicle.set_autopilot(True)
        # Randomly set the probability that the vehicle will ignore traffic lights 
        traffic_manager.ignore_lights_percentage(vehicle, random.randint(0, 50))

def add_ego_vehicle(blueprint_lib, vehicle_id, autonomy = False, spawn_index = -1):
    ego_vehicle = blueprint_lib.find(vehicle_id)
    # print(ego_vehicle)
    
    
    
    ego_vehicle.set_attribute("role_name", "hero")
    if spawn_index == -1:
        vehicle = world.try_spawn_actor(ego_vehicle, random.choice(spawn_points))
    else:
        vehicle = world.try_spawn_actor(ego_vehicle, spawn_points[spawn_index])
    
    if autonomy:
            vehicle.set_autopilot(True)
    else: 
        # Controlling the car manually        
        vehicle.apply_control(carla.VehicleControl(throttle = 0.2, steer = 0.0))
        # pass 
    
    if vehicle is not None: 
        actorList.append(vehicle)
        # return ego_vehicle
        return vehicle


def printing_blueprint_lib(): 
    blueprints = [bp for bp in world.get_blueprint_library().filter("*")]
    for blueprint in blueprints: 
        print(blueprint.id)
        for attr in blueprint: 
            print(' - {}'.format(attr))

# TODO: ADD CAMERA TO EGO VEHICLE
def vehicle_cam(blueprint_lib, vehicle_id):
    camera_bp = blueprint_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_bp.set_attribute("fov", "110")
    
    # Spawning camera relative to the car
    spawn_point = carla.Transform(carla.Location(x = -5, z = 2))
    camera = world.spawn_actor(camera_bp, spawn_point, attach_to = vehicle_id)
    actorList.append(camera)
    
    # Getting info from the sensor 
    camera.listen(lambda data: process_img(data))
    

def manage_traffic(synchronous_mode = True, seed = 0):    
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(synchronous_mode)
    traffic_manager.set_random_device_seed(seed)
    random.seed(seed)

def visualize_spawn_points(spawn_points):
    for i, spawn_point in enumerate(spawn_points):
        world.debug.draw_string(spawn_point.location, str(i), life_time = 100)
    
    # Visualizing a single spawn point
    # world.debug.draw_string(spawn_points[80].location, str(80), life_time = 100)

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(50.0)

    client.load_world("Town05")

    world = client.get_world() 
    
    # Checking blueprint library 
    # printing_blueprint_lib()

    # Setting up synchronous mode 
    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.025
    world.apply_settings(settings)

    # Traffic Manager 
    # manage_traffic()

    # spectator = world.get_spectator()

    blueprint_library= world.get_blueprint_library()

    # Spawning a vehicle 
    spawn_points = world.get_map().get_spawn_points()
    # print(len(spawn_points))
        
    # Checking all the spawn points 
    # visualize_spawn_points(spawn_points)
        
    # Spawning single vehicle
    # merc_car = blueprint_library.filter("vehicle.mercedes.coupe_2020")[0]
    # print(merc_car)
    
    # merc_car = blueprint_library.filter("vehicle.mercedes.coupe_2020")[0]
    # print(merc_car)

    # ego_vehicle = add_ego_vehicle(blueprint_library, "vehicle.mercedes.coupe_2020", autonomy=False)
    # print(ego_vehicle)
    # ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0))
    
    # SPAWNING EGO VEHICLE 
    vehicle = add_ego_vehicle(blueprint_library, "vehicle.tesla.model3", spawn_index = 176)


    vehicle_cam(blueprint_library, vehicle)

    # Spawning multiple vehicles 
    # spawn_traffic(models, 50)



    while True:
        world.tick()

finally:
    for actor in actorList: 
        actor.destroy()
    print("Removed all the actors!")