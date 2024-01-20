import time
import carla 
import random 
import numpy as np
import cv2 
import math

actorList = []

models = ["dodge", "audi", "mini", "mustang", "nissan", "jeep"]


IM_WIDTH = 640
IM_HEIGHT = 480
    
SHOW_PREVIEW = False
SECONDS_PER_EPISODE = 10

class CarEnvironment: 
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_WIDTH = IM_WIDTH 
    im_HEIGHT = IM_HEIGHT 
    rear_camera = None 
    
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(50.0)
        self.world = self.client.get_world() 
        self.blueprint_library = self.world.get_blueprint_library()
        self.model3 = blueprint_library.filter("model3")[0]
        
    def reset(self):
        self.collision_history = []
        self.actor_list = [] 
        
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model3, self.transform)
        
        self.actor_list.append(self.vehicle)
        
        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_WIDTH}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_HEIGHT}")
        self.rgb_cam.set_attribute("fov", f"110")
        
        transform = carla.Transform(carla.Location(x = -5, z = 2))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to = self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        time.sleep(4)
        
        col_sensor = self.blueprint_library.find("sensor.other.collision")
        self.col_sensor = self.world.spawn_actor(col_sensor, transform, attach_to = self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.collision_data(event))
        
        while self.front_camera is None: 
            time.sleep(0.01)
            
        self.episode_start = time.time() 
        
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        
        return self.front_camera
    
    def collision_data(self, event):
        self.collision_history.append(event)
    
    # Processing image from camera sensor 
    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_HEIGHT, self.im_WIDTH, 4))
        i3 = i2[:, :, :3]
        
        if self.SHOW_CAM: 
            cv2.imshow("", i3)
            cv2.waitKey(1)
        
        self.front_camera = i3
        
    def step(self, action):
        # Go left 
        if action == 0: 
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0.2, steer = -1.0 * self.STEER_AMT))
        # Go straight
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0.2, steer = 0))
        # Turn right
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer = 1.0 * self.STEER_AMT))
            
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        if len(self.collision_history) != 0: 
            done = True 
            reward = -200 
        elif kmh < 50: 
            done = False 
            reward = -1 
        else: 
            done = False 
            reward = 1 
            
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True 
        
        return self.front_camera, reward, done, None 
            
            
        
        

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
    visualize_spawn_points(spawn_points)
        
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