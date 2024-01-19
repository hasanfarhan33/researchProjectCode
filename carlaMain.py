import carla 
import random 
import numpy as np
import cv2 

actorList = []

models = ["dodge", "audi", "mini", "mustang", "nissan", "jeep"]


IM_WIDTH = 640
IM_HEIGHT = 480

# Processing image from camera sensor 
def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3/255.0
    

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