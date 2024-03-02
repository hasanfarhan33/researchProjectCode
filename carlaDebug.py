# Use this to understand the environment and the information about the vehicles 
import carla 
import random 
import numpy as np 
import cv2 
import math
import time
from carla import command

actorList = [] 
models = ["dodge", "audi", "mini", "mustang", "nissan", "jeep"]
IM_WIDTH = 200
IM_HEIGHT = 200
FPS = 10

def add_lidar(blueprint_lib, vehicle_id):
    lidar_bp = blueprint_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", str(32))
    lidar_bp.set_attribute("points_per_second", str(5000))
    lidar_bp.set_attribute("rotation_frequency", str(40))
    lidar_bp.set_attribute("range", str(15))
    lidar_location = carla.Location(0, 0, 2) 
    lidar_rotation = carla.Rotation(0, 0, 0)
    lidar_transform = carla.Transform(lidar_location, lidar_rotation)
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to = vehicle_id)
    actorList.append(lidar_sensor)
    lidar_sensor.listen(lambda point_cloud: point_cloud.save_to_disk('./lidarData/%.6d.ply' % point_cloud.frame))
    

def visualize_spawn_points(spawn_points):
    for i, spawn_point in enumerate(spawn_points):
        world.debug.draw_string(spawn_point.location, str(i), life_time = 400)
    
    # Visualizing a single spawn point
    # world.debug.draw_string(spawn_points[80].location, str(80), life_time = 100)

def manage_traffic(synchronous_mode = True, seed = 0):    
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(synchronous_mode)
    traffic_manager.set_random_device_seed(seed)
    random.seed(seed)

# Processing image from camera sensor 
def process_img(image, windowName):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow(windowName, i3)
    cv2.waitKey(1)
    return i3/255.0

def vehicle_cam(blueprint_lib, vehicle_id):
    camera_bp = blueprint_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_bp.set_attribute("fov", "90")
    
    # Spawning camera relative to the car
    spawn_point = carla.Transform(carla.Location(x = -5, z = 2))
    camera = world.spawn_actor(camera_bp, spawn_point, attach_to = vehicle_id)
    actorList.append(camera)
    
    # Getting info from the sensor 
    camera.listen(lambda data: process_img(data, "Chase Cam"))
    
def add_lane_sensor(blueprint_lib, vehicle_id): 
    lane_sensor = blueprint_lib.find("sensor.other.lane_invasion")
    spawn_point = carla.Transform(carla.Location(x = 0, z = 0))
    lane_sensor = world.spawn_actor(lane_sensor, spawn_point, attach_to = vehicle_id)
    actorList.append(lane_sensor)
    
    lane_sensor.listen(lambda event: lane_invasion_data(event))

def lane_invasion_data(event): 
    print("The car crossed the lane: ", event)

def heli_cam(blueprint_lib, vehicle_id):
    camera_bp = blueprint_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_bp.set_attribute("fov", "90")
    
    # Spawning the camera relative to the car 
    spawn_point = carla.Transform(carla.Location(x =0.5, z=5), carla.Rotation(pitch=8.0))
    heli_sensor = world.spawn_actor(camera_bp, spawn_point, attach_to = vehicle_id, attachment_type = carla.AttachmentType.SpringArm)
    actorList.append(heli_sensor)
    
    heli_sensor.listen(lambda data: process_img(data, "Helicopter Camera"))
    
def printing_blueprint_lib(): 
    blueprints = [bp for bp in world.get_blueprint_library().filter("*")]
    for blueprint in blueprints: 
        print(blueprint.id)
        for attr in blueprint: 
            print(' - {}'.format(attr))

def add_ego_vehicle(blueprint_lib, vehicle_id, autonomy = False, spawn_index = -1):
    ego_vehicle = blueprint_lib.find(vehicle_id)
    # print(ego_vehicle)
        
    ego_vehicle.set_attribute("role_name", "hero")
    if spawn_index == -1:
        vehicle = world.try_spawn_actor(ego_vehicle, random.choice(spawn_points))
    else:
        vehicle = world.try_spawn_actor(ego_vehicle, spawn_points[spawn_index])
        spawn_location = spawn_points[spawn_index].location
        print("LOCATION OF THE SPAWN POINT: ", spawn_location)
        
    
    if autonomy:
            vehicle.set_autopilot(True)
    else: 
        # Controlling the car manually  
        steerAmount = 0.01     
        # vehicle.apply_control(carla.VehicleControl(throttle = 0.2))
        vehicle.apply_ackermann_control(carla.VehicleAckermannControl(speed = float(2.77778), steer = float(-0.0436332)))
        # command.ApplyTargetVelocity(vehicle, float(1.38889))
        # pass 
    
    if vehicle is not None: 
        actorList.append(vehicle)
        # return ego_vehicle
        return vehicle, spawn_location

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(150.0)

    client.load_world("Town04")

    world = client.get_world() 
    world.set_weather(carla.WeatherParameters.ClearNoon)
    
    # Checking blueprint library 
    # printing_blueprint_lib()
    
    # Changing the settings
    frame = world.apply_settings(
        carla.WorldSettings(
            synchronous_mode = True,
            fixed_delta_seconds = 1.0 / FPS, 
        )
    )

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
    vehicle, spawn_location = add_ego_vehicle(blueprint_library, "vehicle.tesla.model3", spawn_index = 330)


    # vehicle_cam(blueprint_library, vehicle)
    heli_cam(blueprint_library, vehicle)
    add_lane_sensor(blueprint_library, vehicle)

    # Spawning multiple vehicles 
    # spawn_traffic(models, 50)
    

    while True:
        world.tick()
        # Printing speed of the vehicle 
        velocity = vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        print("The vehicle is moving at ", int(kmh), "km/h")
        # print("Distance Travelled: ", distance_travelled)
        # print("Vehicle Location: ", vehicle.get_location())
        distance_travelled = int(spawn_location.distance(vehicle.get_location()))
        # print("Distance Travelled: ", distance_travelled)
        if distance_travelled == 100:
            print("REACHED 100")
        elif distance_travelled == 100 + 50: 
            print("REACHED 150")
            


finally:
    for actor in actorList: 
        actor.destroy()
    print("Removed all the actors!")
