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


FIRST_PED_LOCATION = carla.Transform(carla.Location(x=-325.865570, y=33.573753, z=0.281942), carla.Rotation(yaw = 180))
SECOND_PED_LOCATION = carla.Transform(carla.Location(x=-300.865570, y=30.573753, z=0.781942), carla.Rotation(yaw = 180))
THIRD_PED_LOCATION = carla.Transform(carla.Location(x = -290.865570, y=36.573753, z = 1.781942), carla.Rotation(yaw = 180))

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

def semantic_process_img(image, windowName): 
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((IM_HEIGHT, IM_WIDTH, 4))[:, :, :3] 
        frontCamera = i 
        
        cv2.imshow(windowName, frontCamera)
        cv2.waitKey(1)
        
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

def add_collision_sensor(blueprint_lib, vehicle_id): 
    # Adding a collision sensor to the vehicle 
    colsensor = blueprint_lib.find("sensor.other.collision")
    sensor_location = carla.Transform(carla.Location(z = 0, y = 0))
    colsensor = world.spawn_actor(colsensor, sensor_location, attach_to = vehicle_id)
    actorList.append(colsensor)
    
    colsensor.listen(lambda event: collision_data(event))
    
def collision_data(event):
    actor_we_collide_against = event.other_actor.type_id
    if "pedestrian" in actor_we_collide_against: 
        print("COLLIED WITH A PEDESTRIAN! HE DEAD ~~")
    else: 
        print("Collided with something else!")

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

def semantic_camera(blueprint_lib, vehicle_id): 
    semantic_camera = blueprint_library.find("sensor.camera.semantic_segmentation")
    semantic_camera.set_attribute("image_size_x", f"{IM_WIDTH}")
    semantic_camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
    semantic_camera.set_attribute("fov", f"90")
    
    camera_init_trans = carla.Transform(carla.Location(z = 1.3, x= 1.4))
    sensor = world.spawn_actor(semantic_camera, camera_init_trans, attach_to = vehicle_id)
    actorList.append(sensor)
    sensor.listen(lambda data: semantic_process_img(data, "Semantic Camera"))
    
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
        steerAmount = 0.1     
        vehicle.apply_control(carla.VehicleControl(throttle = 0))
        # vehicle.apply_ackermann_control(carla.VehicleAckermannControl(speed = float(2.77778), steer = float(-0.0436332)))
        # command.ApplyTargetVelocity(vehicle, float(1.38889))
        # pass 
    
    if vehicle is not None: 
        actorList.append(vehicle)
        # return ego_vehicle
        return vehicle, spawn_location
        
def spawn_pedestrian(location, pedestrian_id): 
    walker_bp = blueprint_library.find(pedestrian_id)
    spawn_location = location
    pedestrian = world.try_spawn_actor(walker_bp, spawn_location)
    if pedestrian is not None: 
        actorList.append(pedestrian)
        print("THE PEDESTRIAN SHOULD BE SPAWNED")
        return pedestrian 
    else: 
        print("THE PEDESTRIAN WAS NOT ADDED FOR SOME REASON")
        return None 
    

def spawn_crash_vehicle(): 
    vehicle_bp = blueprint_library.find("vehicle.mini.cooper_s_2021")
    spawn_location = carla.Transform(carla.Location(x=-325.865570, y=33.573753, z=0.281942), carla.Rotation(yaw = 180))
    crash_vehicle = world.try_spawn_actor(vehicle_bp, spawn_location)
    print("CRASH VEHICLE SPAWNED")


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
    # spawn_point_location = spawn_points[351].location
    # print(spawn_point_location)
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
    # vehicle, spawn_location = add_ego_vehicle(blueprint_library, "vehicle.tesla.model3", spawn_index = 330)
    vehicle, spawn_location = add_ego_vehicle(blueprint_library, "vehicle.tesla.model3", spawn_index = 350)

    # SPAWNING A PEDESTRIAN 
    firstPedestrian = spawn_pedestrian(FIRST_PED_LOCATION, "walker.pedestrian.0001")
    secondPedestrian = spawn_pedestrian(SECOND_PED_LOCATION, "walker.pedestrian.0016")
    thirdPedestrian = spawn_pedestrian(THIRD_PED_LOCATION, "walker.pedestrian.0002")
    
    if firstPedestrian and secondPedestrian and thirdPedestrian is not None: 
        print("Pedestrians Spawned Successfully!")
    else: 
        print("PEDESTRIAN SPAWN ERROR!")
    
    # SPAWNING A CRASH VEHICLE 
    # spawn_crash_vehicle()

    vehicle_cam(blueprint_library, vehicle)
    # heli_cam(blueprint_library, vehicle)
    # add_lane_sensor(blueprint_library, vehicle)
    semantic_camera(blueprint_library, vehicle)
    add_collision_sensor(blueprint_library, vehicle)

    # Spawning multiple vehicles 
    # spawn_traffic(models, 50)
    

    while True:
        world.tick()
        # Printing speed of the vehicle 
        # velocity = vehicle.get_velocity()
        # kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        # print("The vehicle is moving at ", int(kmh), "km/h")
        # print("Distance Travelled: ", distance_travelled)
        # print("Vehicle Location: ", vehicle.get_location())
        # distance_travelled = int(spawn_location.distance(vehicle.get_location()))
        # print("Distance Travelled: ", distance_travelled)
        # if distance_travelled == 100:
        #     print("REACHED 100")
        # elif distance_travelled == 100 + 50: 
        #     print("REACHED 150")
        
        # Printing the location of the pedestrian 
        vehicle_location = vehicle.get_location() 
        first_ped_location = firstPedestrian.get_location() 
        # print("The location of the first pedestrian: ", first_ped_location)
        first_ped_vehicle_distance = int(vehicle_location.distance(first_ped_location))
        # print("DISTANCE BETWEEN CAR AND MAN: ", first_ped_vehicle_distance)
        
        
            


finally:
    for actor in actorList: 
        actor.destroy()  
    print("Removed all the actors!")
