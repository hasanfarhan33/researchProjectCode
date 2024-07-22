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


# FIRST_PED_LOCATION = carla.Transform(carla.Location(x=-325.865570, y=33.573753, z=0.281942), carla.Rotation(yaw = 180))
# SECOND_PED_LOCATION = carla.Transform(carla.Location(x=-300.865570, y=30.573753, z=0.781942), carla.Rotation(yaw = 180))
# THIRD_PED_LOCATION = carla.Transform(carla.Location(x = -275.865570, y=36.573753, z = 1.781942), carla.Rotation(yaw = 180))
# FOURTH_PED_LOCATION = carla.Transform(carla.Location(x = -250.865570, y=26.573753, z = 2.781942), carla.Rotation(yaw = 180))

# Choose a location at random 
# first_loc = carla.Transform(carla.Location(x = -325.865570, y=33.573753, z=2), carla.Rotation(yaw = 180))
# second_loc = carla.Transform(carla.Location(x = -325.865570, y = 30.573753, z = 2), carla.Rotation(yaw = 180))
# third_loc = carla.Transform(carla.Location(x = -325.865570, y = 36.573753, z = 2), carla.Rotation(yaw = 180))
# fourth_loc = carla.Transform(carla.Location(x = -325.865570, y = 26.573753, z = 2), carla.Rotation(yaw = 180))

# ped_locations = [first_loc, second_loc, third_loc, fourth_loc]

spectator_location = carla.Transform(carla.Location(x=-365.865570, y=33.573753, z=10.0))

FLAG_LOCATION = carla.Location(x = -300.865570, y=33.573753, z=0.281942)
flagCollected = False

# VARIABLES FOR PARKING LOT 
bottomLeftPL = carla.Transform(carla.Location(x = 15, y = -12.5, z = 4), carla.Rotation(yaw = 180))
topLeftPL = carla.Transform(carla.Location(x = -35, y = -12.5, z = 4), carla.Rotation(yaw = -90))
topRightPL = carla.Transform(carla.Location(x = -35, y = -47, z = 4), carla.Rotation(yaw = 0))
bottomRightPL = carla.Transform(carla.Location(x = 15, y = -47, z = 4), carla.Rotation(yaw = 90))

npcVehicleLocOne = carla.Transform(carla.Location(x= -40, y= -25.75, z= 4), carla.Rotation(yaw = -90))
npcVehicleLocTwo = carla.Transform(carla.Location(x = -40, y= -29.5, z = 4), carla.Rotation(yaw = -90)) 
npcVehicleLocThree = carla.Transform(carla.Location(x = -40, y = -33.5, z = 4), carla.Rotation(yaw = -90))

npcVehicleLocFour = carla.Transform(carla.Location(x= 20, y= -32.75, z= 4), carla.Rotation(yaw = -90))
npcVehicleLocFive = carla.Transform(carla.Location(x= 20, y= -26.5, z= 4), carla.Rotation(yaw = -90))
# npcVehicleLocSix = carla.Transform(carla.Location(x= 19, y= -33.5, z= 4), carla.Rotation(yaw = -90))

Reached_TL = False 

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
        # steerAmount = 0.1 
        
        # vehicle_location = vehicle.get_location()
        # distance_from_flag = vehicle_location.distance(FLAG_LOCATION)
        # distance_from_spawn = vehicle_location.distance(spawn_location)
        
        # if distance_from_flag == 0: 
        #     flagCollected = True 
        # elif distance_from_spawn == 0 and flagCollected == True: 
        #     print("FINISHED SCENARIO")
        
        vehicle.apply_control(carla.VehicleControl(throttle = 0, reverse = True))    
            
        # if not flagCollected:     
        #     vehicle.apply_control(carla.VehicleControl(throttle = 0.3, reverse = False))
        # elif flagCollected: 
        #     print("You are here!")
        #     vehicle.apply_control(carla.VehicleControl(throttle = 0.3, reverse = True))
        # elif flagCollected: 
        #     vehicle.apply_control(carla.VehicleControl(throttle = 0.3, reverse = True))
        # elif flagCollected and distance_from_spawn == 0: 
        #     vehicle.apply_control(carla.VehicleControl(throttle = 0, brake = 1.0))
        # vehicle.apply_ackermann_control(carla.VehicleAckermannControl(speed = float(2.77778), steer = float(-0.0436332)))
        # command.ApplyTargetVelocity(vehicle, float(1.38889))
        pass 
    
    if vehicle is not None: 
        actorList.append(vehicle)
        # return ego_vehicle
    return vehicle, spawn_location
    
def control_vehicle(vehicle, flag, spawn_loc): 
    vehicle_location = vehicle.get_location()
    distance_from_flag = vehicle_location.distance(FLAG_LOCATION)
    distance_from_spawn = vehicle_location.distance(spawn_loc)
    reverseBool = False
    
    if distance_from_flag == 0: 
        reverseBool = True 
        flag = True
        vehicle.apply_control(carla.VehicleControl(throttle = 0.3, reverse = reverseBool)) 
    elif distance_from_spawn == 0 and flagCollected == True: 
        vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 1.0, reverse = False, handbrake = True))

def spawn_vehicle_parkingLot(blueprint_lib, vehicle_id, location): 
    init_loc = location
    ego_vehicle = blueprint_lib.find(vehicle_id)
    ego_vehicle.set_attribute("role_name", "hero")
    vehicle = world.try_spawn_actor(ego_vehicle, init_loc)

    if vehicle is not None: 
        actorList.append(vehicle)
        print("THE VEHICLE SHOULD BE SPAWNED")
        
        return vehicle 
    else: 
        print("THE VEHICLE WAS NOT SPAWNED!")
        return None
    
        
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
    

def spawn_ped_random(location_arr, pedestrian_id): 
    walker_bp = blueprint_library.find(pedestrian_id)
    spawn_location = random.choice(ped_locations)
    print(spawn_location)
    pedestrian = world.try_spawn_actor(walker_bp, spawn_location)
    if pedestrian is not None: 
        actorList.append(pedestrian)
        print("THE PEDESTRIAN SHOULD BE SPAWNED")
        return pedestrian 
    else: 
        print("THE PEDESTRIAN HAS NOT BEEN SPAWNED FOR SOME REASON")
        return None 

def spawn_vehicle(blueprint_lib, vehicle_id, location): 
    init_loc = location
    vehicle_bp = blueprint_lib.find(vehicle_id)
    npc_vehicle = world.try_spawn_actor(vehicle_bp, init_loc)
    if npc_vehicle is not None: 
        actorList.append(npc_vehicle)
        print("Vehicle added")
        return npc_vehicle
    else: 
        print("Vehicle could not be added")    
        return npc_vehicle    

def maintain_speed(s, preferred_speed, speed_threshold): 
    if s >= preferred_speed: 
        return 0 
    elif s < preferred_speed - speed_threshold: 
        return 0.7
    else: 
        return 0.5 

# Function to spawn NPC vehicles
def spawn_npc(blueprint_lib, vehicle_id, location): 
    init_loc = location
    vehicle_bp = blueprint_lib.find(vehicle_id)
    npc_vehicle = world.try_spawn_actor(vehicle_bp, init_loc)
    if npc_vehicle is not None: 
        actorList.append(npc_vehicle)
        # print("Vehicle added")
    else: 
        return 

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(150.0)

    client.load_world("Town04")
    # client.load_world("Town05_Opt")

    world = client.get_world() 
    world.set_weather(carla.WeatherParameters.ClearNoon)
    
    # Removing unnecessary things from the map
    # world.unload_map_layer(carla.MapLayer.Buildings)
    # world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    # world.unload_map_layer(carla.MapLayer.StreetLights)
    # world.unload_map_layer(carla.MapLayer.Decals)
    # world.unload_map_layer(carla.MapLayer.Foliage)
    # world.unload_map_layer(carla.MapLayer.Walls)
    
    # All the buildings have been removed
        
    
    # Setting the location of the spectator 
    world.get_spectator().set_transform(spectator_location)
    
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
    # vehicle, spawn_location = add_ego_vehicle(blueprint_library, "vehicle.tesla.model3", spawn_index = 62)
    # print(vehicle.bounding_box.extent)
    vehicle, spawn_location = add_ego_vehicle(blueprint_library, "vehicle.tesla.model3", spawn_index = 350)
    
    # SPAWNING EGO PARKING VEHICLE 
    # bottomLeftVehicle = spawn_vehicle_parkingLot(blueprint_library, "vehicle.tesla.model3", location = bottomLeftPL)
    # topLeftVehicle = spawn_vehicle_parkingLot(blueprint_library, "vehicle.tesla.model3", location = topLeftPL)
    # topRightVehicle = spawn_vehicle_parkingLot(blueprint_library, "vehicle.tesla.model3", location = topRightPL)
    # bottomRightVehicle = spawn_vehicle_parkingLot(blueprint_library, "vehicle.tesla.model3", location = bottomRightPL)
    
    # SPAWNING NPC VEHICLES
    # spawn_npc(blueprint_library, "vehicle.mini.cooper_s", npcVehicleLocOne)
    # spawn_npc(blueprint_library, "vehicle.mini.cooper_s", npcVehicleLocTwo)
    # spawn_npc(blueprint_library, "vehicle.mini.cooper_s", npcVehicleLocThree)
    # spawn_npc(blueprint_library, "vehicle.mini.cooper_s", npcVehicleLocFour)
    # spawn_npc(blueprint_library, "vehicle.mini.cooper_s", npcVehicleLocFive)
    
    #Calculating distance 
    # blTlDistance = int(bottomLeftPL.distance(topLeftPL))
    # tltRDistance = int(topLeftPL.distance(topRightPL))
    # trBrDistance = int(topRightPL.distance(bottomRightPL))
    # brBlDistance = int(bottomRightPL.distance(bottomLeftPL))
    
    # print("BL TL Distance: ", blTlDistance)
    # print("TL TR Distance: ", tltRDistance)
    # print("TR BR Distance: ", trBrDistance)
    # print("BR BL Distance: ", brBlDistance)
    
    # SPAWNING A PEDESTRIAN 
    # firstPedestrian = spawn_pedestrian(FIRST_PED_LOCATION, "walker.pedestrian.0030")
    # secondPedestrian = spawn_pedestrian(SECOND_PED_LOCATION, "walker.pedestrian.0032")
    # thirdPedestrian = spawn_pedestrian(THIRD_PED_LOCATION, "walker.pedestrian.0002")
    # fourthPedestrian = spawn_pedestrian(FOURTH_PED_LOCATION, "walker.pedestrian.0034")
    
    # random_ped = spawn_ped_random(ped_locations, "walker.pedestrian.0030")
    
    # if firstPedestrian and secondPedestrian and thirdPedestrian is not None: 
    #     print("Pedestrians Spawned Successfully!")
    # else: 
    #     print("PEDESTRIAN SPAWN ERROR!")
    
    # SPAWNING A CRASH VEHICLE 
    # spawn_crash_vehicle()

    # CAMERAS AND SENSORS 
    # vehicle_cam(blueprint_library, vehicle)
    # heli_cam(blueprint_library, vehicle)
    # add_lane_sensor(blueprint_library, vehicle)
    # semantic_camera(blueprint_library, vehicle)
    # add_collision_sensor(blueprint_library, vehicle)

    # Spawning multiple vehicles 
    # spawn_traffic(models, 50)
    

    while True:
        world.tick()
        
        # Printing speed of the vehicle 
        velocity = vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Maintaining the speed of the vehicle 
        estimated_throttle = maintain_speed(kmh, 60, 50)
        vehicle.apply_control(carla.VehicleControl(throttle = estimated_throttle, steer = 0.0, brake = 0.0))
        
        print("The vehicle is moving at ", int(kmh), "km/h")
        # print("Distance Travelled: ", distance_travelled)
        # print("Vehicle Location: ", vehicle.get_location())
        distance_travelled = int(spawn_location.distance(vehicle.get_location()))
        print("Distance Travelled: ", distance_travelled)
        if distance_travelled == 50:
            print("REACHED EASY DISTANCE")
        elif distance_travelled == 100: 
            print("REACHED MEDIUM DISTANCE")
        elif distance_travelled == 200: 
            print("REACHED HARD DISTANCE")
        
        # Printing the location of the pedestrian 
        # vehicle_location = vehicle.get_location() 
        
        # distance_from_flag = int(vehicle_location.distance(FLAG_LOCATION))
        # distance_from_spawn = int(vehicle_location.distance(spawn_location))
        # distance_between_fs = int(spawn_location.distance(FLAG_LOCATION))
        
        # first_ped_location = firstPedestrian.get_location() 
        # second_ped_location = secondPedestrian.get_location()
        # third_ped_location = thirdPedestrian.get_location() 
        # fourth_ped_location = fourthPedestrian.get_location()
        
        # first_ped_vehicle_distance = int(vehicle_location.distance(first_ped_location))
        # second_ped_vehicle_distance = int(vehicle_location.distance(second_ped_location))
        # third_ped_vehicle_distance = int(vehicle_location.distance(third_ped_location))
        # fourth_ped_vehicle_distance = int(vehicle_location.distance(fourth_ped_location))
        
        # print("\nFIRST PED DISTANCE: ", first_ped_vehicle_distance)op
        # print("SECOND PED DISTANCE: ", second_ped_vehicle_distance)
        # print("THIRD PED DISTANCE: ", third_ped_vehicle_distance)
        # print("FOURTH PED DISTANCE: ", fourth_ped_vehicle_distance, "\n")
        
        # print("\nVEHICLE LOCATION: ", vehicle_location)   
        # print("\nDISTANCE FROM FLAG:", distance_from_flag)     
        # print("DISTANCE FROM SPAWN:", distance_from_spawn)
        # print("DISTANCE BETWEEN SPAWN AND FLAG:", distance_between_fs, "\n")
        
        # control_vehicle(vehicle, flagCollected, spawn_location)
        
        # Getting locations of parking lot corners 
        # blLocation = bottomLeftVehicle.get_location()
        # tlLocation = topLeftVehicle.get_location()
        # trLocation = topRightVehicle.get_location()
        # brLocation = bottomRightVehicle.get_location()
        
        # blTlDistance = int(blLocation.distance(tlLocation))
        # tlTrDistance = int(tlLocation.distance(trLocation))
        # trBrDistance = int(trLocation.distance(brLocation))
        # brLrDistance = int(brLocation.distance(blLocation))
        
        # print("\nBL TL Distance: ", blTlDistance)
        # print("TL TR Distance: ", tlTrDistance)
        # print("TR BR Distance: ", trBrDistance)
        # print("BR LR Distance: ", brLrDistance,"\n")
        
            


finally:
    for actor in actorList: 
        actor.destroy()  
    print("Removed all the actors!")
