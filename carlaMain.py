import carla 
import random 

actorList = []

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    
    client.load_world("Town05")
    
    world = client.get_world() 
    
    # Setting up synchronous mode 
    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.05 
    world.apply_settings(settings)
    
    # Traffic Manager 
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    
    traffic_manager.set_random_device_seed(0)
    random.seed(0)
    
    spectator = world.get_spectator()
    
    blueprint_library= world.get_blueprint_library()
    
    # Spawning a vehicle 
    spawn_points = world.get_map().get_spawn_points()
    
    # Checking all the spawn points 
    '''
    for i, spawn_point in enumerate(spawn_points):
        world.debug.draw_string(spawn_point.location, str(i), life_time = 10q0)
    '''
    
    # Different vehicles 
    # dodge_charger = blueprint_library.filter("vehicle.dodge.charger_2020")[0]
    # ford_mustang = blueprint_library.filter("vehicle.ford.mustang")[0]
    # jeep_wrangler = blueprint_library.filter("vehicle.jeep.wrangler_rubicon")[0]
    # mini_cooper = blueprint_library.filter("vehicle.mini.cooper_s_2021")[0]
    # nissan_patrol = blueprint_library.filter("vehicle.nissan.patrol_2021")[0]
    # truck = blueprint_library.filter("vehicle.carlamotors.carlacola")[0]
    # ambulance = blueprint_library.filter("vehicle.ford.ambulance")[0]
    # sprinter = blueprint_library.filter("vehicle.mercedes.sprinter")[0]
    
    
    
    # Spawning single vehicle
    merc_car = blueprint_library.filter("vehicle.mercedes.coupe_2020")[0]
    # print(merc_car)
    
    '''
    temp = world.try_spawn_actor(merc_car, random.choice(spawn_points))
    if temp is not None: 
        actorList.append(temp)
        
    while True:
        world.tick()
    '''
    
    # Spawning multiple vehicles 
    models = ["dodge", "audi", "mini", "mustang", "nissan", "jeep", "mercedes"]
    blueprints = [] 
    for vehicle in blueprint_library.filter("*vehicle*"):
        if any(model in vehicle.id for model in models):
            blueprints.append(vehicle)
            
    max_vehicles = 20
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
    
    
        
    # temp = world.try_spawn_actor(random.choice(actorList))
    
    while True:
        world.tick()
    
finally:
    for actor in actorList: 
        actor.destroy()
    print("Removed all the actors!")