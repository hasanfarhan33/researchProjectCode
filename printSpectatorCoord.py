# THIS FILE IS USED TO EASILY GET THE COORDINATES ON THE MAP

import time 
import carla 

HOST = '127.0.0.1'
PORT = 2000 
SLEEP_TIME = 1 

try: 
    client = carla.Client(HOST, PORT)
    client.set_timeout(150.0)
    
    client.load_world("Town05_Opt")
    world = client.get_world()
    
    while True: 
        t = world.get_spectator().get_transform() 
        coordinate_str = ("(x, y, z) = ({},{},{})".format(t.location.x, t.location.y, t.location.z))
        print(coordinate_str)
        time.sleep(SLEEP_TIME)
finally: 
    pass 

