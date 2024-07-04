from stable_baselines3 import PPO 
from typing import Callable
import os 
from carEnv import CarEnv 
import time 
import wandb
from wandb.integration.sb3 import WandbCallback
from carEnvBrake import CarEnvBrake
from highwayEnvironment import HighwayEnvironment
from highwayEnvironmentNoBrakes import HighwayEnvironmentNoBrakes
from highwayEnvPedestrian import HighwayEnvPedestrian
from highwayEnvMultiPedestrian import HighwayEnvMultiPedestrian
from highwayFlagEnv import HighwayFlagEnv
from parkingLotEnv import ParkingLotEnv

LEARNING_RATE = 0.001
loadPreviousModel = False

# modelName = "Jimothy"
# modelsDirectory = f"models/{modelName}"
# logDirectory = f"logs/{modelName}"

# if not os.path.exists(modelsDirectory): 
#     os.makedirs(modelsDirectory)
    
# if not os.path.exists(logDirectory): 
#     os.makedirs(logDirectory)

config = {
    "policy_type":"MlpPolicy", 
}

# IF TRAINING A NEW MODEL
run = wandb.init(
        project = "Carla_Research_Project", 
        config = config, 
        sync_tensorboard=True, 
        name="PPO-parkingLotSlow3rd"
    )

# If you want to load a previous model 
# run = wandb.init(project = "Carla_Research_Project", id = "p06s3sem", config = config, sync_tensorboard= True, resume = "must")
    

if wandb.run.resumed: 
    loadPreviousModel = True 
else: 
    loadPreviousModel = False 

print("Connecting to environment...")

# steeringThrottleEnv = CarEnv() 
# steeringThrottleBrakeEnv = CarEnvBrake()
# highwayEnv = HighwayEnvironment() 
# highwayNoBrakeEnv = HighwayEnvironmentNoBrakes()
# pedestrianEnv = HighwayEnvPedestrian() 
# highwayMultiPedEnv = HighwayEnvMultiPedestrian()
# flagEnvHighway = HighwayFlagEnv() 
parkingLot = ParkingLotEnv() 

if loadPreviousModel:
    timestepNumber = 0 
    model = PPO.load("./models/p06s3sem/model.zip", device = "cuda", env = parkingLot)
else:
    model = PPO(config["policy_type"], parkingLot, verbose = 1, learning_rate = LEARNING_RATE, 
                tensorboard_log=f"runs/{run.id}", device = "cuda")

TIMESTEPS = 250_000
iters = 0 

while iters < 2: 
    iters += 1 
    print(loadPreviousModel)
    print("Iteration ", iters, " is to commence...")
    model.learn(total_timesteps = TIMESTEPS, 
                callback=WandbCallback(gradient_save_freq=100_000, model_save_path = f"models/{run.id}", 
                                       model_save_freq=100_000, verbose = 2), reset_num_timesteps=False, log_interval = 4)
    print("Iteration ", iters, " has been trained...")

run.finish() 
    
    