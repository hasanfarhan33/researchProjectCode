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

LEARNING_RATE = 0.001
loadPreviousModel = True 

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

# run = wandb.init(
#     project = "Carla_Research_Project", 
#     config = config, 
#     sync_tensorboard=True, 
# )

# If loading a previous model 
run = wandb.init(project = "Carla_Research_Project", id = "hf9pquse", config = config, sync_tensorboard= True, resume = "must")

    
print("Connecting to environment...")

steeringThrottleEnv = CarEnv() 
steeringThrottleBrakeEnv = CarEnvBrake()
highwayEnv = HighwayEnvironment() 
highwayNoBrakeEnv = HighwayEnvironmentNoBrakes() 

if loadPreviousModel:
    timestepNumber = 0 
    model = PPO.load("./models/hf9pquse/model.zip", device = "cuda", env = highwayNoBrakeEnv)
else:
    model = PPO(config["policy_type"], highwayNoBrakeEnv, verbose = 1, learning_rate = LEARNING_RATE, 
                tensorboard_log=f"runs/{run.id}", device = "cuda")

TIMESTEPS = 500_000 
iters = 0 

while iters < 2: 
    iters += 1 
    print("Iteration ", iters, " is to commence...")
    model.learn(total_timesteps = TIMESTEPS, 
                callback=WandbCallback(gradient_save_freq=100, model_save_path = f"models/{run.id}", 
                                       model_save_freq=200, verbose = 2), reset_num_timesteps=False, log_interval = 4)
    print("Iteration ", iters, " has been trained...")

run.finish() 
    
    