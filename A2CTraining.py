from stable_baselines3 import A2C
from typing import Callable
import os 
from carEnv import CarEnv 
from carEnvBrake import CarEnvBrake
import time 
import wandb
from wandb.integration.sb3 import WandbCallback
import gymnasium as gym 

LEARNING_RATE = 0.001
loadPreviousModel = False 

config = {
    "policy_type":"MlpPolicy", 
}

run = wandb.init(project = "Carla_Research_Project", id = "ln4bk548", config = config, sync_tensorboard= True, resume = "must")

if wandb.run.resumed: 
    loadPreviousModel = True 
else: 
    loadPreviousModel = False 

    
print("Connecting to environment...")

steeringThrottleEnv = CarEnv() 
# steeringThrottleEnv = gym.make("stCarEnv")
steeringThrottleBrakeEnv = CarEnvBrake() 
# steeringThrottleBrakeEnv = gym.make("stbCarEnv")

if loadPreviousModel:
    # TODO: Figure out how to load previously trained models 
    timestepNumber = 0 
    # model = PPO.load(f"{modelsDirectory}/{timestepNumber}", device = "cuda", env)
    model = A2C.load("./models/ln4bk548/model.zip", env = steeringThrottleBrakeEnv)
else:
    model = A2C(config["policy_type"], CarEnvBrake, verbose = 1, learning_rate = LEARNING_RATE, tensorboard_log=f"runs/{run.id}", device = "cuda")

TIMESTEPS = 500_000 
iters = 0 

while iters < 3: 
    iters += 1 
    print("Iteration ", iters, " is to commence...")
    model.learn(total_timesteps = TIMESTEPS, callback=WandbCallback(gradient_save_freq=100, model_save_path = f"models/{run.id}", model_save_freq=200, verbose = 2), reset_num_timesteps=False, log_interval = 4)    
    print("Iteration ", iters, " has been trained...")

run.finish() 