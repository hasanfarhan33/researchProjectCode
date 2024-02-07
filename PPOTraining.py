from stable_baselines3 import PPO 
from typing import Callable
import os 
from carEnv import CarEnv 
import time 

LEARNING_RATE = 0.001
loadPreviousModel = False 

modelName = "Jimothy"
modelsDirectory = f"models/{modelName}"
logDirectory = f"logs/{modelName}"

if not os.path.exists(modelsDirectory): 
    os.makedirs(modelsDirectory)
    
if not os.path.exists(logDirectory): 
    os.makedirs(logDirectory)
    
print("Connecting to environment...")

env = CarEnv() 

if loadPreviousModel:
    timestepNumber = 0 
    model = PPO.load(f"{modelsDirectory}/{timestepNumber}", device = "cuda")
else:
    model = PPO("MlpPolicy", env, verbose = 1, learning_rate = LEARNING_RATE, tensorboard_log=logDirectory, device = "cuda")

TIMESTEPS = 500_000 
iters = 0 

while iters < 3: 
    iters += 1 
    print("Iteration ", iters, " is to commence...")
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"Jimothy", log_interval = 4)
    print("Iteration ", iters, " has been trained...")
    model.save(f"{modelsDirectory}/{TIMESTEPS*iters}")
    