from carEnv import CarEnv 
from carEnvBrake import CarEnvBrake 
import gymnasium as gym  

gym.register(id = "stCarEnv", entry_point = "gym.envs:CarEnv", max_episode_steps=500_000)
gym.register(id = "stbCarEnv", entry_point = "gym.envs:CarEnvBrake", max_episode_steps=500_000)