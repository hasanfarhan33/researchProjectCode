from stable_baselines3.common.env_checker import check_env
from carEnv import CarEnv
from highwayEnvMultiPedestrian import HighwayEnvMultiPedestrian 
from highwayEnvPedestrian import HighwayEnvPedestrian
from highwayFlagEnv import HighwayFlagEnv

# env = CarEnv()
env = HighwayFlagEnv()
episodes = 50

for episode in range(episodes):
    done = False 
    obs = env.reset() 
    while True: 
        random_action = env.action_space.sample()
        print("\nAction: ", random_action)
        observation, reward, terminated, truncated, info = env.step(random_action)
        print("Reward: ", reward)
        print("Terminated: ", terminated)
        print("Truncated: ", truncated)
        print("Info: ", info)
        
# It will check your custom environment and output additional warnings if needed
check_env(env)