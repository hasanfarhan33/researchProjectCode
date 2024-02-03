import gymnasium as gym 
import numpy as np 
from stable_baselines3 import PPO 
from stable_baselines3.ppo import CnnPolicy 
from stable_baselines3.common.noise import NormalActionNoise 
from carlaEnv import CarEnvironment 
import sys 
import argparse 

def main(model_name, load_model, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, 
         enable_preview, steps_per_episode, seed = 7, action_type = "continuous"):
    
    env = CarEnvironment(town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, action_type, enable_preview, steps_per_episode, 
                         playing = False)
    # test_env = CarEnvironment(town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, action_type, 
    #                           enable_preview=True, steps_per_episode=steps_per_episode, playing=True)
    
    try: 
        if load_model: 
            model = PPO.load(model_name, env, action_noise = NormalActionNoise(mean=np.array([0.3, 0.0]), sigma=np.array([0.5, 0.1])))
        else:
            model = PPO(CnnPolicy, env, verbose=2, seed = seed, device = "cuda", 
                        tensorboard_log="./jimothyLog", action_noise = NormalActionNoise(mean=np.array([0.3, 0]), sigma=np.array([0.5, 0.1])))
        
        print(model.__dict__)
        model.learn(
            total_timesteps=10000, 
            log_interval=4, 
            tb_log_name=model_name, 
            reset_num_timesteps=False
            )
        model.save(model_name)
    finally:
        env.close() 
        # test_env.close() 
        

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', help='name of the model when saving')
    parser.add_argument('--load', type = bool, help = "whether to load an existing model")
    parser.add_argument('--map', type = str, default = "Town04", help = "name of Carla map")
    parser.add_argument('--fps', type = int, default=10, help = "fps (frames per second) of carla environment")
    parser.add_argument('--width', type = int, help = "width of the camera window")
    parser.add_argument('--height', type = int, help = "height of the camera window")
    parser.add_argument('--repeat-action', type = int, help = "number of steps to repeat each action")
    parser.add_argument('--start-location', type = str, help = "start location type")
    parser.add_argument('--sensor', type = "str", action = 'append', help = "type of sensor (can be multiple): [rgb, semantic]")
    parser.add_argument('--preview', action="store_true", help = "whether to enable preview camera")
    parser.add_argument('--episode-length', type = int, help = "maximum number of steps per each episode")
    parser.add_argument('--seed', type = int, default=7, help = "random seed for initialization")
    
    args = parser.parse_args() 
    model_name = args.model_name 
    load_model = args.load 
    town = args.map 
    fps = args.fps 
    im_width = args.width 
    im_height = args.height 
    repeat_action = args.repeat_action
    start_transform_type = args.start_location
    sensors = args.sensor
    enable_preview = args.preview
    steps_per_episode = args.episode_length
    seed = args.seed
    
    main(model_name, load_model, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, enable_preview, steps_per_episode, seed)    