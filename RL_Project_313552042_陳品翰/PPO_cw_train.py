import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise , OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import SubprocVecEnv , make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from racecar_gym.cnn_mod_circle_env import RaceEnv
from gymnasium.wrappers import FrameStack






def create_env(scenario):
    def _init():
        print(f"Initializing environment for scenario: {scenario}")
        base_env = RaceEnv(scenario = scenario , reset_when_collision = False)
        stacked_env = FrameStack(base_env, num_stack=4)
        print(f"Environment after FrameStack: {type(stacked_env)}")
        print(f"Environment initialized: {type(stacked_env)}")
        return stacked_env
    return _init

if __name__ == '__main__':
    num_envs = 4
    print("Starting SubprocVecEnv initialization...")
    vec_env = SubprocVecEnv([create_env("circle_cw_competition_collisionStop") for i in range(num_envs)])
    print("SubprocVecEnv initialized successfully.")
    #print("Starting DummyVecEnv initialization...")
    #vec_env = DummyVecEnv([create_env("austria_competition") for i in range(num_envs)])
    #print("DummyVecEnv initialized successfully.")
    model_path = "./modified_car_model_circle_PPO_1.zip"
    model = PPO.load(model_path , env = vec_env)

    model.set_env(vec_env)


    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean = np.zeros(n_actions) , sigma = 0.1 * np.ones(n_actions))

    model.learning_rate = 2 * (1e-4)
    print(f"Current learning rate: {model.learning_rate}")
    model.learn(total_timesteps = 0.8 * 1e7 , log_interval = 10 ,  progress_bar = True)
    model.save("modified_car_model_circle_PPO_final")

