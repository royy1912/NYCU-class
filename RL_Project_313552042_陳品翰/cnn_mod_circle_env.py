from collections import OrderedDict
from collections import deque
import gymnasium as gym
import numpy as np
from numpy import array, float32
# noinspection PyUnresolvedReferences
import racecar_gym.envs.gym_api
import cv2

class RaceEnv(gym.Env):
    camera_name = 'camera_competition'
    motor_name = 'motor_competition'
    steering_name = 'steering_competition'
    """The environment wrapper for RaceCarGym.
    
    - scenario: str, the name of the scenario.
        'austria_competition' or
        'plechaty_competition'
    
    Notes
    -----
    - Assume there are only two actions: motor and steering.
    - Assume the observation is the camera value.
    """

    def __init__(self,
                 scenario: str,
                 render_mode: str = 'rgb_array_birds_eye',
                 reset_when_collision: bool = True,
                 **kwargs):
        self.scenario = scenario.upper()[0] + scenario.lower()[1:]
        self.env_id = f'SingleAgent{self.scenario}-v0'
        self.env = gym.make(id=self.env_id,
                            render_mode=render_mode,
                            reset_when_collision=reset_when_collision,
                            **kwargs)
        self.render_mode = render_mode
        # Assume actions only include: motor and steering
        self.action_space = gym.spaces.box.Box(low=-1., high=1., shape=(2,), dtype=float32)
        # Assume observation is the camera value
        # noinspection PyUnresolvedReferences
        observation_spaces = {k: v for k, v in self.env.observation_space.items()}
        assert self.camera_name in observation_spaces, f'One of the sensors must be {self.camera_name}. Check the scenario file.'
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
        #
        self.cur_step = 0
        self.pre_progress = 0

    def observation_postprocess(self, obs):
        # 提取圖像數據
        img = obs[self.camera_name].astype(np.uint8) 
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized_img = cv2.resize(gray_img, (84, 84))
        return resized_img

    def reset(self, *args, **kwargs: dict):
        if kwargs.get('options'):
            kwargs['options']['mode'] = 'random'
        else:
            kwargs['options'] = {'mode' : 'random'}
        self.cur_step = 0
        obs, *others = self.env.reset(*args, **kwargs)
        obs = self.observation_postprocess(obs)
        #print(f"Reset observation shape: {obs.shape}")
        return obs, *others

    def step(self, actions):
        self.cur_step += 1
        motor_action, steering_action = actions

        # Add a small noise and clip the actions
        motor_scale = 0.001
        steering_scale = 0.01
        motor_action = np.clip(motor_action + np.random.normal(scale=motor_scale), -1., 1.)
        steering_action = np.clip(steering_action + np.random.normal(scale=steering_scale), -1., 1.)


        dict_actions = OrderedDict([(self.motor_name, array(motor_action, dtype=float32)),
                                    (self.steering_name, array(steering_action, dtype=float32))])
        
        #obs, *others = self.env.step(dict_actions)
        obs , reward , done , truncated ,  info = self.env.step(dict_actions)
        obs = self.observation_postprocess(obs)
        reward = self.compute_reward(obs , dict_actions , reward , info)
        
        
        return obs, reward , done , truncated , info


    def compute_reward(self, obs, actions, base_reward, info):
        """
        Compute the custom reward for the environment.
    
        Parameters:
        - obs: The current observation.
        - actions: The action taken.
        - base_reward: The base reward provided by the environment.
        - info: Additional info from the environment.

        Returns:
        - reward: The computed reward.
        """
        action_penalty_1 = -0.07 * np.abs(actions[self.motor_name])
        action_penalty_2 = -0.1 * np.abs(actions[self.steering_name]) 
        progress = info.get("progress")  

        # Example collision penalty
        collision_penalty = -1 if info.get("wall_collision", False) else 0
        if(progress == self.pre_progress):
            progress_reward = -0.5
        else:
            progress_reward = progress - self.pre_progress
        # Combine rewards
        reward = base_reward + progress_reward  + collision_penalty + action_penalty_1 + action_penalty_2
        self.pre_progress = progress
        return reward

    def render(self):
        return self.env.render()
    

