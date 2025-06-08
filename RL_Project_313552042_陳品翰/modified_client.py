import argparse
import json
import numpy as np
import requests
from stable_baselines3 import TD3
from collections import deque
import gymnasium as gym
import cv2
from stable_baselines3 import PPO
class FrameStackWrapper:
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self):
        self.frames.clear()

    def stack_frames(self, frame):
        if frame.shape != (1, 84, 84):
            raise ValueError(f"Unexpected frame shape: {frame.shape}. Expected (1, 84, 84).")

        frame = frame.squeeze(axis=0)

        if len(self.frames) < self.stack_size:
            while len(self.frames) < self.stack_size:
                self.frames.append(frame)
        else:
            self.frames.append(frame)


        stacked_frames = np.stack(self.frames, axis=0)
        print(f"Stacked frames shape: {stacked_frames.shape}") 
        return stacked_frames

# 加載預訓練模型
model_path = ".//modified_car_model_austra_PPO.zip"
model = PPO.load(model_path)

# 初始化 FrameStackWrapper
frame_stack_wrapper = FrameStackWrapper(stack_size=4)

def connect(agent, url: str = 'http://localhost:5000'):
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)
        print(f"Original observation from server: {obs.shape}, dtype: {obs.dtype}")

        obs_gray = obs[0]
        resized_obs_model = cv2.resize(obs_gray, (84, 84))  
        resized_obs_model = resized_obs_model[np.newaxis, :, :]  


        stacked_obs = agent.frame_stack_wrapper.stack_frames(resized_obs_model)
        print(f"Stacked observation shape for model: {stacked_obs.shape}")

        ####################################################
        if json.loads(response.text).get('terminal', False):
            agent.frame_stack_wrapper.reset()
            print("New episode started.")

        #####################################################
        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = agent.act(stacked_obs)

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break

        result = json.loads(response.text)
        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return

class RLAgent:
    def __init__(self, model, frame_stack_wrapper):
        self.model = model
        self.frame_stack_wrapper = frame_stack_wrapper

    def act(self, stacked_obs):
        #stacked_obs = self.frame_stack_wrapper.stack_frames(observation)
        print(f"Input to model.predict shape: {stacked_obs.shape}") 
        action, _ = self.model.predict(stacked_obs, deterministic=True)
        return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='The url of the server.')
    args = parser.parse_args()



    # Initialize the RL Agent
    

    rl_agent = RLAgent(model=model, frame_stack_wrapper=frame_stack_wrapper)

    connect(rl_agent, url=args.url)
