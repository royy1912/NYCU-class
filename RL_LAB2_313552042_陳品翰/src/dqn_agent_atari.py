import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
from gym.wrappers import FrameStack
import random

class AtariDQNAgent(DQNBaseAgent):
    def __init__(self, config):
        super(AtariDQNAgent, self).__init__(config)
        ### TODO ###
        # initialize env
        # self.env = ???
        #, mode = 'rgb_array'
        self.env = gym.make('ALE/MsPacman-v5', render_mode = 'rgb_array')
        #self.env.metadata['render_fps'] = 30  # Set to your desired FPS, e.g., 30
        self.env = gym.wrappers.RecordVideo(self.env, 'env_video',episode_trigger = self.video_trigger)
        self.env = gym.wrappers.AtariPreprocessing(self.env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
        self.env = gym.wrappers.FrameStack(self.env, 4)
        ### TODO ###
        # initialize test_env
        # self.test_env = ???
        self.test_env = gym.make('ALE/MsPacman-v5', render_mode = 'rgb_array')
        #self.test_env.metadata['render_fps'] = 30
        self.test_env = gym.wrappers.RecordVideo(self.test_env, 'test_env_video',episode_trigger =self.video_trigger)
        self.test_env = gym.wrappers.AtariPreprocessing(self.test_env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
        self.test_env = gym.wrappers.FrameStack(self.test_env, 4)
    
        # initialize behavior network and target network
        self.behavior_net = AtariNetDQN(self.env.action_space.n)
        self.behavior_net.to(self.device)
        #self.load_and_evaluate('./log/DQN/Enduro/model_9818485_3628.pth')
        self.target_net = AtariNetDQN(self.env.action_space.n)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        # initialize optimizer
        self.lr = config["learning_rate"]
        self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
        
    def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
        ### TODO ###
        # get action from behavior net, with epsilon-greedy selection
        
        # if random.random() < epsilon:
        #   action = ???
        # else:
        #   action = ???

        # return action
        if np.random.random() < epsilon:
            action = action_space.sample()
        else:
            array_observation = np.array(observation)
            # Convert the observation to a PyTorch tensor and add a batch dimension
            tensor_observation = torch.IntTensor(array_observation).unsqueeze(0).to(self.device)

            #with torch.no_grad():
            action_val = self.behavior_net(tensor_observation)
            action = action_val.max(1)[1].item()
                #action = torch.max(action_val, 1)[1].item()
                
        return action

    
    def update_behavior_network(self):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

        ### TODO ###
        # calculate the loss and update the behavior network
        # 1. get Q(s,a) from behavior net
        # 2. get max_a Q(s',a) from target net
        # 3. calculate Q_target = r + gamma * max_a Q(s',a)
        # 4. calculate loss between Q(s,a) and Q_target
        # 5. update behavior net
        action = action.to(torch.int64)
        q_value = self.behavior_net(state).gather(1, action)

        with torch.no_grad():
            q_next = self.target_net(next_state).max(1)[0].unsqueeze(1)
        q_target = torch.zeros( self.batch_size , 1 , dtype = torch.float32).to(self.device)
        for i in range (self.batch_size):
            if done[i]:
                q_target[i][0] = reward[i]
            else:
                q_target[i][0] = reward[i] + self.gamma * q_next[i]
        criterion = torch.nn.MSELoss()
        loss = criterion(q_value, q_target)
        # q_value = ???
        # with torch.no_grad():
            # q_next = ???

            # if episode terminates at next_state, then q_target = reward
            # q_target = ???
        
        
        # criterion = ???
        # loss = criterion(q_value, q_target)

        self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
