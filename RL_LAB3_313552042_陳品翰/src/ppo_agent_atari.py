import gym.wrappers
import gym.wrappers
import gym.wrappers.atari_preprocessing
import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym


class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		### TODO ###
		# initialize env
		# self.env = ???
		self.env = gym.make(self.env_id , render_mode = 'rgb_array' , frameskip=1)
		self.env = gym.wrappers.AtariPreprocessing(self.env , screen_size=84 , grayscale_obs=True , frame_skip=4 , 
													noop_max=30 , terminal_on_life_loss=False ,scale_obs=False)
		self.env = gym.wrappers.FrameStack(self.env , num_stack=4)
		#self.env = gym.wrappers.RecordVideo(self.env, 'video_0')
		### TODO ###
		# initialize test_env
		# self.test_env = ???
		self.test_env = gym.make(self.env_id , render_mode = 'rgb_array' , frameskip=1)
		self.test_env = gym.wrappers.AtariPreprocessing(self.test_env , screen_size=84 , grayscale_obs=True , frame_skip=4 ,
														 noop_max=30 , terminal_on_life_loss=False ,scale_obs=False)
		self.test_env = gym.wrappers.FrameStack(self.test_env , num_stack=4)
		#self.test_env.metadata["render_fps"] = 30
		#self.test_env = gym.wrappers.RecordVideo(self.test_env, 'video_1')



		self.net = AtariNet(self.env.action_space.n)
		self.net.to(self.device)
		#self.load('./log/Enduro_release/model_18935849_2186.pth')
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
	def decide_agent_actions(self, observation, eval=False):
		### TODO ###
		# add batch dimension in observation
		# get action, value, logp from net
		self.obs = torch.tensor(np.array(observation) , dtype=torch.float32).to(self.device)
		#print(np.array(observation).shape)
		#self.obs = torch.tensor(np.array(observation), dtype=torch.float32).to(self.device)
		#if(eval == False):
		self.obs = self.obs.unsqueeze(0)  # 將通道維度放在前面
		



		# if eval:
		# 	with torch.no_grad():
		# 		???, ???, ???, _ = self.net(observation, eval=True)
		# else:
		# 	???, ???, ???, _ = self.net(observation)
		if eval:
			with torch.no_grad():
				action , value , logp , entropy = self.net(self.obs , eval=True)
				#print(action)
		else:
			action , value , logp , entropy = self.net(self.obs)
			#print(action)
		
		
		return action.cpu() , value.cpu().detach() , logp.cpu().detach() , entropy.cpu().detach()
		#return NotImplementedError

	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)

				### TODO ###
				# calculate loss and update network
				# ???, ???, ???, ??? = self.net(...)
				ac_train_batch = ac_train_batch.squeeze()
				action , value , logp , entropy = self.net(ob_train_batch , False , ac_train_batch)
				#logp = logp.view(-1,1)
				#adv_train_batch = adv_train_batch.unsqueeze(1)
				logp_pi_train_batch = logp_pi_train_batch.squeeze()
				# calculate policy loss
				# ratio = ???
				# surrogate_loss = ???
				# calculate value loss
				# value_criterion = nn.MSELoss()
				# v_loss = value_criterion(...)
				# calculate total loss
				# loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy
				ratio = torch.exp(logp - logp_pi_train_batch)
				surrogate_loss = torch.mean(-torch.min(ratio * adv_train_batch ,torch.clamp(ratio , 1 - self.clip_epsilon , 1 + self.clip_epsilon) * adv_train_batch))
				value_criterion = nn.MSELoss()
				v_loss = value_criterion(value , return_train_batch) 
				loss = surrogate_loss + self.value_coefficient * v_loss.mean() - self.entropy_coefficient * entropy.mean() ###surrogate可能要.mean v_loss.mean()
				# update network
				# self.optim.zero_grad()
				# loss.backward()
				# nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				# self.optim.step()
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters() , self.max_gradient_norm)
				self.optim.step()
				# total_surrogate_loss += surrogate_loss.item()
				# total_v_loss += v_loss.item()
				# total_entropy += entropy.item()
				# total_loss += loss.item()
				# loss_counter += 1
				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.mean().item()
				total_loss += loss.item()
				loss_counter += 1
		#print("ob_train_batch =" , ob_train_batch.size())
		#print("ac_train_batch =" , ac_train_batch.size())
		#print("return_train_batch =" ,return_train_batch.size())
		#print("adv_train_batch =" ,adv_train_batch.size())
		#print("logp_train_batch =" , logp_pi_train_batch.size())
		#print("logp =" ,logp.size())
		#print("logp_pi_train_batch= ",logp_pi_train_batch)
		#print("logp =" , logp.size())
		#print("logp_pi_train_batch =" , logp_pi_train_batch.size())
		#print("ratio =" , ratio.size())
		#print("adv_train_batch = ",adv_train_batch.size())


		#print("adv_train_batch=",adv_train_batch)

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")
	



