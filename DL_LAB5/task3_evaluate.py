import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse


class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
    
    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized
    
    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)
    
    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked


def evaluate_checkpoint(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    
    # Initialize preprocessor
    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n
    
    # Create model
    q_net = DQN(4, num_actions).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    q_net.load_state_dict(checkpoint['q_net_state_dict'])
    q_net.eval()
    
    # Display checkpoint info
    print(f"Loaded checkpoint from episode {checkpoint.get('episode', 'unknown')}")
    print(f"Epsilon at checkpoint: {checkpoint.get('epsilon', 'unknown')}")
    print(f"Environment steps: {checkpoint.get('env_count', 'unknown')}")
    print(f"Training steps: {checkpoint.get('train_count', 'unknown')}")
    print(f"Best reward: {checkpoint.get('best_reward', 'unknown')}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a summary file
    summary_path = os.path.join(args.output_dir, "checkpoint_evaluation_summary.txt")
    
    final_reward = 0
    
    with open(summary_path, "w") as summary_file:
        summary_file.write(f"Checkpoint Evaluation Summary\n")
        summary_file.write(f"===========================\n")
        summary_file.write(f"Checkpoint: {args.checkpoint_path}\n")
        summary_file.write(f"Number of episodes: {args.episodes}\n")
        summary_file.write(f"Seed: {args.seed}\n\n")
        
        if 'episode' in checkpoint:
            summary_file.write(f"Checkpoint from episode: {checkpoint['episode']}\n")
        if 'epsilon' in checkpoint:
            summary_file.write(f"Epsilon at checkpoint: {checkpoint['epsilon']}\n")
        if 'best_reward' in checkpoint:
            summary_file.write(f"Best reward at checkpoint: {checkpoint['best_reward']}\n\n")
            
        summary_file.write(f"Episode Results:\n")
        
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            state = preprocessor.reset(obs)
            done = False
            total_reward = 0
            frames = []
            frame_idx = 0
            
            while not done:
                frame = env.render()
                frames.append(frame)
                
                # Model expects normalized input
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device) / 255.0
                with torch.no_grad():
                    action = q_net(state_tensor).argmax().item()
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = preprocessor.step(next_obs)
                frame_idx += 1
            
            # Save video
            if args.save_videos:
                out_path = os.path.join(args.output_dir, f"eval_ep{ep}_reward_{int(total_reward)}.mp4")
                with imageio.get_writer(out_path, fps=30) as video:
                    for f in frames:
                        video.append_data(f)
                print(f"  → Video saved: {out_path}")
            
            print(f"Episode {ep}: Total reward = {total_reward}")
            summary_file.write(f"Episode {ep}: {total_reward}\n")
            final_reward += total_reward
        
        average_score = final_reward / args.episodes
        print(f"\nAverage score over {args.episodes} episodes: {average_score}")
        
        summary_file.write(f"\nAverage score: {average_score}\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a DDQN+PER checkpoint on Atari Pong")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to checkpoint file (.pt)")
    parser.add_argument("--output-dir", type=str, default="./task3_600K_handin", help="Directory to save videos and summary")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    parser.add_argument("--save-videos", action="store_true", help="Save gameplay videos")
    args = parser.parse_args()
    
    evaluate_checkpoint(args)