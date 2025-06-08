import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import imageio
import os
import argparse

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(4, 128),  
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    
    def forward(self, x):
        return self.network(x)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    
    
    num_actions = env.action_space.n
    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    final_reward = 0
    os.makedirs(args.output_dir, exist_ok=True)
    
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0
        frames = []
        
        while not done:
            
            frame = env.render()
            frames.append(frame)
            
            
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs
        
        
        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                video.append_data(f)
        
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")
        final_reward += total_reward
    
    average_score = final_reward / args.episodes
    print(f"average score: {average_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pth model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos_demo")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=313551076, help="Random seed for evaluation")
    args = parser.parse_args()
    evaluate(args)