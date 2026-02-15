import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. The Hyperparameters (Start with these)
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
EPOCHS = 10        # How many times to reuse the dataset
BATCH_SIZE = 64    # Mini-batch size
ROLLOUT_LEN = 2048 # How much data to collect before updating

# 2. The Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # TODO: Define your Actor layers (output: action logits)
        # TODO: Define your Critic layers (output: single value)
        pass

    def forward(self):
        raise NotImplementedError("Use get_action and get_value instead")

    def get_action(self, state):
        # TODO: 
        # 1. Pass state through Actor
        # 2. Create a distribution (Categorical)
        # 3. Sample an action
        # 4. Return action and its log_prob
        pass

    def get_value(self, state):
        # TODO: Pass state through Critic
        pass

# 3. The PPO Agent
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.policy = ActorCritic(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.buffer = [] # Store your rollouts here

    def collect_rollout(self):
        # TODO: Run the agent in the env for ROLLOUT_LEN steps
        # Store (state, action, log_prob, reward, done, value)
        pass

    def compute_gae(self):
        # TODO: Implement GAE calculation
        pass

    def update(self):
        # TODO: 
        # 1. Calculate Advantages
        # 2. Loop for EPOCHS:
        #    a. Calculate Ratio (pi_new / pi_old)
        #    b. Calculate Surrogate Loss (Clip logic)
        #    c. Calculate Value Loss (MSE)
        #    d. Backprop
        pass

# 4. Main Training Loop
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = PPOAgent(env)
    
    for i in range(1000): # Number of updates
        agent.collect_rollout()
        agent.update()
        # TODO: Log average reward to see if it's learning!
