import gymnasium as gym
from sympy.strategies.traverse import do_one
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange

# 1. The Hyperparameters (Start with these)
LR = 3e-4
DISCOUNT = 0.99
GAE_PARAM = 0.95
EPS_CLIP = 0.2
EPOCHS = 10             # How many times to reuse the dataset
BATCH_SIZE = 64         # Mini-batch size
ROLLOUT_LEN = 2048      # How much data to collect before updating
NUM_UPDATES = 50        # No. of policy update loops
# total timesteps = rollout len * num_updates = 100K (enough for cart-pole)
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


# 2. The Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        print(f"Using device: {DEVICE}")

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        ).to(DEVICE)   # output logits for actions

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(DEVICE)   # output value score
        

    def forward(self, state):
        action, log_prob = self.get_action(state)
        value = self.get_value(state)
        return action, log_prob, value

    def get_action(self, state):
        # 1. Pass state through Actor
        logits = self.actor(state) # (,2)
        # 2. Create a distribution (Categorical)
        # https://docs.pytorch.org/docs/stable/distributions.html#categorical
        # Why Categorical? It is not possible to back prop through random samples.
        # REINFORCE is a trick for creating surrogate function that can be backproped.
        # This is commonly used for policy-gradient methods in RL.
        dist = torch.distributions.Categorical(logits=logits)
        # 3. Sample an action
        action = dist.sample()  # sampled index (,1)
        log_prob = dist.log_prob(action)    # (, 1)
        # 4. Return action and its log_prob
        return action, log_prob

    def get_log_prob(self, state, action):
        # Get log_prob given state for specific action
        logits = self.actor(state)  # (, 2)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        return log_prob
    
    def get_value(self, state):
        value = self.critic(state)
        return value

# 3. The PPO Agent
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.policy = ActorCritic(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.buffer = [] # Store your rollouts here

    def collect_rollout(self):
        """
        Use "old" / "current" policy to sample fresh data for training. 
        On-policy learning.
        """
        self.buffer = []
        state, info = self.env.reset()
        episode_reward = 0
        for i in range(ROLLOUT_LEN):
            with torch.no_grad():
                # Note: training data is constant, even though generated with policy. 
                # So no_grad applied. (hard bug to catch)
                action, log_prob, value = self.policy(torch.Tensor(state).to(DEVICE))
            new_state, reward, done, truncated, info = self.env.step(action.item())
            self.buffer.append(
                (state, new_state, action.detach(), log_prob.detach(), value.detach(), reward, done)
            )
            state = new_state
            episode_reward += reward
            if done or truncated:
                state, info = self.env.reset()
                episode_reward = 0
    
    def compute_gae(self):
        # Calculate GAE calculation for all timestep in rollout
        # Lookahead till end of episode or end of buffer recursively
        gae_buffer = []                # gae for last timestep in buffer, since we can't see future
        future_rewards_buffer = []
        gae = 0
        future_reward = 0
        value_next = 0
        # init last timestamp if not done
        last_done = self.buffer[-1][6]
        if not last_done:
            # end of buffer is not done state, so we "assume" future values using value model
            next_state = self.buffer[-1][1]
            value_next = self.policy.get_value(torch.Tensor(next_state).to(DEVICE)).item()
            future_reward = value_next      # target for critic training. TODO verify if right approach???

        for i in reversed(range(len(self.buffer))): # start reversed
            b = self.buffer[i]
            value, reward, done = b[4], b[5], b[6]
            if done:  # end of episode
                # reset
                value_next = 0
                gae = 0
                future_reward = 0

            delta = (reward + DISCOUNT * value_next) - value    # TD error
            gae = delta + (DISCOUNT * GAE_PARAM * gae)  # recursive formula
            future_reward += reward                    # future rewards for training critic
            gae_buffer.append(gae)
            future_rewards_buffer.append(future_reward)
            value_next = value

        gae_buffer.reverse()
        future_rewards_buffer.reverse()
        return gae_buffer, future_rewards_buffer

    def _iter_minibatches(self, dataset, batch_size):
        total_size = dataset["states"].shape[0]
        indices = torch.randperm(total_size, device=DEVICE)
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_idx = indices[start:end]
            yield {key: value[batch_idx] for key, value in dataset.items()}

    def _build_dataset(self):
        advantages, future_rewards = self.compute_gae()
        return {
            "states": torch.tensor(
                np.array([b[0] for b in self.buffer]), dtype=torch.float32, device=DEVICE
            ),
            "actions": torch.stack([b[2] for b in self.buffer]).to(DEVICE),
            "log_prob": torch.stack([b[3] for b in self.buffer]).to(DEVICE).detach(),
            "advantage": torch.tensor(
                [a.item() if torch.is_tensor(a) else a for a in advantages],
                dtype=torch.float32,
                device=DEVICE,
            ),
            "future_rewards": torch.tensor(
                [r.item() if torch.is_tensor(r) else r for r in future_rewards],
                dtype=torch.float32,
                device=DEVICE,
            ),
        }

    def update(self):
        """
        Learning loop / backprop
        """
        dataset = self._build_dataset()     # rollouts, GAE, rewards, etc.

        # TRAINING LOOP
        for i in range(EPOCHS):
            for batch in self._iter_minibatches(dataset, BATCH_SIZE):
                self.optimizer.zero_grad()
                adv = batch['advantage']
                states = batch['states']
                actions = batch['actions']
                log_prob_old = batch['log_prob']              # from old policy rollouts
                future_rewards = batch['future_rewards']

                # on-policy update (sample log_probs, values from latest policy for past actions)
                log_prob_new = self.policy.get_log_prob(states, actions) # (, 1)
                values_new = self.policy.get_value(states)  
                # PPO
                ratio = torch.exp(
                    log_prob_new - log_prob_old
                )
                unclipped = ratio * adv
                clipped = torch.clip(
                    ratio, 1 - EPS_CLIP, 1 + EPS_CLIP
                ) * adv
                # clipped surrogate loss -- PPO loss
                actor_loss = - torch.min(unclipped, clipped).mean()
                # MSE
                critic_loss = torch.square(values_new - future_rewards).mean()
                # combined loss to backprop both actor and critic
                loss = actor_loss + critic_loss 
                loss.backward()
                self.optimizer.step()


# 4. Main Training Loop
def train(mode=None):
    env = gym.make("CartPole-v1", render_mode=mode)
    agent = PPOAgent(env)

    # train loop
    for i in trange(NUM_UPDATES, desc="Update loop", unit="epoch"): # Number of updates
        agent.collect_rollout()
        agent.update()
        # TODO: Log average reward to see if it's learning!
        evaluate(agent)

    return agent


def evaluate(agent, mode=None):
    EVAL_EPISODES = 100
    env = gym.make("CartPole-v1", render_mode=mode)
    episode_rewards = []
    for i in trange(EVAL_EPISODES, desc="Evaluating", unit="episode"):
        state, info = env.reset()
        episode_reward = 0
        for j in range(ROLLOUT_LEN):
            action, log_prob, value = agent.policy(torch.Tensor(state).to(DEVICE))
            state, reward, done, truncated, info = env.step(action.item())
            episode_reward += reward
            if done or truncated:
                break
        episode_rewards.append(episode_reward)
    env.close()
    print(f"Average reward for {EVAL_EPISODES} episodes: {sum(episode_rewards) / EVAL_EPISODES}")
    return episode_rewards


if __name__ == "__main__":
    print("Train PPO agent...")
    agent = train(mode=None)

    print("PPO Final agent eval...")
    evaluate(agent, mode="human")
