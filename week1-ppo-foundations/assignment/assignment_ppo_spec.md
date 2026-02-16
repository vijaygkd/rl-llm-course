# Week 1 Coding Assignment: PPO from Scratch

**Objective:** Implement the Proximal Policy Optimization (PPO) algorithm in raw PyTorch to solve the `CartPole-v1` control environment.

## 1. The Challenge

You must train an Agent to balance a pole on a cart for 500 steps (the maximum for v1).

* **Input:** A vector of 4 floats (Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity).
* **Output:** A discrete action (0: Push Left, 1: Push Right).
* **Success Metric:** The agent achieves an average reward of **475+** over 100 consecutive episodes.

## 2. Constraints & Rules

* **Allowed Libraries:** `torch`, `numpy`, `gymnasium` (or `gym`), `matplotlib`.
* **Forbidden Libraries:** `stable-baselines3`, `rllib`, `ray`, `huggingface trl` (We are building *the* wheel, not using one).
* **Architecture:** You must implement an **Actor-Critic** architecture (shared or separate networks).

## 3. Implementation Steps

### Step 1: The Networks (`models.py`)

Create a PyTorch `nn.Module` class (or two) that serves as your Actor and Critic.

* **Actor (Policy):** Takes `state` (4 dims)  Returns `logits` for actions (2 dims). Use `Categorical` distribution to sample.
* **Critic (Value):** Takes `state` (4 dims)  Returns a single scalar `value` (1 dim).
* *Hint:* Simple MLPs (e.g., 2 layers of 64 neurons with Tanh or ReLU activation) are sufficient.

### Step 2: The Rollout Buffer

PPO is an *on-policy* algorithm. You need to collect a batch of data using your *current* policy before you can update.

* **Collection:** Run the agent in the environment for a fixed number of steps (e.g., 2048).
* **Store:** `states`, `actions`, `log_probs`, `rewards`, `is_terminals`, and `values`.
* *Why store log_probs?* You need the probability of the action *at the time it was taken* to calculate the ratio $r_t(\theta)$ later.

### Step 3: Generalized Advantage Estimation (GAE)
You cannot just use raw rewards. You need to calculate **Advantages** ($\hat{A}_t$) to reduce variance.

* **Implement the GAE formula:**
    $$A_t = \delta_t + (\gamma \lambda) A_{t+1}$$
    Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ (The TD Error).
* **Hyperparameters:**
    * $\gamma = 0.99$ (Discount factor)
    * $\lambda = 0.95$ (GAE smoothing parameter)

### Step 4: The PPO Loss (The Core)
Implement the "Clipped Surrogate Objective" in your update loop.

1.  **Calculate Ratio:**
    $$r_t(\theta) = \frac{\pi_{new}(a|s)}{\pi_{old}(a|s)} = \exp(\log\pi_{new} - \log\pi_{old})$$
2.  **Calculate Surrogate 1:**
    $$\text{Surr}_1 = r_t \cdot A_t$$
3.  **Calculate Surrogate 2:**
    $$\text{Surr}_2 = \text{clamp}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t$$
4.  **Final Actor Loss:**
    $$\mathcal{L}_{CLIP} = -\min(\text{Surr}_1, \text{Surr}_2).\text{mean}()$$
5.  **Critic Loss:**
    MSE between `new_values` and `returns` (where `returns` = $A_t + V_{old}$).

## 4. Suggested File Structure

```text
week1/
├── assignment_ppo_spec.md  <-- You are here
├── ppo_cartpole.py         <-- Your main implementation (can be a script or a class)
├── utils.py                <-- Optional: Helper functions for GAE or plotting
└── README.md               <-- Log your results (plots of reward vs. episodes)

```

## 5. Starter Snippet (Environment Setup)

Use this to get your gym environment running correctly:

```python
import gymnasium as gym
import torch

# Create the environment
env = gym.make("CartPole-v1")
state, info = env.reset()

# Hyperparameters to start with
HYPERPARAMS = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "ent_coef": 0.01,  # Entropy coefficient to encourage exploration
    "vf_coef": 0.5,    # Value function coefficient
    "batch_size": 64,
    "total_timesteps": 100_000 # enough to solve CartPole
}

```

---

### **What Now?**

1. **Save this file.**
2. **Go read the paper.** (Seriously, don't start coding the loss function until you can answer the clipping question).
3. **Ping me** when you think you have the answer to: *"What happens to the gradient if  and ?"*