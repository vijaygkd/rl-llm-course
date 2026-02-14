# Week 1: The Foundations of RLHF & PPO

**Goal:** Understand the "Classic" RLHF pipeline used by InstructGPT/ChatGPT and implement the Proximal Policy Optimization (PPO) algorithm from scratch on a toy environment.

## 1. The Big Picture: From SFT to RLHF

We start by understanding *why* we need Reinforcement Learning at all. Supervised Fine-Tuning (SFT) is excellent for teaching a model *how* to speak (syntax, format, style), but it struggles with *what* to say (alignment, helpfulness, safety).

In SFT, the loss function is simple: **Next Token Prediction (Cross-Entropy)**.



If the dataset has a noisy or sub-optimal answer, the model is forced to learn it. There is no signal for "this answer is okay, but this other one is better."

**Enter RLHF (Reinforcement Learning from Human Feedback):**
We treat the Language Model as an **Agent** in an environment.

* **State ():** The prompt + tokens generated so far.
* **Action ():** The next token selected from the vocabulary.
* **Reward ():** A scalar score (e.g., +1.5) given to the *entire completed sequence*.
* **Policy ():** The LLM itself (the neural network weights).

## 2. The Algorithm: Proximal Policy Optimization (PPO)

Standard Policy Gradient methods (like REINFORCE) are notoriously unstable. A single "lucky" run with a high reward can cause the model to update its weights too drastically, destroying the delicate linguistic representations learned during pre-training. This is called **Policy Collapse**.

PPO solves this by enforcing a **Trust Region**: "You can update the model to get more reward, but don't stray too far from the previous version of the model."

### A. The Objective Function

The core of PPO is the **Clipped Surrogate Objective**. You will implement this equation:

Let's break it down:

1. **Probability Ratio :**


* This measures how much the probability of an action has changed compared to the old policy.
* If , the action is more likely now.


2. **Advantage :**
* This tells us *how much better* this action was compared to the average action in that state.
* We estimate this using **GAE (Generalized Advantage Estimation)**.


3. **Clipping (The Safety Rail):**
* We clip the ratio  to be within  (usually ).
* This prevents the "massive update." If the model tries to increase an action's probability by 100x (), the gradient is cut off.



### B. The Value Function (Critic)

PPO is an **Actor-Critic** method.

* **The Actor:** The Policy (LLM) that generates text.
* **The Critic:** A separate Value Head (or separate model) that predicts the expected future reward from the current state.
* The Critic helps calculate the **Advantage** by providing a baseline.

## 3. Reading List

Read these papers in this order. Do not get stuck on the proofs; focus on the architecture and loss functions.

1. **[Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)**
* *Focus:* **Section 3** (The Clipped Surrogate Objective) and **Figure 1** (The visual explanation of clipping).
* *Goal:* Understand Equation 7.


2. **[Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)](https://arxiv.org/abs/1706.03741)**
* *Focus:* Abstract and **Section 2**.
* *Goal:* Understand how "human preferences" (Sequence A > Sequence B) are converted into a Reward Model.



## 4. Assignment: "Gridworld PPO"

Before applying PPO to an expensive 7B parameter LLM, you must prove you understand the math by implementing it on a toy problem.

**Task:** Implement PPO from scratch to solve the `CartPole-v1` environment in Gymnasium.

**Requirements:**

1. **No High-Level Libraries:** You cannot use `Stable-Baselines3`, `RLLib`, or `HuggingFace TRL`. You must write the training loop in PyTorch.
2. **Actor-Critic Architecture:** Build a small neural net with two heads (Actor: outputs action probabilities, Critic: outputs value scalar).
3. **PPO Loss:** Manually implement Equation 7.
4. **GAE:** Implement Generalized Advantage Estimation for the advantage calculation.

**Suggested Structure:**

* `agent.py`: The `PPOAgent` class (networks + `get_action` + `update`).
* `train.py`: The main loop (collect rollout -> calculate advantage -> update agent).
* `utils.py`: Helper for GAE calculation.

## 5. Discussion Questions (Answer these in `notes/week1_answers.md`)

1. Why do we need two networks (Policy and Value) for PPO? What is the specific job of the Value network during the "Update" phase?
2. In the PPO clipping equation, what happens to the gradient if the Advantage is positive () but the ratio  is greater than ? Why is this behavior desirable?
3. How is the "Reward" in `CartPole` (usually +1 for staying alive) different from the "Reward" in an LLM (usually a preference score)? How does this change the difficulty of the problem?

---

### **Next Step**

Once you have created this file in your repo and read the Schulman paper, reply with your answer to **Discussion Question #2**. That will be our "Gatekeeper" to ensure you understand the math before you start coding the loss function.