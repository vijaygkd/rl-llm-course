# PPO Evaluation Results

This file tracks evaluation results over 100 consecutive episodes for different agents.

## Results

| Agent | Training Status | Eval Episodes | Average Reward |
| --- | --- | --- | --- |
| Random Agent | Untrained (init model, no training) | 100 | 20.5 |
| PPO Agent | Critic trained with Monte Carlo return (sum of future rewards) | 100 | 168.05 |
