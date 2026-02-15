## Discussion Question Answers

**1. Why do we need two networks (Policy and Value) for PPO? What is the specific job of the Value network during the "Update" phase?**
- The role of the Policy network is to take actions given state where as the role of the Value network is to predict future reward given the state and action.
- During the "update" phase, the Value network outputs are used to calcuate "advantage" for the actions.
- Not sure, how it is trained (what's the target value) and what GAE represents. 

**2. In the PPO clipping equation, what happens to the gradient if the Advantage is positive (A > 0) but the ratio is greater than 1 + ep? Why is this behavior desirable?**
- The clipping of the probability ratio prevents significant policy updates, thereby preventing it from deviating too far from its original state.
- When A > 0, the clipping function establishes an upper bound on the ratio at `1 + ep`, preventing accidental deviations to a larger value beyond this threshold.
- Similarly, when A < 0, the ratio is clipped to `1 - ep`, which prevents excessive penalties for policy mistakes.
- This approach facilitates smoother policy learning and prevents erratic behavior due to large updates, leading to more stable policy convergence.

**3. How is the "Reward" in CartPole (usually +1 for staying alive) different from the "Reward" in an LLM (usually a preference score)? How does this change the difficulty of the problem?**
- TBD
