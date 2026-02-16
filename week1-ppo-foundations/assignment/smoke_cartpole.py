import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset()

    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(env.step(action))
        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
