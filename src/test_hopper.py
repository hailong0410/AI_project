import gymnasium as gym

env = gym.make("Hopper-v5")

obs, info = env.reset()

print("Observation shape:", obs.shape)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}, Reward: {reward}")
    if terminated or truncated:
        print("Episode ended, resetting...")
        obs, info = env.reset()

env.close()
print("Done.")