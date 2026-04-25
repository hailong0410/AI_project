import os
import json
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def moving_average(values, window=10):
    if len(values) == 0:
        return []
    values = np.array(values, dtype=np.float32)
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(values[start:i + 1].mean())
    return smoothed


def save_rewards_to_json(rewards, filepath):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"rewards": rewards}, f, indent=2)


def save_training_summary(summary, filepath):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def plot_rewards(rewards, save_path, window=10, title="Training Reward Curve"):
    ensure_dir(os.path.dirname(save_path))

    episodes = np.arange(1, len(rewards) + 1)
    smoothed = moving_average(rewards, window=window)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label="Episode Reward")
    plt.plot(episodes, smoothed, label=f"Moving Average ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_deterministic_policy(actor_critic, env_name, seed=42, episodes=20, device="cpu"):
    env = gym.make(env_name)
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + ep)
        done = False
        ep_return = 0.0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_t = actor_critic.act(obs_t, deterministic=True)

            action = action_t.squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)

            done = bool(terminated or truncated)
            ep_return += float(reward)

        returns.append(ep_return)

    env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "returns": returns,
    }