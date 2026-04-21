import os
import json
import numpy as np
import matplotlib.pyplot as plt


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