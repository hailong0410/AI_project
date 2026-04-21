import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_rewards(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["rewards"]


def moving_average(values, window=10):
    values = np.array(values, dtype=np.float32)
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(values[start:i + 1].mean())
    return smoothed


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, "results", "logs")
    plots_dir = os.path.join(project_root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    seeds = [42, 123, 999]

    plt.figure(figsize=(10, 6))

    for seed in seeds:
        rewards_path = os.path.join(logs_dir, f"hopper_rewards_seed_{seed}.json")
        rewards = load_rewards(rewards_path)
        smoothed = moving_average(rewards, window=10)
        episodes = np.arange(1, len(rewards) + 1)

        plt.plot(episodes, smoothed, label=f"Seed {seed}")

    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.title("PPO Reproduction on Hopper-v5 (3 Seeds)")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(plots_dir, "hopper_reproduction_3seed_comparison.png")
    plt.savefig(save_path)
    plt.close()

    print("Saved comparison plot to:", save_path)


if __name__ == "__main__":
    main()