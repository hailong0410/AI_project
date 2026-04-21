import os
import json
import numpy as np


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, "results", "logs")

    seeds = [42, 123, 999]
    rows = []

    for seed in seeds:
        summary_path = os.path.join(logs_dir, f"hopper_summary_seed_{seed}.json")
        data = load_json(summary_path)

        rows.append({
            "seed": seed,
            "num_episodes": data["num_episodes"],
            "final_avg_reward_last_10": data["final_avg_reward_last_10"],
        })

    print("\nReproduction Summary (Hopper-v5 + PPO)")
    print("-" * 60)
    print(f"{'Seed':<10}{'Episodes':<15}{'Final Avg Reward (Last 10)':<25}")
    print("-" * 60)

    final_rewards = []
    for row in rows:
        print(f"{row['seed']:<10}{row['num_episodes']:<15}{row['final_avg_reward_last_10']:<25.2f}")
        final_rewards.append(row["final_avg_reward_last_10"])

    print("-" * 60)
    print(f"Mean final avg reward: {np.mean(final_rewards):.2f}")
    print(f"Std final avg reward : {np.std(final_rewards):.2f}")
    print(f"Best seed            : {rows[np.argmax(final_rewards)]['seed']}")
    print(f"Worst seed           : {rows[np.argmin(final_rewards)]['seed']}")


if __name__ == "__main__":
    main()