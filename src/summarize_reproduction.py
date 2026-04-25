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

        final_metric = data.get("final_eval_mean_return", data["final_avg_reward_last_10"])

        rows.append({
            "seed": seed,
            "num_episodes": data["num_episodes"],
            "final_eval_mean_return": final_metric,
        })

    print("\nReproduction Summary (Hopper-v5 + PPO)")
    print("-" * 60)
    print(f"{'Seed':<10}{'Episodes':<15}{'Final Deterministic Eval Mean':<30}")
    print("-" * 60)

    final_scores = []
    for row in rows:
        print(f"{row['seed']:<10}{row['num_episodes']:<15}{row['final_eval_mean_return']:<30.2f}")
        final_scores.append(row["final_eval_mean_return"])

    print("-" * 60)
    print(f"Mean final eval return: {np.mean(final_scores):.2f}")
    print(f"Std final eval return : {np.std(final_scores):.2f}")
    print(f"Best seed             : {rows[np.argmax(final_scores)]['seed']}")
    print(f"Worst seed            : {rows[np.argmin(final_scores)]['seed']}")


if __name__ == "__main__":
    main()