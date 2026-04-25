import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, "results", "logs")
    plots_dir = os.path.join(project_root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    seeds = [42, 123, 999]

    baseline_scores = []
    adapted_scores = []

    print("\nSimple Spread: Baseline vs Adapted")
    print("-" * 80)
    print(f"{'Seed':<10}{'Baseline':<20}{'Adapted':<20}{'Difference':<20}")
    print("-" * 80)

    for seed in seeds:
        baseline_path = os.path.join(
            logs_dir,
            f"simple_spread_baseline_summary_seed_{seed}.json"
        )

        adapted_path = os.path.join(
            logs_dir,
            f"simple_spread_adapted_summary_seed_{seed}.json"
        )

        baseline_data = load_json(baseline_path)
        adapted_data = load_json(adapted_path)

        baseline_score = baseline_data.get(
            "final_eval_mean_team_reward",
            baseline_data["final_avg_team_reward_last_10"],
        )
        adapted_score = adapted_data.get(
            "final_eval_mean_team_reward",
            adapted_data["final_avg_team_reward_last_10"],
        )

        baseline_scores.append(baseline_score)
        adapted_scores.append(adapted_score)

        difference = adapted_score - baseline_score

        print(
            f"{seed:<10}"
            f"{baseline_score:<20.2f}"
            f"{adapted_score:<20.2f}"
            f"{difference:<20.2f}"
        )

    baseline_mean = np.mean(baseline_scores)
    baseline_std = np.std(baseline_scores)

    adapted_mean = np.mean(adapted_scores)
    adapted_std = np.std(adapted_scores)

    print("-" * 80)
    print(f"{'Mean':<10}{baseline_mean:<20.2f}{adapted_mean:<20.2f}{adapted_mean - baseline_mean:<20.2f}")
    print(f"{'Std':<10}{baseline_std:<20.2f}{adapted_std:<20.2f}")
    print("-" * 80)

    # Save summary table as JSON
    summary = {
        "seeds": seeds,
        "baseline_scores": baseline_scores,
        "adapted_scores": adapted_scores,
        "baseline_mean": float(baseline_mean),
        "baseline_std": float(baseline_std),
        "adapted_mean": float(adapted_mean),
        "adapted_std": float(adapted_std),
        "mean_difference_adapted_minus_baseline": float(adapted_mean - baseline_mean),
    }

    summary_path = os.path.join(
        logs_dir,
        "simple_spread_baseline_vs_adapted_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved comparison summary to:", summary_path)

    # Bar plot
    x = np.arange(len(seeds))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, baseline_scores, width, label="Baseline PPO")
    plt.bar(x + width / 2, adapted_scores, width, label="Adapted PPO")

    plt.xlabel("Seed")
    plt.ylabel("Final Deterministic Evaluation Mean Team Reward")
    plt.title("Simple Spread: Baseline PPO vs Adapted PPO")
    plt.xticks(x, [str(seed) for seed in seeds])
    plt.legend()
    plt.tight_layout()

    bar_plot_path = os.path.join(
        plots_dir,
        "simple_spread_baseline_vs_adapted_bar.png"
    )
    plt.savefig(bar_plot_path)
    plt.close()

    print("Saved bar plot to:", bar_plot_path)

    # Mean comparison plot
    methods = ["Baseline PPO", "Adapted PPO"]
    means = [baseline_mean, adapted_mean]
    stds = [baseline_std, adapted_std]

    plt.figure(figsize=(8, 6))
    plt.bar(methods, means, yerr=stds, capsize=8)
    plt.ylabel("Mean Final Deterministic Evaluation Team Reward")
    plt.title("Simple Spread: Mean Performance Across 3 Seeds")
    plt.tight_layout()

    mean_plot_path = os.path.join(
        plots_dir,
        "simple_spread_baseline_vs_adapted_mean.png"
    )
    plt.savefig(mean_plot_path)
    plt.close()

    print("Saved mean comparison plot to:", mean_plot_path)


if __name__ == "__main__":
    main()