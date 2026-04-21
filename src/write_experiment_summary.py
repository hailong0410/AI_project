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

    # -------------------------
    # Reproduction: Hopper-v5
    # -------------------------
    hopper_scores = []

    for seed in seeds:
        path = os.path.join(logs_dir, f"hopper_summary_seed_{seed}.json")
        data = load_json(path)
        hopper_scores.append(data["final_avg_reward_last_10"])

    hopper_mean = np.mean(hopper_scores)
    hopper_std = np.std(hopper_scores)

    # -------------------------
    # Adaptation baseline
    # -------------------------
    baseline_scores = []

    for seed in seeds:
        path = os.path.join(
            logs_dir,
            f"simple_spread_baseline_summary_seed_{seed}.json"
        )
        data = load_json(path)
        baseline_scores.append(data["final_avg_team_reward_last_10"])

    baseline_mean = np.mean(baseline_scores)
    baseline_std = np.std(baseline_scores)

    # -------------------------
    # Adapted method
    # -------------------------
    adapted_scores = []

    for seed in seeds:
        path = os.path.join(
            logs_dir,
            f"simple_spread_adapted_summary_seed_{seed}.json"
        )
        data = load_json(path)
        adapted_scores.append(data["final_avg_team_reward_last_10"])

    adapted_mean = np.mean(adapted_scores)
    adapted_std = np.std(adapted_scores)

    difference = adapted_mean - baseline_mean

    # -------------------------
    # Write text summary
    # -------------------------
    output_path = os.path.join(logs_dir, "experiment_summary.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("AI Final Project Experiment Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write("Track Chosen:\n")
        f.write("Track 2 - PPO\n\n")

        f.write("Project Goal:\n")
        f.write(
            "The project reproduces PPO on a standard MuJoCo benchmark "
            "and adapts PPO to a multi-agent coordination environment.\n\n"
        )

        f.write("Phase 1: Reproduction\n")
        f.write("-" * 60 + "\n")
        f.write("Environment: Hopper-v5\n")
        f.write("Algorithm: PPO\n")
        f.write("Seeds: 42, 123, 999\n\n")

        f.write("Hopper-v5 Results:\n")
        for seed, score in zip(seeds, hopper_scores):
            f.write(f"Seed {seed}: Final average reward last 10 episodes = {score:.2f}\n")

        f.write(f"\nMean: {hopper_mean:.2f}\n")
        f.write(f"Std: {hopper_std:.2f}\n\n")

        f.write("Phase 2: Adaptation Environment\n")
        f.write("-" * 60 + "\n")
        f.write("Environment: MPE2 Simple Spread\n")
        f.write("Number of agents: 3\n")
        f.write("Local observation dimension: 18\n")
        f.write("Action dimension: 5\n")
        f.write("Task: Multi-agent coordination to cover landmarks and avoid collisions.\n\n")

        f.write("Baseline Method:\n")
        f.write(
            "Naive shared-policy PPO. All agents share one policy network, "
            "and each agent uses only its own local observation. The critic is also local.\n\n"
        )

        f.write("Adapted Method:\n")
        f.write(
            "Shared actor with centralized critic. Each actor still receives local "
            "observations, but the critic receives the joint observation of all agents "
            "(18 x 3 = 54 dimensions).\n\n"
        )

        f.write("Simple Spread Results:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Seed':<10}{'Baseline':<15}{'Adapted':<15}{'Difference':<15}\n")

        for seed, b, a in zip(seeds, baseline_scores, adapted_scores):
            f.write(f"{seed:<10}{b:<15.2f}{a:<15.2f}{a - b:<15.2f}\n")

        f.write("\n")
        f.write(f"Baseline Mean: {baseline_mean:.2f}\n")
        f.write(f"Baseline Std: {baseline_std:.2f}\n")
        f.write(f"Adapted Mean: {adapted_mean:.2f}\n")
        f.write(f"Adapted Std: {adapted_std:.2f}\n")
        f.write(f"Mean Difference Adapted - Baseline: {difference:.2f}\n\n")

        f.write("Interpretation:\n")
        f.write(
            "Because Simple Spread rewards are negative, a value closer to zero is better. "
            "In these experiments, the adapted centralized-critic version did not outperform "
            "the naive baseline under the current limited training budget. This does not mean "
            "the adaptation is invalid. The centralized critic is still a meaningful algorithmic "
            "change because it provides the critic with joint multi-agent information. However, "
            "the current result suggests that the adapted method may require longer training, "
            "hyperparameter tuning, or improved action/reward handling to show stronger performance.\n\n"
        )

        f.write("Ablation Explanation:\n")
        f.write(
            "The comparison between the baseline and adapted method serves as an ablation. "
            "The main difference is the critic input. The baseline uses a local critic, while "
            "the adapted method uses a centralized critic with joint observations. Therefore, "
            "this comparison isolates the effect of adding centralized value estimation.\n"
        )

    print("Saved experiment summary to:", output_path)


if __name__ == "__main__":
    main()