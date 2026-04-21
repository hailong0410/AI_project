import os
import json
import numpy as np
import torch

from mpe2 import simple_spread_v3

from buffer import RolloutBuffer
from ppo_agent import PPOAgent
from utils import plot_rewards


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(data, filepath):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def train_simple_spread_baseline(
    seed=42,
    total_updates=20,
    rollout_steps=1024,
    max_cycles=25,
    gamma=0.99,
    lam=0.95,
    update_epochs=10,
    hidden_dim=64,
    device="cpu",
):
    """
    Naive multi-agent PPO baseline for MPE2 Simple Spread.

    Design:
    - One shared PPO policy for all agents.
    - Each agent uses only its own local observation.
    - No centralized critic.
    - This is the baseline that we will later compare against an adapted method.
    """

    set_seed(seed)

    env = simple_spread_v3.parallel_env(
        N=3,
        local_ratio=0.5,
        max_cycles=max_cycles,
        continuous_actions=True,
    )

    observations, infos = env.reset(seed=seed)

    agents = env.agents
    first_agent = agents[0]

    obs_dim = env.observation_space(first_agent).shape[0]
    act_dim = env.action_space(first_agent).shape[0]

    print("Environment: MPE2 Simple Spread")
    print("Agents:", agents)
    print("Observation dim:", obs_dim)
    print("Action dim:", act_dim)
    print("Device:", device)

    # Shared PPO agent used by all environment agents
    ppo_agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        device=device,
    )

    team_reward_history = []
    update_history = []

    episode_team_reward = 0.0
    episode_count = 0
    episode_step = 0

    for update_idx in range(1, total_updates + 1):
        buffer = RolloutBuffer(device=device)

        steps_collected = 0

        while steps_collected < rollout_steps:
            actions = {}
            step_data = {}

            active_agents = list(env.agents)

            for agent_name in active_agents:
                obs = observations[agent_name]

                action, log_prob, value = ppo_agent.select_action(obs)

                # Simple Spread continuous action space is Box(0.0, 1.0, shape=(5,))
                action = np.clip(action, 0.0, 1.0).astype(np.float32)

                actions[agent_name] = action

                step_data[agent_name] = {
                    "obs": obs,
                    "action": action,
                    "log_prob": log_prob,
                    "value": value,
                }

            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            team_step_reward = 0.0

            for agent_name in active_agents:
                reward = rewards[agent_name]
                done = terminations[agent_name] or truncations[agent_name]

                buffer.add(
                    obs=step_data[agent_name]["obs"],
                    action=step_data[agent_name]["action"],
                    reward=reward,
                    done=done,
                    value=step_data[agent_name]["value"],
                    log_prob=step_data[agent_name]["log_prob"],
                )

                team_step_reward += reward
                steps_collected += 1

            episode_team_reward += team_step_reward
            episode_step += 1

            observations = next_observations

            all_done = all(terminations.values()) or all(truncations.values())

            if all_done:
                episode_count += 1
                team_reward_history.append(float(episode_team_reward))

                print(
                    f"[Update {update_idx:03d}] "
                    f"Episode {episode_count:03d} | "
                    f"Team reward: {episode_team_reward:.2f} | "
                    f"Steps: {episode_step}"
                )

                observations, infos = env.reset()
                episode_team_reward = 0.0
                episode_step = 0

        # Estimate last value using the first active agent's observation.
        # This is a simple baseline approximation.
        if len(env.agents) > 0:
            last_agent = env.agents[0]
            last_obs = observations[last_agent]
            last_value = ppo_agent.get_value(last_obs)
        else:
            last_value = 0.0

        buffer.compute_advantages_and_returns(
            last_value=last_value,
            gamma=gamma,
            lam=lam,
        )

        data = buffer.get_tensors()
        update_info = ppo_agent.update(data, update_epochs=update_epochs)

        avg_recent_team_reward = (
            np.mean(team_reward_history[-10:])
            if len(team_reward_history) > 0
            else 0.0
        )

        update_history.append({
            "update": update_idx,
            "buffer_size": buffer.size(),
            "actor_loss": float(update_info["actor_loss"]),
            "critic_loss": float(update_info["critic_loss"]),
            "entropy": float(update_info["entropy"]),
            "avg_team_reward_last_10": float(avg_recent_team_reward),
        })

        print(
            f"=== Baseline PPO Update {update_idx:03d}/{total_updates} ===\n"
            f"Buffer size: {buffer.size()} | "
            f"Actor loss: {update_info['actor_loss']:.4f} | "
            f"Critic loss: {update_info['critic_loss']:.4f} | "
            f"Entropy: {update_info['entropy']:.4f} | "
            f"Avg team reward (last 10 eps): {avg_recent_team_reward:.2f}\n"
        )

    env.close()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, "results", "logs")
    plots_dir = os.path.join(project_root, "results", "plots")

    rewards_path = os.path.join(
        logs_dir,
        f"simple_spread_baseline_team_rewards_seed_{seed}.json"
    )

    summary_path = os.path.join(
        logs_dir,
        f"simple_spread_baseline_summary_seed_{seed}.json"
    )

    plot_path = os.path.join(
        plots_dir,
        f"simple_spread_baseline_team_rewards_seed_{seed}.png"
    )

    save_json(
        {"team_rewards": team_reward_history},
        rewards_path,
    )

    summary = {
        "method": "naive_shared_policy_ppo_baseline",
        "environment": "MPE2 Simple Spread",
        "seed": seed,
        "num_agents": 3,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "total_updates": total_updates,
        "rollout_steps": rollout_steps,
        "max_cycles": max_cycles,
        "gamma": gamma,
        "lam": lam,
        "update_epochs": update_epochs,
        "hidden_dim": hidden_dim,
        "final_avg_team_reward_last_10": (
            float(np.mean(team_reward_history[-10:]))
            if len(team_reward_history) > 0
            else 0.0
        ),
        "num_episodes": len(team_reward_history),
        "update_history": update_history,
    }

    save_json(summary, summary_path)

    plot_rewards(
        team_reward_history,
        save_path=plot_path,
        window=10,
        title=f"Naive PPO Baseline on Simple Spread (seed={seed})"
    )

    print("Training finished.")
    print("Saved rewards to:", rewards_path)
    print("Saved summary to:", summary_path)
    print("Saved plot to:", plot_path)

    if len(team_reward_history) > 0:
        print(
            "Final average team reward over last 10 episodes:",
            np.mean(team_reward_history[-10:])
        )

    return team_reward_history


if __name__ == "__main__":
    train_simple_spread_baseline(
        seed=42,
        total_updates=20,
        rollout_steps=1024,
        max_cycles=25,
        gamma=0.99,
        lam=0.95,
        update_epochs=10,
        hidden_dim=64,
        device="cpu",
    )