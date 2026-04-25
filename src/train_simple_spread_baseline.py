import os
import json
import math
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


def evaluate_simple_spread_baseline_deterministic(
    ppo_agent,
    seed=42,
    episodes=20,
    max_cycles=25,
    device="cpu",
):
    env = simple_spread_v3.parallel_env(
        N=3,
        local_ratio=0.5,
        max_cycles=max_cycles,
        continuous_actions=True,
    )

    episode_returns = []

    for ep in range(episodes):
        observations, infos = env.reset(seed=seed + 10_000 + ep)
        ep_team_reward = 0.0

        while len(env.agents) > 0:
            actions = {}
            for agent_name in list(env.agents):
                obs = observations[agent_name]
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_t = ppo_agent.ac.act(obs_t, deterministic=True)
                action = action_t.squeeze(0).cpu().numpy()
                action = np.clip(action, 0.0, 1.0).astype(np.float32)
                actions[agent_name] = action

            observations, rewards, terminations, truncations, infos = env.step(actions)
            ep_team_reward += float(sum(rewards.values()))

            if all(terminations.values()) or all(truncations.values()):
                break

        episode_returns.append(ep_team_reward)

    env.close()

    return {
        "mean_team_reward": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "std_team_reward": float(np.std(episode_returns)) if episode_returns else 0.0,
        "episode_team_rewards": episode_returns,
    }


def train_simple_spread_baseline(
    seed=42,
    total_timesteps=1_000_000,
    rollout_steps=2048,
    num_envs=1,
    max_cycles=25,
    gamma=0.99,
    lam=0.95,
    update_epochs=10,
    minibatch_size=64,
    hidden_dim=64,
    eval_episodes=20,
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

    envs = []
    observations_list = []
    infos_list = []

    for env_idx in range(num_envs):
        env = simple_spread_v3.parallel_env(
            N=3,
            local_ratio=0.5,
            max_cycles=max_cycles,
            continuous_actions=True,
        )
        observations, infos = env.reset(seed=seed + env_idx)
        envs.append(env)
        observations_list.append(observations)
        infos_list.append(infos)

    ref_env = envs[0]
    agents = ref_env.agents
    num_agents = len(agents)
    first_agent = agents[0]

    total_updates = max(int(total_timesteps // rollout_steps), 1)
    # In this multi-agent setup each env step contributes one sample per agent.
    # Scale per-env rollout length so each update targets rollout_steps transitions.
    steps_per_env = math.ceil(rollout_steps / (num_envs * num_agents))
    collected_batch_size = steps_per_env * num_envs * num_agents

    obs_dim = ref_env.observation_space(first_agent).shape[0]
    act_dim = ref_env.action_space(first_agent).shape[0]

    print("Environment: MPE2 Simple Spread")
    print("Agents:", agents)
    print("Observation dim:", obs_dim)
    print("Action dim:", act_dim)
    print("Num agents:", num_agents)
    print("Num envs:", num_envs)
    print("Rollout steps per update (effective):", rollout_steps)
    print("Rollout steps per env:", steps_per_env)
    print("Agent-level samples collected per update:", collected_batch_size)
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

    episode_team_rewards = np.zeros(num_envs, dtype=np.float64)
    episode_steps = np.zeros(num_envs, dtype=np.int64)
    episode_count = 0

    for update_idx in range(1, total_updates + 1):
        buffers = [RolloutBuffer(device=device) for _ in range(num_envs)]

        for _ in range(steps_per_env):
            for env_idx, env in enumerate(envs):
                observations = observations_list[env_idx]
                active_agents = list(env.agents)

                if len(active_agents) == 0:
                    observations, infos = env.reset()
                    observations_list[env_idx] = observations
                    infos_list[env_idx] = infos
                    active_agents = list(env.agents)

                actions = {}
                step_data = {}

                for agent_name in active_agents:
                    obs = observations[agent_name]

                    action_raw, log_prob, value = ppo_agent.select_action(obs)

                    # Simple Spread continuous action space is Box(0.0, 1.0, shape=(5,))
                    action = np.clip(action_raw, 0.0, 1.0).astype(np.float32)

                    actions[agent_name] = action

                    step_data[agent_name] = {
                        "obs": obs,
                        "action": action_raw,
                        "log_prob": log_prob,
                        "value": value,
                    }

                next_observations, rewards, terminations, truncations, infos = env.step(actions)

                team_step_reward = 0.0

                for agent_name in active_agents:
                    reward = rewards[agent_name]
                    done = terminations[agent_name] or truncations[agent_name]

                    buffers[env_idx].add(
                        obs=step_data[agent_name]["obs"],
                        action=step_data[agent_name]["action"],
                        reward=reward,
                        done=done,
                        value=step_data[agent_name]["value"],
                        log_prob=step_data[agent_name]["log_prob"],
                    )

                    team_step_reward += reward

                episode_team_rewards[env_idx] += team_step_reward
                episode_steps[env_idx] += 1

                observations_list[env_idx] = next_observations
                infos_list[env_idx] = infos

                all_done = all(terminations.values()) or all(truncations.values())

                if all_done:
                    episode_count += 1
                    ep_reward = float(episode_team_rewards[env_idx])
                    ep_steps = int(episode_steps[env_idx])
                    team_reward_history.append(ep_reward)

                    print(
                        f"[Update {update_idx:03d}] "
                        f"Episode {episode_count:03d} | "
                        f"Team reward: {ep_reward:.2f} | "
                        f"Steps: {ep_steps}"
                    )

                    observations, infos = env.reset()
                    observations_list[env_idx] = observations
                    infos_list[env_idx] = infos
                    episode_team_rewards[env_idx] = 0.0
                    episode_steps[env_idx] = 0

        buffer_tensors = []
        for env_idx, env in enumerate(envs):
            observations = observations_list[env_idx]

            # Estimate last value using the first active agent's observation.
            # This is a simple baseline approximation.
            if len(env.agents) > 0:
                last_agent = env.agents[0]
                last_obs = observations[last_agent]
                last_value = ppo_agent.get_value(last_obs)
            else:
                last_value = 0.0

            buffers[env_idx].compute_advantages_and_returns(
                last_value=last_value,
                gamma=gamma,
                lam=lam,
            )
            buffer_tensors.append(buffers[env_idx].get_tensors())

        data = {
            "obs": torch.cat([x["obs"] for x in buffer_tensors], dim=0),
            "actions": torch.cat([x["actions"] for x in buffer_tensors], dim=0),
            "log_probs": torch.cat([x["log_probs"] for x in buffer_tensors], dim=0),
            "advantages": torch.cat([x["advantages"] for x in buffer_tensors], dim=0),
            "returns": torch.cat([x["returns"] for x in buffer_tensors], dim=0),
        }

        if collected_batch_size > rollout_steps:
            data = {
                "obs": data["obs"][:rollout_steps],
                "actions": data["actions"][:rollout_steps],
                "log_probs": data["log_probs"][:rollout_steps],
                "advantages": data["advantages"][:rollout_steps],
                "returns": data["returns"][:rollout_steps],
            }

        effective_buffer_size = int(data["obs"].shape[0])
        update_info = ppo_agent.update(
            data,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
        )

        avg_recent_team_reward = (
            np.mean(team_reward_history[-10:])
            if len(team_reward_history) > 0
            else 0.0
        )

        update_history.append({
            "update": update_idx,
            "buffer_size": effective_buffer_size,
            "actor_loss": float(update_info["actor_loss"]),
            "critic_loss": float(update_info["critic_loss"]),
            "entropy": float(update_info["entropy"]),
            "avg_team_reward_last_10": float(avg_recent_team_reward),
        })

        print(
            f"=== Baseline PPO Update {update_idx:03d}/{total_updates} ===\n"
            f"Buffer size: {effective_buffer_size} | "
            f"Actor loss: {update_info['actor_loss']:.4f} | "
            f"Critic loss: {update_info['critic_loss']:.4f} | "
            f"Entropy: {update_info['entropy']:.4f} | "
            f"Avg team reward (last 10 eps): {avg_recent_team_reward:.2f}\n"
        )

    for env in envs:
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

    eval_result = evaluate_simple_spread_baseline_deterministic(
        ppo_agent=ppo_agent,
        seed=seed,
        episodes=eval_episodes,
        max_cycles=max_cycles,
        device=device,
    )

    summary = {
        "method": "naive_shared_policy_ppo_baseline",
        "environment": "MPE2 Simple Spread",
        "seed": seed,
        "num_agents": num_agents,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "total_timesteps": total_timesteps,
        "total_updates": total_updates,
        "rollout_steps": rollout_steps,
        "num_envs": num_envs,
        "rollout_steps_per_env": steps_per_env,
        "collected_agent_samples_per_update": collected_batch_size,
        "max_cycles": max_cycles,
        "gamma": gamma,
        "lam": lam,
        "update_epochs": update_epochs,
        "minibatch_size": minibatch_size,
        "hidden_dim": hidden_dim,
        "eval_episodes": eval_episodes,
        "final_avg_team_reward_last_10": (
            float(np.mean(team_reward_history[-10:]))
            if len(team_reward_history) > 0
            else 0.0
        ),
        "final_eval_mean_team_reward": eval_result["mean_team_reward"],
        "final_eval_std_team_reward": eval_result["std_team_reward"],
        "final_eval_team_rewards": eval_result["episode_team_rewards"],
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

    print(
        "Final deterministic evaluation mean team reward:",
        f"{summary['final_eval_mean_team_reward']:.2f}",
    )

    return summary


if __name__ == "__main__":
    train_simple_spread_baseline(
        seed=42,
        total_timesteps=1_000_000,
        rollout_steps=2048,
        num_envs=8,
        max_cycles=25,
        gamma=0.99,
        lam=0.95,
        update_epochs=10,
        minibatch_size=64,
        hidden_dim=64,
        eval_episodes=20,
        device="cpu",
    )