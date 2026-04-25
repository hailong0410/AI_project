import os
import json
import numpy as np
import torch

from mpe2 import simple_spread_v3

from ma_ppo_agent import MAPPOAgent
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


def evaluate_simple_spread_adapted_deterministic(
    agent,
    seed=42,
    episodes=20,
    max_cycles=25,
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
        agent_order = list(env.agents)
        ep_team_reward = 0.0

        while len(env.agents) > 0:
            actions = {}
            for agent_name in list(env.agents):
                local_obs = observations[agent_name]
                action = agent.select_action_deterministic(local_obs)
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


class MARolloutBuffer:
    """
    Rollout buffer for multi-agent PPO with centralized critic.

    Stores:
    - local observations for actor
    - joint observations for centralized critic
    - actions
    - rewards
    - dones
    - values
    - log probabilities
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.clear()

    def clear(self):
        self.local_obs = []
        self.joint_obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

        self.advantages = None
        self.returns = None

    def add(self, local_obs, joint_obs, action, reward, done, value, log_prob):
        self.local_obs.append(local_obs)
        self.joint_obs.append(joint_obs)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def compute_advantages_and_returns(self, last_value, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0.0

        values = self.values + [float(last_value)]

        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t]
                + gamma * values[t + 1] * (1.0 - self.dones[t])
                - values[t]
            )
            gae = delta + gamma * lam * (1.0 - self.dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]

        self.advantages = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        self.returns = torch.tensor(
            returns, dtype=torch.float32, device=self.device
        )

    def get_tensors(self):
        return {
            "local_obs": torch.tensor(
                np.array(self.local_obs), dtype=torch.float32, device=self.device
            ),
            "joint_obs": torch.tensor(
                np.array(self.joint_obs), dtype=torch.float32, device=self.device
            ),
            "actions": torch.tensor(
                np.array(self.actions), dtype=torch.float32, device=self.device
            ),
            "log_probs": torch.tensor(
                self.log_probs, dtype=torch.float32, device=self.device
            ),
            "values": torch.tensor(
                self.values, dtype=torch.float32, device=self.device
            ),
            "dones": torch.tensor(
                self.dones, dtype=torch.float32, device=self.device
            ),
            "rewards": torch.tensor(
                self.rewards, dtype=torch.float32, device=self.device
            ),
            "advantages": self.advantages,
            "returns": self.returns,
        }

    def size(self):
        return len(self.rewards)


def build_joint_obs(observations, agent_order):
    """
    Concatenate all agents' observations into one joint observation.
    For Simple Spread with 3 agents:
        local obs dim = 18
        joint obs dim = 18 * 3 = 54
    """
    return np.concatenate([observations[agent] for agent in agent_order], axis=0)


def train_simple_spread_adapted(
    seed=42,
    total_timesteps=1_000_000,
    rollout_steps=1024,
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
    Adapted multi-agent PPO for MPE2 Simple Spread.

    Adaptation:
    - Shared actor: each agent acts based on its own local observation.
    - Centralized critic: critic receives the joint observation of all agents.
    """

    set_seed(seed)

    env = simple_spread_v3.parallel_env(
        N=3,
        local_ratio=0.5,
        max_cycles=max_cycles,
        continuous_actions=True,
    )

    observations, infos = env.reset(seed=seed)

    total_updates = max(int(total_timesteps // rollout_steps), 1)

    agent_order = list(env.agents)
    first_agent = agent_order[0]

    local_obs_dim = env.observation_space(first_agent).shape[0]
    act_dim = env.action_space(first_agent).shape[0]
    joint_obs_dim = local_obs_dim * len(agent_order)

    print("Environment: MPE2 Simple Spread")
    print("Method: Shared actor + centralized critic")
    print("Agents:", agent_order)
    print("Local observation dim:", local_obs_dim)
    print("Joint observation dim:", joint_obs_dim)
    print("Action dim:", act_dim)
    print("Device:", device)

    agent = MAPPOAgent(
        local_obs_dim=local_obs_dim,
        joint_obs_dim=joint_obs_dim,
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
        buffer = MARolloutBuffer(device=device)

        steps_collected = 0

        while steps_collected < rollout_steps:
            actions = {}
            step_data = {}

            active_agents = list(env.agents)
            joint_obs = build_joint_obs(observations, agent_order)

            for agent_name in active_agents:
                local_obs = observations[agent_name]

                action_raw, log_prob, value = agent.select_action(
                    local_obs=local_obs,
                    joint_obs=joint_obs,
                )

                action = np.clip(action_raw, 0.0, 1.0).astype(np.float32)

                actions[agent_name] = action

                step_data[agent_name] = {
                    "local_obs": local_obs,
                    "joint_obs": joint_obs,
                    "action": action_raw,
                    "log_prob": log_prob,
                    "value": value,
                }

            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            team_step_reward = 0.0

            for agent_name in active_agents:
                reward = rewards[agent_name]
                done = terminations[agent_name] or truncations[agent_name]

                buffer.add(
                    local_obs=step_data[agent_name]["local_obs"],
                    joint_obs=step_data[agent_name]["joint_obs"],
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

        if len(env.agents) > 0:
            last_joint_obs = build_joint_obs(observations, agent_order)
            last_value = agent.get_value(last_joint_obs)
        else:
            last_value = 0.0

        buffer.compute_advantages_and_returns(
            last_value=last_value,
            gamma=gamma,
            lam=lam,
        )

        data = buffer.get_tensors()
        update_info = agent.update(
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
            "buffer_size": buffer.size(),
            "actor_loss": float(update_info["actor_loss"]),
            "critic_loss": float(update_info["critic_loss"]),
            "entropy": float(update_info["entropy"]),
            "avg_team_reward_last_10": float(avg_recent_team_reward),
        })

        print(
            f"=== Adapted PPO Update {update_idx:03d}/{total_updates} ===\n"
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
        f"simple_spread_adapted_team_rewards_seed_{seed}.json"
    )

    summary_path = os.path.join(
        logs_dir,
        f"simple_spread_adapted_summary_seed_{seed}.json"
    )

    plot_path = os.path.join(
        plots_dir,
        f"simple_spread_adapted_team_rewards_seed_{seed}.png"
    )

    save_json(
        {"team_rewards": team_reward_history},
        rewards_path,
    )

    eval_result = evaluate_simple_spread_adapted_deterministic(
        agent=agent,
        seed=seed,
        episodes=eval_episodes,
        max_cycles=max_cycles,
    )

    summary = {
        "method": "shared_actor_centralized_critic",
        "environment": "MPE2 Simple Spread",
        "seed": seed,
        "num_agents": len(agent_order),
        "local_obs_dim": local_obs_dim,
        "joint_obs_dim": joint_obs_dim,
        "act_dim": act_dim,
        "total_timesteps": total_timesteps,
        "total_updates": total_updates,
        "rollout_steps": rollout_steps,
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
        title=f"Adapted PPO on Simple Spread (seed={seed})"
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
    train_simple_spread_adapted(
        seed=42,
        total_timesteps=1_000_000,
        rollout_steps=1024,
        max_cycles=25,
        gamma=0.99,
        lam=0.95,
        update_epochs=10,
        minibatch_size=64,
        hidden_dim=64,
        eval_episodes=20,
        device="cpu",
    )