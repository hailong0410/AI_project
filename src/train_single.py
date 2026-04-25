import gymnasium as gym
import math
import numpy as np
import torch

from ppo_agent import PPOAgent
from utils import (
    evaluate_deterministic_policy,
    save_rewards_to_json,
    save_training_summary,
    plot_rewards,
)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_vec_env(env_name, num_envs, seed):
    def make_single_env(rank):
        def thunk():
            env = gym.make(env_name)
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            env.observation_space.seed(seed + rank)
            return env

        return thunk

    return gym.vector.AsyncVectorEnv([make_single_env(i) for i in range(num_envs)])


def train_ppo_on_hopper(
    env_name="Hopper-v5",
    seed=42,
    total_timesteps=1_000_000,
    rollout_steps=2048,
    num_envs=1,
    minibatch_size=64,
    gamma=0.99,
    lam=0.95,
    update_epochs=10,
    hidden_dim=64,
    lr=3e-4,
    eval_episodes=20,
    device="cpu",
):
    set_seed(seed)
    torch_device = torch.device(device)

    envs = make_vec_env(env_name=env_name, num_envs=num_envs, seed=seed)
    obs_np, info = envs.reset(seed=seed)
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=torch_device)

    total_updates = max(int(total_timesteps // rollout_steps), 1)
    steps_per_env = math.ceil(rollout_steps / num_envs)
    collected_batch_size = steps_per_env * num_envs

    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape
    if len(obs_shape) != 1 or len(act_shape) != 1:
        raise ValueError("This trainer supports 1D Box observation/action spaces only.")

    obs_dim = obs_shape[0]
    act_dim = act_shape[0]

    print(f"Environment: {env_name}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {act_dim}")
    print(f"Num envs: {num_envs}")
    print(f"Rollout steps per update (effective): {rollout_steps}")
    print(f"Rollout steps per env: {steps_per_env}")
    print(f"Device: {device}")

    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        device=device,
    )

    current_ep_returns = np.zeros(num_envs, dtype=np.float64)
    current_ep_lengths = np.zeros(num_envs, dtype=np.int64)
    episode_count = 0

    reward_history = []
    update_history = []

    obs_buf = torch.zeros((steps_per_env, num_envs, obs_dim), dtype=torch.float32, device=torch_device)
    actions_buf = torch.zeros((steps_per_env, num_envs, act_dim), dtype=torch.float32, device=torch_device)
    logprobs_buf = torch.zeros((steps_per_env, num_envs), dtype=torch.float32, device=torch_device)
    rewards_buf = torch.zeros((steps_per_env, num_envs), dtype=torch.float32, device=torch_device)
    dones_buf = torch.zeros((steps_per_env, num_envs), dtype=torch.float32, device=torch_device)
    values_buf = torch.zeros((steps_per_env, num_envs), dtype=torch.float32, device=torch_device)

    act_low = envs.single_action_space.low
    act_high = envs.single_action_space.high

    for update_idx in range(1, total_updates + 1):
        for t in range(steps_per_env):
            obs_buf[t] = obs

            with torch.no_grad():
                actions, log_probs, values = agent.ac.step(obs)

            actions_buf[t] = actions
            logprobs_buf[t] = log_probs
            values_buf[t] = values

            action_np = np.clip(actions.cpu().numpy(), act_low, act_high)
            next_obs_np, reward_np, terminated_np, truncated_np, info = envs.step(action_np)
            done_np = np.logical_or(terminated_np, truncated_np)

            rewards_buf[t] = torch.as_tensor(reward_np, dtype=torch.float32, device=torch_device)
            dones_buf[t] = torch.as_tensor(done_np, dtype=torch.float32, device=torch_device)

            current_ep_returns += reward_np
            current_ep_lengths += 1

            for i in range(num_envs):
                if done_np[i]:
                    episode_count += 1
                    ep_reward = float(current_ep_returns[i])
                    ep_length = int(current_ep_lengths[i])
                    reward_history.append(ep_reward)

                    print(
                        f"[Update {update_idx:03d}] "
                        f"Episode {episode_count:03d} | "
                        f"Reward: {ep_reward:.2f} | "
                        f"Length: {ep_length}"
                    )

                    current_ep_returns[i] = 0.0
                    current_ep_lengths[i] = 0

            obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=torch_device)

        with torch.no_grad():
            next_value = agent.ac.critic(obs)

        advantages = torch.zeros_like(rewards_buf, device=torch_device)
        lastgaelam = torch.zeros(num_envs, dtype=torch.float32, device=torch_device)

        for t in reversed(range(steps_per_env)):
            if t == steps_per_env - 1:
                next_non_terminal = 1.0 - dones_buf[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones_buf[t]
                next_values = values_buf[t + 1]

            delta = rewards_buf[t] + gamma * next_values * next_non_terminal - values_buf[t]
            lastgaelam = delta + gamma * lam * next_non_terminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, obs_dim))
        b_actions = actions_buf.reshape((-1, act_dim))
        b_log_probs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        if collected_batch_size > rollout_steps:
            b_obs = b_obs[:rollout_steps]
            b_actions = b_actions[:rollout_steps]
            b_log_probs = b_log_probs[:rollout_steps]
            b_advantages = b_advantages[:rollout_steps]
            b_returns = b_returns[:rollout_steps]

        data = {
            "obs": b_obs,
            "actions": b_actions,
            "log_probs": b_log_probs,
            "advantages": b_advantages,
            "returns": b_returns,
        }

        update_info = agent.update(
            data,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
        )

        avg_recent_reward = np.mean(reward_history[-10:]) if len(reward_history) > 0 else 0.0

        update_history.append({
            "update": update_idx,
            "actor_loss": float(update_info["actor_loss"]),
            "critic_loss": float(update_info["critic_loss"]),
            "entropy": float(update_info["entropy"]),
            "avg_reward_last_10": float(avg_recent_reward),
        })

        print(
            f"=== PPO Update {update_idx:03d}/{total_updates} ===\n"
            f"Buffer size: {rollout_steps} | "
            f"Actor loss: {update_info['actor_loss']:.4f} | "
            f"Critic loss: {update_info['critic_loss']:.4f} | "
            f"Entropy: {update_info['entropy']:.4f} | "
            f"Avg reward (last 10 eps): {avg_recent_reward:.2f}\n"
        )

    rewards_path = f"../results/logs/hopper_rewards_seed_{seed}.json"
    summary_path = f"../results/logs/hopper_summary_seed_{seed}.json"
    plot_path = f"../results/plots/hopper_rewards_seed_{seed}.png"

    save_rewards_to_json(reward_history, rewards_path)

    eval_result = evaluate_deterministic_policy(
        actor_critic=agent.ac,
        env_name=env_name,
        seed=seed,
        episodes=eval_episodes,
        device=torch_device,
    )

    summary = {
        "env_name": env_name,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "total_updates": total_updates,
        "rollout_steps": rollout_steps,
        "num_envs": num_envs,
        "rollout_steps_per_env": steps_per_env,
        "minibatch_size": minibatch_size,
        "gamma": gamma,
        "lam": lam,
        "update_epochs": update_epochs,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "eval_episodes": eval_episodes,
        "device": device,
        "final_avg_reward_last_10": float(np.mean(reward_history[-10:])) if len(reward_history) > 0 else 0.0,
        "final_eval_mean_return": eval_result["mean_return"],
        "final_eval_std_return": eval_result["std_return"],
        "final_eval_returns": eval_result["returns"],
        "num_episodes": len(reward_history),
        "update_history": update_history,
    }

    save_training_summary(summary, summary_path)

    plot_rewards(
        reward_history,
        save_path=plot_path,
        window=10,
        title=f"PPO on {env_name} (seed={seed})"
    )

    envs.close()
    return summary


if __name__ == "__main__":
    summary = train_ppo_on_hopper(
        env_name="Hopper-v5",
        seed=42,
        total_timesteps=1_000_000,
        rollout_steps=2048,
        num_envs=1,
        minibatch_size=64,
        gamma=0.99,
        lam=0.95,
        update_epochs=10,
        hidden_dim=64,
        lr=3e-4,
        eval_episodes=20,
        device="cpu",
    )

    print("Training finished.")
    print(
        "Final deterministic evaluation mean return:",
        f"{summary['final_eval_mean_return']:.2f}",
    )
    print("Saved reward logs and plot to results folder.")