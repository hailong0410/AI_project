import gymnasium as gym
import numpy as np
import torch

from buffer import RolloutBuffer
from ppo_agent import PPOAgent
from utils import save_rewards_to_json, save_training_summary, plot_rewards


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_ppo_on_hopper(
    env_name="Hopper-v5",
    seed=42,
    total_updates=20,
    rollout_steps=1024,
    gamma=0.99,
    lam=0.95,
    update_epochs=10,
    hidden_dim=64,
    device="cpu",
):
    set_seed(seed)

    env = gym.make(env_name)
    obs, info = env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"Environment: {env_name}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {act_dim}")
    print(f"Device: {device}")

    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        device=device,
    )

    episode_reward = 0.0
    episode_length = 0
    episode_count = 0

    reward_history = []
    update_history = []

    for update_idx in range(1, total_updates + 1):
        buffer = RolloutBuffer(device=device)

        for step in range(rollout_steps):
            action, log_prob, value = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(obs, action, reward, done, value, log_prob)

            episode_reward += reward
            episode_length += 1
            obs = next_obs

            if done:
                reward_history.append(float(episode_reward))
                episode_count += 1

                print(
                    f"[Update {update_idx:03d}] "
                    f"Episode {episode_count:03d} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Length: {episode_length}"
                )

                obs, info = env.reset()
                episode_reward = 0.0
                episode_length = 0

        last_value = agent.get_value(obs)
        buffer.compute_advantages_and_returns(
            last_value=last_value,
            gamma=gamma,
            lam=lam,
        )

        data = buffer.get_tensors()
        update_info = agent.update(data, update_epochs=update_epochs)

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
            f"Buffer size: {buffer.size()} | "
            f"Actor loss: {update_info['actor_loss']:.4f} | "
            f"Critic loss: {update_info['critic_loss']:.4f} | "
            f"Entropy: {update_info['entropy']:.4f} | "
            f"Avg reward (last 10 eps): {avg_recent_reward:.2f}\n"
        )

    rewards_path = f"../results/logs/hopper_rewards_seed_{seed}.json"
    summary_path = f"../results/logs/hopper_summary_seed_{seed}.json"
    plot_path = f"../results/plots/hopper_rewards_seed_{seed}.png"

    save_rewards_to_json(reward_history, rewards_path)

    summary = {
        "env_name": env_name,
        "seed": seed,
        "total_updates": total_updates,
        "rollout_steps": rollout_steps,
        "gamma": gamma,
        "lam": lam,
        "update_epochs": update_epochs,
        "hidden_dim": hidden_dim,
        "device": device,
        "final_avg_reward_last_10": float(np.mean(reward_history[-10:])) if len(reward_history) > 0 else 0.0,
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

    env.close()
    return reward_history


if __name__ == "__main__":
    rewards = train_ppo_on_hopper(
        env_name="Hopper-v5",
        seed=42,
        total_updates=20,
        rollout_steps=1024,
        gamma=0.99,
        lam=0.95,
        update_epochs=10,
        hidden_dim=64,
        device="cpu",
    )

    print("Training finished.")
    if len(rewards) > 0:
        print("Final average reward over last 10 episodes:", np.mean(rewards[-10:]))
        print("Saved reward logs and plot to results folder.")