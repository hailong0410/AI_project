import numpy as np
from train_simple_spread_adapted import train_simple_spread_adapted


def main():
    seeds = [42, 123, 999]
    final_results = []

    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"Starting Simple Spread adapted run for seed = {seed}")
        print("=" * 70)

        rewards = train_simple_spread_adapted(
            seed=seed,
            total_updates=20,
            rollout_steps=1024,
            max_cycles=25,
            gamma=0.99,
            lam=0.95,
            update_epochs=10,
            hidden_dim=64,
            device="cpu",
        )

        final_avg = float(np.mean(rewards[-10:])) if len(rewards) > 0 else 0.0

        final_results.append({
            "seed": seed,
            "num_episodes": len(rewards),
            "final_avg_team_reward_last_10": final_avg,
        })

        print(f"Finished seed {seed}")
        print(f"Final average team reward over last 10 episodes: {final_avg:.2f}")

    print("\n" + "=" * 70)
    print("All Simple Spread adapted runs finished.")
    print("=" * 70)

    for item in final_results:
        print(
            f"Seed {item['seed']} | "
            f"Episodes: {item['num_episodes']} | "
            f"Final avg team reward last 10: {item['final_avg_team_reward_last_10']:.2f}"
        )


if __name__ == "__main__":
    main()