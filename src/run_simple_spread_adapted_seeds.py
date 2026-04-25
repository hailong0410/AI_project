from train_simple_spread_adapted import train_simple_spread_adapted


def main():
    seeds = [42, 123, 999]
    final_results = []

    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"Starting Simple Spread adapted run for seed = {seed}")
        print("=" * 70)

        summary = train_simple_spread_adapted(
            seed=seed,
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

        final_eval_mean = float(summary["final_eval_mean_team_reward"])
        final_eval_std = float(summary["final_eval_std_team_reward"])

        final_results.append({
            "seed": seed,
            "num_episodes": int(summary["num_episodes"]),
            "final_eval_mean_team_reward": final_eval_mean,
            "final_eval_std_team_reward": final_eval_std,
        })

        print(f"Finished seed {seed}")
        print(f"Final deterministic eval mean team reward (20 eps): {final_eval_mean:.2f}")

    print("\n" + "=" * 70)
    print("All Simple Spread adapted runs finished.")
    print("=" * 70)

    for item in final_results:
        print(
            f"Seed {item['seed']} | "
            f"Episodes: {item['num_episodes']} | "
            f"Final deterministic eval mean: {item['final_eval_mean_team_reward']:.2f} | "
            f"Eval std: {item['final_eval_std_team_reward']:.2f}"
        )


if __name__ == "__main__":
    main()