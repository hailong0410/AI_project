from mpe2 import simple_spread_v3


def main():
    env = simple_spread_v3.parallel_env(
        N=3,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=True,
    )

    observations, infos = env.reset(seed=42)

    print("Agents:", env.agents)
    print("Number of agents:", len(env.agents))

    for agent in env.agents:
        obs = observations[agent]
        print(f"\nAgent: {agent}")
        print("Observation shape:", obs.shape)
        print("Observation sample:", obs)

    print("\nAction spaces:")
    for agent in env.agents:
        print(agent, "->", env.action_space(agent))

    print("\nObservation spaces:")
    for agent in env.agents:
        print(agent, "->", env.observation_space(agent))

    for step in range(3):
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }

        observations, rewards, terminations, truncations, infos = env.step(actions)

        print(f"\nStep {step + 1}")
        print("Rewards:", rewards)
        print("Terminations:", terminations)
        print("Truncations:", truncations)

    env.close()
    print("\nSimple Spread test finished.")


if __name__ == "__main__":
    main()