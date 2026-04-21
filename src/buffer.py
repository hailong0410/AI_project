import torch


class RolloutBuffer:
    """
    Buffer for storing trajectories collected from interacting with the environment.
    This buffer is designed for PPO with Generalized Advantage Estimation (GAE).
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

        self.advantages = None
        self.returns = None

    def add(self, obs, action, reward, done, value, log_prob):
        """
        Store one transition step.
        """
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def compute_advantages_and_returns(self, last_value, gamma=0.99, lam=0.95):
        """
        Compute GAE advantages and discounted returns.

        Args:
            last_value: value estimate of the final state after rollout
            gamma: discount factor
            lam: GAE lambda
        """
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

        self.advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        self.returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

    def get_tensors(self):
        """
        Convert stored rollout data into tensors for PPO training.
        """
        obs = torch.tensor(self.obs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions, dtype=torch.float32, device=self.device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.values, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)

        return {
            "obs": obs,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "dones": dones,
            "rewards": rewards,
            "advantages": self.advantages,
            "returns": self.returns,
        }

    def size(self):
        return len(self.rewards)