import torch
import torch.nn as nn
import torch.optim as optim

from models import ActorCritic


class PPOAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=64,
        actor_lr=3e-4,
        critic_lr=1e-3,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        device="cpu",
    ):
        self.device = device
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.ac = ActorCritic(obs_dim, act_dim, hidden_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.ac.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.ac.critic.parameters(), lr=critic_lr)

    def select_action(self, obs):
        """
        Select action for environment interaction.
        Input:
            obs: numpy array or list, shape [obs_dim]
        Returns:
            action_np: numpy array, shape [act_dim]
            log_prob: float
            value: float
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value = self.ac.step(obs_tensor)

        action = action.squeeze(0)
        log_prob = log_prob.squeeze(0)
        value = value.squeeze(0)

        # Hopper action space is bounded to [-1, 1]
        action_clipped = torch.clamp(action, -1.0, 1.0)

        return (
            action_clipped.cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def get_value(self, obs):
        """
        Get value estimate for a single observation.
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.ac.critic(obs_tensor)
        return value.item()

    def update(self, data, update_epochs=10):
        """
        PPO update using collected rollout data.

        data keys:
            obs, actions, log_probs, advantages, returns
        """
        obs = data["obs"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]

        # Normalize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss_value = 0.0
        critic_loss_value = 0.0
        entropy_value = 0.0

        for _ in range(update_epochs):
            # New log probs under current policy
            new_log_probs = self.ac.actor.get_log_prob(obs, actions)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            values = self.ac.critic(obs)
            critic_loss = nn.MSELoss()(values, returns)

            entropy = self.ac.actor.get_entropy(obs).mean()

            total_actor_loss = actor_loss - self.entropy_coef * entropy
            total_critic_loss = self.value_coef * critic_loss

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            total_critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss_value += actor_loss.item()
            critic_loss_value += critic_loss.item()
            entropy_value += entropy.item()

        num_updates = update_epochs
        return {
            "actor_loss": actor_loss_value / num_updates,
            "critic_loss": critic_loss_value / num_updates,
            "entropy": entropy_value / num_updates,
        }