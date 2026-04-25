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
        lr=3e-4,
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

        self.optimizer = optim.Adam(
            list(self.ac.actor.parameters()) + list(self.ac.critic.parameters()),
            lr=lr,
        )

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

        return (
            action.cpu().numpy(),
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

    def update(self, data, update_epochs=10, minibatch_size=64):
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
        minibatch_count = 0

        batch_size = obs.shape[0]
        indices = torch.arange(batch_size, device=self.device)

        for _ in range(update_epochs):
            perm = indices[torch.randperm(batch_size, device=self.device)]

            for start in range(0, batch_size, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # New log probs under current policy
                new_log_probs = self.ac.actor.get_log_prob(mb_obs, mb_actions)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * mb_advantages
                )

                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.ac.critic(mb_obs)
                critic_loss = nn.MSELoss()(values, mb_returns)

                entropy = self.ac.actor.get_entropy(mb_obs).mean()

                total_actor_loss = actor_loss - self.entropy_coef * entropy
                total_critic_loss = self.value_coef * critic_loss

                total_loss = total_actor_loss + total_critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                actor_loss_value += actor_loss.item()
                critic_loss_value += critic_loss.item()
                entropy_value += entropy.item()
                minibatch_count += 1

        num_updates = max(minibatch_count, 1)
        return {
            "actor_loss": actor_loss_value / num_updates,
            "critic_loss": critic_loss_value / num_updates,
            "entropy": entropy_value / num_updates,
        }