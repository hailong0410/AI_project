import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class SharedActor(nn.Module):
    """
    Shared actor for all agents.
    Input: local observation of one agent
    Output: Gaussian policy for continuous action
    """

    def __init__(self, local_obs_dim, act_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(local_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, local_obs):
        mean = self.net(local_obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_dist(self, local_obs):
        mean, std = self.forward(local_obs)
        return Normal(mean, std)

    def sample_action(self, local_obs):
        dist = self.get_dist(local_obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def get_log_prob(self, local_obs, action):
        dist = self.get_dist(local_obs)
        return dist.log_prob(action).sum(dim=-1)

    def get_entropy(self, local_obs):
        dist = self.get_dist(local_obs)
        return dist.entropy().sum(dim=-1)

    def act_deterministic(self, local_obs):
        mean, _ = self.forward(local_obs)
        return mean


class CentralizedCritic(nn.Module):
    """
    Centralized critic.
    Input: joint observation of all agents
    Output: scalar value
    """

    def __init__(self, joint_obs_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs):
        value = self.net(joint_obs)
        return value.squeeze(-1)


class MAPPOAgent:
    """
    Multi-agent PPO style agent with:
    - shared actor
    - centralized critic
    """

    def __init__(
        self,
        local_obs_dim,
        joint_obs_dim,
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

        self.actor = SharedActor(local_obs_dim, act_dim, hidden_dim).to(device)
        self.critic = CentralizedCritic(joint_obs_dim, hidden_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, local_obs, joint_obs):
        local_obs_tensor = torch.tensor(
            local_obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        joint_obs_tensor = torch.tensor(
            joint_obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.actor.sample_action(local_obs_tensor)
            value = self.critic(joint_obs_tensor)

        action = action.squeeze(0)
        log_prob = log_prob.squeeze(0)
        value = value.squeeze(0)

        return action.cpu().numpy(), log_prob.item(), value.item()

    def select_action_deterministic(self, local_obs):
        local_obs_tensor = torch.tensor(
            local_obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action = self.actor.act_deterministic(local_obs_tensor)

        return action.squeeze(0).cpu().numpy()

    def get_value(self, joint_obs):
        joint_obs_tensor = torch.tensor(
            joint_obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            value = self.critic(joint_obs_tensor)

        return value.item()

    def update(self, data, update_epochs=10, minibatch_size=64):
        local_obs = data["local_obs"]
        joint_obs = data["joint_obs"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss_value = 0.0
        critic_loss_value = 0.0
        entropy_value = 0.0
        minibatch_count = 0

        batch_size = local_obs.shape[0]
        indices = torch.arange(batch_size, device=self.device)

        for _ in range(update_epochs):
            perm = indices[torch.randperm(batch_size, device=self.device)]

            for start in range(0, batch_size, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]

                mb_local_obs = local_obs[mb_idx]
                mb_joint_obs = joint_obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                new_log_probs = self.actor.get_log_prob(mb_local_obs, mb_actions)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps
                ) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(mb_joint_obs)
                critic_loss = nn.MSELoss()(values, mb_returns)

                entropy = self.actor.get_entropy(mb_local_obs).mean()

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
                minibatch_count += 1

        denom = max(minibatch_count, 1)

        return {
            "actor_loss": actor_loss_value / denom,
            "critic_loss": critic_loss_value / denom,
            "entropy": entropy_value / denom,
        }