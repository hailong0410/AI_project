import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """
    Policy network for continuous action spaces.
    Input: state vector
    Output:
        - action mean
        - log standard deviation (learnable parameter)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim)
        )

        # One log_std value per action dimension
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        """
        Returns mean and std for the Gaussian policy.
        """
        mean = self.net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_dist(self, obs: torch.Tensor) -> Normal:
        """
        Build the action distribution.
        """
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def sample_action(self, obs: torch.Tensor):
        """
        Sample an action from the policy.
        Returns:
            action
            log_prob
        """
        dist = self.get_dist(obs)
        action = dist.rsample()  # rsample supports reparameterization
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def get_log_prob(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Compute log probability of a given action under current policy.
        """
        dist = self.get_dist(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return log_prob

    def get_entropy(self, obs: torch.Tensor):
        """
        Compute entropy of the action distribution.
        """
        dist = self.get_dist(obs)
        entropy = dist.entropy().sum(dim=-1)
        return entropy

    def act_deterministic(self, obs: torch.Tensor):
        """
        Return the mean action for evaluation.
        """
        mean, _ = self.forward(obs)
        return mean


class Critic(nn.Module):
    """
    Value network.
    Input: state vector
    Output: scalar state value
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor):
        """
        Returns state value.
        Shape:
            input:  [batch_size, obs_dim] or [obs_dim]
            output: [batch_size] or scalar-like tensor
        """
        value = self.net(obs)
        return value.squeeze(-1)


class ActorCritic(nn.Module):
    """
    Convenience wrapper containing both actor and critic.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, hidden_dim)
        self.critic = Critic(obs_dim, hidden_dim)

    def step(self, obs: torch.Tensor):
        """
        Used during training rollout.
        Returns:
            action, log_prob, value
        """
        with torch.no_grad():
            action, log_prob = self.actor.sample_action(obs)
            value = self.critic(obs)
        return action, log_prob, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Used for acting in the environment.
        """
        with torch.no_grad():
            if deterministic:
                action = self.actor.act_deterministic(obs)
            else:
                action, _ = self.actor.sample_action(obs)
        return action