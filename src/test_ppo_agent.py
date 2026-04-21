import torch
from ppo_agent import PPOAgent

obs_dim = 11
act_dim = 3

agent = PPOAgent(obs_dim, act_dim)

# Fake training batch
batch_size = 8
data = {
    "obs": torch.randn(batch_size, obs_dim),
    "actions": torch.randn(batch_size, act_dim).clamp(-1.0, 1.0),
    "log_probs": torch.randn(batch_size),
    "advantages": torch.randn(batch_size),
    "returns": torch.randn(batch_size),
}

result = agent.update(data, update_epochs=3)

print("Update result:", result)

# Test action selection
obs = [0.1] * obs_dim
action, log_prob, value = agent.select_action(obs)

print("Action:", action)
print("Action shape:", action.shape)
print("Log prob:", log_prob)
print("Value:", value)