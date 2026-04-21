import torch
from ma_ppo_agent import MAPPOAgent

local_obs_dim = 18
joint_obs_dim = 54
act_dim = 5

agent = MAPPOAgent(
    local_obs_dim=local_obs_dim,
    joint_obs_dim=joint_obs_dim,
    act_dim=act_dim,
)

local_obs = [0.1] * local_obs_dim
joint_obs = [0.2] * joint_obs_dim

action, log_prob, value = agent.select_action(local_obs, joint_obs)

print("Action:", action)
print("Action shape:", action.shape)
print("Log prob:", log_prob)
print("Value:", value)

batch_size = 8

data = {
    "local_obs": torch.randn(batch_size, local_obs_dim),
    "joint_obs": torch.randn(batch_size, joint_obs_dim),
    "actions": torch.rand(batch_size, act_dim),
    "log_probs": torch.randn(batch_size),
    "advantages": torch.randn(batch_size),
    "returns": torch.randn(batch_size),
}

result = agent.update(data, update_epochs=3)

print("Update result:", result)