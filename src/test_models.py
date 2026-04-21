import torch
from models import Actor, Critic, ActorCritic

obs_dim = 11
act_dim = 3
batch_size = 4

obs = torch.randn(batch_size, obs_dim)

actor = Actor(obs_dim, act_dim)
critic = Critic(obs_dim)
ac = ActorCritic(obs_dim, act_dim)

mean, std = actor(obs)
print("Actor mean shape:", mean.shape)
print("Actor std shape:", std.shape)

action, log_prob = actor.sample_action(obs)
print("Sampled action shape:", action.shape)
print("Log prob shape:", log_prob.shape)

value = critic(obs)
print("Critic value shape:", value.shape)

a2, lp2, v2 = ac.step(obs)
print("ActorCritic action shape:", a2.shape)
print("ActorCritic log_prob shape:", lp2.shape)
print("ActorCritic value shape:", v2.shape)