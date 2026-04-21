from buffer import RolloutBuffer

buffer = RolloutBuffer()

# Add 5 fake transitions
for i in range(5):
    obs = [0.1] * 11
    action = [0.2] * 3
    reward = 1.0
    done = 0.0
    value = 0.5
    log_prob = -0.7

    buffer.add(obs, action, reward, done, value, log_prob)

# Assume last state value is 0.3
buffer.compute_advantages_and_returns(last_value=0.3)

data = buffer.get_tensors()

print("Buffer size:", buffer.size())
print("Obs shape:", data["obs"].shape)
print("Actions shape:", data["actions"].shape)
print("Log probs shape:", data["log_probs"].shape)
print("Values shape:", data["values"].shape)
print("Advantages shape:", data["advantages"].shape)
print("Returns shape:", data["returns"].shape)

print("Advantages:", data["advantages"])
print("Returns:", data["returns"])