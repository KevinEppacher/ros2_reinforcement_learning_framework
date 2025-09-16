import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
import os
from datetime import datetime

# date, hour, minute
now = datetime.now()
date = now.strftime("%Y%m%d")
hour = now.strftime("%H")
minute = now.strftime("%M")
log_dir = f"/app/src/cartpole/data/logs/PPO/cartpole_{date}_{hour}_{minute}"
run_dir = f"/app/src/cartpole/data/model/PPO/cartpole_{date}_{hour}_{minute}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(run_dir, exist_ok=True)

model_path = os.path.join(run_dir, "model")

print("Training CartPole-v1...")

# Headless training (no window)
train_env = gym.make("CartPole-v1")

# Configure logger (including Tensorboard)
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    n_steps=2048,
    batch_size=256,
    learning_rate=3e-4,
    tensorboard_log=log_dir,
)

# Replace logger
model.set_logger(new_logger)

model.learn(total_timesteps=100_000, progress_bar=True)

# Save model
model.save(model_path)

train_env.close()

print("Evaluating CartPole-v1...")

# Evaluation with visible window
eval_env = gym.make("CartPole-v1", render_mode="human")

obs, info = eval_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    if terminated or truncated:
        obs, info = eval_env.reset()

eval_env.close()
