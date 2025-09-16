import os
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

# Paths
now = datetime.now()
date, hour, minute = now.strftime("%Y%m%d"), now.strftime("%H"), now.strftime("%M")
log_dir = f"/app/src/cartpole/data/logs/PPO/cartpole_{date}_{hour}_{minute}"
run_dir = f"/app/src/cartpole/data/model/PPO/cartpole_{date}_{hour}_{minute}"
os.makedirs(log_dir, exist_ok=True); os.makedirs(run_dir, exist_ok=True)
model_path = os.path.join(run_dir, "model")

print("Training CartPole-v1...")

# ----- Vectorized Training -----
n_envs = 8
venv = make_vec_env("CartPole-v1", n_envs=n_envs)

new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

model = PPO(
    "MlpPolicy",
    venv,
    verbose=1,
    n_steps=128,            # per env -> 128*8 = 1024 samples/rollout
    batch_size=256,         # divisor of 1024
    learning_rate=3e-4,
    gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.0, vf_coef=0.5, n_epochs=10,
    tensorboard_log=log_dir,
)
model.set_logger(new_logger)

model.learn(total_timesteps=100_000, progress_bar=True)
model.save(model_path)
venv.close()

print("Evaluating CartPole-v1...")

# ----- Simple evaluation (no VecEnv needed) -----
eval_env = gym.make("CartPole-v1", render_mode="human")
obs, info = eval_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    if terminated or truncated:
        obs, info = eval_env.reset()
eval_env.close()
