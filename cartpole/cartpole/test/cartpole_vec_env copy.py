import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

print("Training CartPole-v1...")

# ----- Vectorized Training -----
n_envs = 8
venv = make_vec_env("CartPole-v1", n_envs=n_envs)

model = PPO(
    "MlpPolicy",
    venv,
    verbose=1,
    n_steps=128,            # per env -> 128*8 = 1024 samples/rollout
    batch_size=256,         # divisor of 1024
    learning_rate=3e-4,
    gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.0, vf_coef=0.5, n_epochs=10,
)

model.learn(total_timesteps=100_000, progress_bar=True)
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
