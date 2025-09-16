import os
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

# -------- Reward Wrapper --------
class CartPoleRewardWrapper(gym.Wrapper):
    def __init__(self, env,
                 w_env=1.0,
                 kx=0.25, kth=1.0, kxd=0.02, ktd=0.05,
                 x_max=2.4, theta_max=np.deg2rad(12),
                 xdot_ref=2.0, thetadot_ref=2.0,
                 potential=False, gamma=0.99):
        super().__init__(env)
        self.w_env=float(w_env)
        self.kx, self.kth, self.kxd, self.ktd = map(float, (kx, kth, kxd, ktd))
        self.x_max, self.theta_max = float(x_max), float(theta_max)
        self.xdot_ref, self.thetadot_ref = float(xdot_ref), float(thetadot_ref)
        self.potential=bool(potential); self.gamma=float(gamma)
        self._last_phi=None

    def _phi(self, obs):
        x, xdot, th, thdot = map(float, obs)
        return (self.kx*(x/self.x_max)**2
              + self.kth*(th/self.theta_max)**2
              + self.kxd*(xdot/self.xdot_ref)**2
              + self.ktd*(thdot/self.thetadot_ref)**2)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_phi = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        env_r = float(r)
        phi = self._phi(obs)
        shaped = self.w_env*env_r - phi
        if self.potential:
            shaped += self.gamma*phi - (self._last_phi if self._last_phi is not None else phi)
        self._last_phi = phi
        info = dict(info)
        info["env_reward"]=env_r; info["phi"]=float(phi); info["reward_shaped_step"]=float(shaped)
        return obs, float(shaped), terminated, truncated, info

# -------- Pfade --------
now = datetime.now()
date, hour, minute = now.strftime("%Y%m%d"), now.strftime("%H"), now.strftime("%M")
log_dir = f"/app/src/cartpole/data/logs/PPO/custom_cartpole_{date}_{hour}_{minute}"
run_dir = f"/app/src/cartpole/data/model/PPO/custom_cartpole_{date}_{hour}_{minute}"
os.makedirs(log_dir, exist_ok=True); os.makedirs(run_dir, exist_ok=True)
model_path = os.path.join(run_dir, "model")

# -------- Nur die ROHE Env checken --------
raw_env = gym.make("CartPole-v1")
check_env(raw_env)
raw_env.close()

print("Training CartPole-v1...")

# -------- Vectorized Training --------
def make_env_fn():
    e = gym.make("CartPole-v1")
    e = RecordEpisodeStatistics(e)
    e = CartPoleRewardWrapper(e,
        w_env=1.0, kx=0.25, kth=1.0, kxd=0.02, ktd=0.05,
        potential=False)
    return e

n_envs = 8
venv = make_vec_env(make_env_fn, n_envs=n_envs)

new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

model = PPO(
    "MlpPolicy",
    venv,
    verbose=1,
    n_steps=128,          # for each Env -> 128*8 = 1024 Samples/Rollout
    batch_size=256,       
    learning_rate=3e-4,
    gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.0, vf_coef=0.5, n_epochs=10,
    tensorboard_log=log_dir,
)
model.set_logger(new_logger)

model.learn(total_timesteps=500_000, progress_bar=True)
model.save(model_path)
venv.close()

print("Evaluating CartPole-v1...")

# -------- Easy Evaluation (no VecEnv needed) --------
eval_env = gym.make("CartPole-v1", render_mode="human")
obs, info = eval_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    if terminated or truncated:
        obs, info = eval_env.reset()
eval_env.close()
