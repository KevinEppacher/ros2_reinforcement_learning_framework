#!/usr/bin/env python3
import os, math
from datetime import datetime
from typing import List, Callable

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# ROS 2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image


# ---------- Episode Logging (VecEnv) ----------
class EpisodePrinter(BaseCallback):
    def __init__(self, print_env='auto', print_actions=True):
        super().__init__()
        self.print_env = print_env
        self.print_actions = print_actions

    def _on_training_start(self) -> None:
        self.n_envs = getattr(self.training_env, "num_envs", 1)
        self.ep_idx = np.zeros(self.n_envs, dtype=int)
        self.ep_len = np.zeros(self.n_envs, dtype=int)
        self.ep_ret = np.zeros(self.n_envs, dtype=float)
        if self.print_env == 'all':
            self.env_ids = list(range(self.n_envs))
        elif self.print_env == 'auto':
            self.env_ids = [0] if self.n_envs > 1 else [0]
        else:
            self.env_ids = [int(self.print_env)]
        for i in self.env_ids:
            self.ep_idx[i] = 1
            print(f"--- Episode {self.ep_idx[i]} (env {i}) START ---")

    def _on_step(self) -> bool:
        actions = self.locals.get("clipped_actions", self.locals["actions"])
        rewards = self.locals["rewards"]
        dones   = self.locals["dones"]
        infos   = self.locals["infos"]

        a = np.atleast_2d(actions) if np.ndim(actions) <= 1 else actions
        r = np.atleast_1d(rewards)
        d = np.atleast_1d(dones)
        if a.shape[0] != len(r):
            a = np.tile(a, (len(r), 1))

        for i in self.env_ids:
            self.ep_len[i] += 1
            self.ep_ret[i] += float(r[i])
            a_list = np.asarray(a[i]).reshape(-1).tolist() if self.print_actions else "hidden"
            print(f"[E{self.ep_idx[i]} S{self.ep_len[i]} | env {i}] a={a_list} r={float(r[i]):+.3f} done={bool(d[i])}")
            if d[i]:
                ep_info = infos[i].get("episode") if isinstance(infos, (list, tuple)) and i < len(infos) else None
                ep_r = ep_info["r"] if ep_info else self.ep_ret[i]
                ep_l = ep_info["l"] if ep_info else self.ep_len[i]
                trunc = bool(infos[i].get("TimeLimit.truncated", False)) if isinstance(infos, (list, tuple)) else False
                print(f">>> Episode {self.ep_idx[i]} (env {i}) finished: return={ep_r:.2f}, length={ep_l}, truncated={trunc}")
                self.ep_idx[i] += 1
                self.ep_len[i] = 0
                self.ep_ret[i] = 0.0
                print(f"--- Episode {self.ep_idx[i]} (env {i}) START ---")
        return True


# ---------- ROS Publisher ----------
class LanderObsPublisher(Node):
    def __init__(self, topic="/lunar_lander/obs_grid", frame_id="lander"):
        super().__init__("lunar_lander_obs_publisher")
        self.pub = self.create_publisher(Image, topic, 1)
        self.frame_id = frame_id

    def publish_mono_grid(self, img_mono8: np.ndarray):
        h, w = img_mono8.shape
        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = h
        msg.width = w
        msg.encoding = "mono8"
        msg.is_bigendian = 0
        msg.step = w
        msg.data = img_mono8.tobytes()
        self.pub.publish(msg)


# ---------- VecEnv → ROS Callback ----------
def _tile_gray_images(batch_hw: np.ndarray) -> np.ndarray:
    n, h, w = batch_hw.shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for idx in range(n):
        r = idx // cols
        c = idx % cols
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = batch_hw[idx]
    return canvas

def _states_to_bars(states: np.ndarray, H=64, W=128) -> np.ndarray:
    """
    states: (N, D) -> mono8 images (N,H,W).
    Draws bars symmetrically around the center line. tanh scaling.
    """
    N, D = states.shape
    y0 = H // 2
    img = np.zeros((N, H, W), dtype=np.uint8)
    # bar width/gaps
    bar_w = max(2, W // (2 * D))
    gap = max(1, bar_w // 2)
    total = D * bar_w + (D + 1) * gap
    x_off = max(0, (W - total) // 2)

    s = np.tanh(states)  # [-1,1]
    max_h = y0 - 2

    for n in range(N):
        x = x_off + gap
        # center line
        img[n, y0-1:y0+1, :] = 80
        for j in range(D):
            v = float(s[n, j])
            h = int(abs(v) * max_h)
            x0, x1 = x, x + bar_w
            if v >= 0:
                y1, y2 = y0 - h, y0
                img[n, y1:y2, x0:x1] = 255
            else:
                y1, y2 = y0, y0 + h
                img[n, y1:y2, x0:x1] = 160
            x += bar_w + gap
    return img

class StableBaselinesVecPublisher(BaseCallback):
    def __init__(self, ros_node: LanderObsPublisher, every: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.node = ros_node
        self.every = every
        self.n_envs = None

    def _on_training_start(self) -> None:
        self.n_envs = getattr(self.training_env, "num_envs", 1)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        print(f"t={self.num_timesteps} | mean_r={float(np.mean(rewards)):+.3f}")

        if self.every > 0 and (self.num_timesteps % self.every == 0):
            obs = np.asarray(self.locals["new_obs"])  # (N,D)
            if obs.ndim != 2:
                return True
            imgs = _states_to_bars(obs, H=64, W=128)   # (N,64,128)
            grid = _tile_gray_images(imgs)
            self.node.publish_mono_grid(grid)
        return True


# ---------- Env factories ----------
def make_env_thunk(render_mode=None):
    def _thunk():
        e = gym.make("LunarLander-v2", render_mode=render_mode)
        e = RecordEpisodeStatistics(e)
        e = TimeLimit(e, max_episode_steps=1000)  # explicit
        return e
    return _thunk

def make_train_vecenv(n_envs=8):
    return make_vec_env(make_env_thunk(render_mode=None), n_envs=n_envs, vec_env_cls=SubprocVecEnv)

def make_eval_vecenv():
    # 1 Env with window
    return make_vec_env(make_env_thunk(render_mode="human"), n_envs=1)


# ---------- Main ----------
def main():
    # Logs/Model
    stamp = datetime.now().strftime("%Y%m%d_%H_%M")
    log_dir = f"/app/src/lunar_lander/data/logs/PPO/lunar_lander_{stamp}"
    run_dir = f"/app/src/lunar_lander/data/model/PPO/lunar_lander_{stamp}"
    os.makedirs(log_dir, exist_ok=True); os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "model")

    print("Training LunarLander-v2 (VecEnv, episodic logs, obs-grid ROS)…")

    # VEC ENV first, possibly with forkserver
    venv = make_vec_env(make_env_thunk(render_mode=None), n_envs=8,
                        vec_env_cls=SubprocVecEnv,
                        vec_env_kwargs={"start_method": "forkserver"})
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=2048, batch_size=512, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.0, vf_coef=0.5, n_epochs=10,
        tensorboard_log=log_dir,
    )
    model.set_logger(logger)

    # ROS AFTER the VEC ENV
    rclpy.init()
    node = LanderObsPublisher(topic="/lunar_lander/obs_grid", frame_id="lander")

    callback = CallbackList([
        StableBaselinesVecPublisher(node, every=50),
        EpisodePrinter(print_env='auto', print_actions=True),
    ])

    try:
        model.learn(total_timesteps=500_000, progress_bar=True, log_interval=1, callback=callback)
        model.save(model_path)
        venv.save(os.path.join(run_dir, "vecnormalize.pkl"))  # save now
    finally:
        venv.close()
        node.destroy_node()
        rclpy.shutdown()

    # --- wait for user input before evaluation ---
    print("\n[PAUSE] Press Enter to start evaluation...", flush=True)
    try:
        input()  # blocks until Enter is pressed
    except (EOFError, KeyboardInterrupt):
        print("Evaluation canceled.")
        return  # exit main() cleanly

    # Evaluation
    print("Evaluating (VecEnv, render_mode=human)…")
    eval_env = make_vec_env(make_env_thunk(render_mode="human"), n_envs=1,
                            vec_env_cls=SubprocVecEnv,
                            vec_env_kwargs={"start_method": "forkserver"})
    eval_env = VecNormalize.load(os.path.join(run_dir, "vecnormalize.pkl"), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        if dones[0]:
            obs = eval_env.reset()
    eval_env.close()
    print("Done.")



if __name__ == "__main__":
    main()
