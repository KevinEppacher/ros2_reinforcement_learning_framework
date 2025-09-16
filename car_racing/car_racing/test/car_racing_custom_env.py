#!/usr/bin/env python3
import os
from datetime import datetime
from typing import Optional, List, Callable
import math

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import ClipAction, TimeLimit, RecordEpisodeStatistics

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# ROS 2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image

# ---------------- Utils (common) ----------------
CROP_BOTTOM_PX = 12

def _tile_gray_images(batch_hw: np.ndarray) -> np.ndarray:
    """
    batch_hw: (N, H, W) mono8 -> kachelweise zusammensetzen.
    """
    n, h, w = batch_hw.shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for idx in range(n):
        r = idx // cols
        c = idx % cols
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = batch_hw[idx]
    return canvas

# ---------------- Clear Episodenlogs (VecEnv) ----------------
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

        # VecEnv-Formate säubern
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

# ---------------- Discrete Action Wrapper ----------------
class DiscretizeAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, Box) and env.action_space.shape == (3,)
        self._lut = [
            np.array([ 0.0, 0.0, 0.0], dtype=np.float32),  # noop
            np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # left hard
            np.array([ 1.0, 0.0, 0.0], dtype=np.float32),  # right hard
            np.array([ 0.0, 1.0, 0.0], dtype=np.float32),  # accel hard
            np.array([ 0.0, 0.0, 1.0], dtype=np.float32),  # brake hard
            np.array([-0.5, 0.5, 0.0], dtype=np.float32),  # left soft + accel
            np.array([ 0.5, 0.5, 0.0], dtype=np.float32),  # right soft + accel
            np.array([-1.0, 0.5, 0.0], dtype=np.float32),  # left hard + accel soft
            np.array([ 1.0, 0.5, 0.0], dtype=np.float32),  # right hard + accel soft
            np.array([-0.5, 0.0, 0.7], dtype=np.float32),  # left soft + brake
            np.array([ 0.5, 0.0, 0.7], dtype=np.float32),  # right soft + brake
        ]
        self.action_space = gym.spaces.Discrete(len(self._lut))
    def action(self, a):
        if isinstance(a, np.ndarray): a = int(a.item())
        return self._lut[int(a)]

# ---------------- Gray + Crop + RewardClip ----------------
class ToGray(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, shape=(h, w, 1), dtype=np.uint8)
    def observation(self, obs):
        gray = (0.299*obs[...,0] + 0.587*obs[...,1] + 0.114*obs[...,2]).astype(np.uint8)
        return gray[..., None]

class CropBottom(gym.ObservationWrapper):
    def __init__(self, env, crop_pixels=12):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.crop = int(crop_pixels)
        self.observation_space = gym.spaces.Box(0, 255, shape=(h-self.crop, w, c), dtype=np.uint8)
    def observation(self, obs):
        return obs[:-self.crop, :, :]

class ClipReward(gym.Wrapper):
    def __init__(self, env, rmin=-1.0, rmax=1.0):
        super().__init__(env); self.rmin=rmin; self.rmax=rmax
    def step(self, action):
        obs, env_r, term, trunc, info = self.env.step(action)
        r = float(np.clip(env_r, self.rmin, self.rmax))
        info = dict(info); info["env_reward"]=float(env_r); info["reward_clipped"]=r
        return obs, r, term, trunc, info

# ---------------- Offtrack Timeout (Info-basiert) ----------------
class OfftrackTimeout(gym.Wrapper):
    def __init__(self, env, max_off_steps=200, offroad_threshold=1e-3):
        super().__init__(env)
        self.max_off=int(max_off_steps); self.th=float(offroad_threshold); self._off=0
    def reset(self, **kw):
        self._off=0; return self.env.reset(**kw)
    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        info = dict(info)
        if "on_track" in info:
            off = not bool(info["on_track"])
        elif "offroad" in info:
            off = float(info["offroad"]) > self.th
        else:
            off = False
        self._off = 0 if not off else self._off + 1
        if self._off >= self.max_off:
            trunc = True; info["offtrack_timeout"]=True; self._off=0
        return obs, r, term, trunc, info

# ---------------- ROS Publisher Node ----------------
class CarRacingObsPublisher(Node):
    def __init__(self, topic="/car_racing/obs_grid", frame_id="camera"):
        super().__init__("car_racing_obs_publisher")
        self.pub = self.create_publisher(Image, topic, 1)
        self.frame_id = frame_id
    def publish_mono_grid(self, img_mono8: np.ndarray):
        h, w = img_mono8.shape
        msg = Image()
        msg.header = Header(); msg.header.stamp = self.get_clock().now().to_msg(); msg.header.frame_id = self.frame_id
        msg.height = h; msg.width = w; msg.encoding = "mono8"; msg.is_bigendian = 0; msg.step = w
        msg.data = img_mono8.tobytes()
        self.pub.publish(msg)

# ---------------- SB3 → ROS Vec Publisher ----------------
class StableBaselinesVecPublisher(BaseCallback):
    """
    Nimmt VecEnv-Observation (nach VecFrameStack, channels_first),
    extrahiert den letzten Frame jeder Env -> (N,H,W) mono8,
    kachelt zu einem Grid und publiziert.
    """
    def __init__(self, ros_node: CarRacingObsPublisher, every: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.node = ros_node
        self.every = every
        self.n_envs = None

    def _on_training_start(self) -> None:
        self.n_envs = getattr(self.training_env, "num_envs", 1)

    def _on_step(self) -> bool:
        # Klarer Step-Print optional
        actions = self.locals.get("clipped_actions", self.locals["actions"])
        rewards = self.locals["rewards"]
        a = np.atleast_2d(actions) if np.ndim(actions) <= 1 else actions
        r = np.atleast_1d(rewards)
        if a.shape[0] != len(r):
            a = np.tile(a, (len(r), 1))
        print(f"t={self.num_timesteps} | mean_r={float(np.mean(r)):+.3f}")

        if self.every > 0 and (self.num_timesteps % self.every == 0):
            obs = self.locals["new_obs"]  # shape: (N, C_stack, H, W)
            x = np.asarray(obs)
            if x.ndim != 4:
                return True
            # letzten Frame je Env (Stack last)
            last = x[:, -1, :, :]  # (N,H,W)
            # zu uint8
            if last.dtype != np.uint8:
                last = np.clip(last, 0, 255).astype(np.uint8)
            grid = _tile_gray_images(last)
            self.node.publish_mono_grid(grid)
        return True

# ---------------- Env-Fabriken (VEC) ----------------
def make_env_thunk(render_mode=None, step_penalty=0.0):
    def _thunk():
        e = gym.make("CarRacing-v2", render_mode=render_mode, domain_randomize=False)
        e = ClipAction(e)
        e = DiscretizeAction(e)
        e = ToGray(e)
        e = CropBottom(e, crop_pixels=CROP_BOTTOM_PX)
        e = ClipReward(e, rmin=-1.0, rmax=1.0)
        e = OfftrackTimeout(e, max_off_steps=200, offroad_threshold=1e-3)
        e = RecordEpisodeStatistics(e)
        e = TimeLimit(e, max_episode_steps=500)
        return e
    return _thunk

def make_train_vecenv(n_envs=6):
    venv = make_vec_env(make_env_thunk(render_mode=None), n_envs=n_envs, vec_env_cls=SubprocVecEnv,
                        vec_env_kwargs={"start_method": "forkserver"})
    venv = VecTransposeImage(venv)                                  # (N,C,H,W)
    venv = VecFrameStack(venv, n_stack=4, channels_order="first")   # (N,4,H,W)
    # Reward normalization only; leave images as uint8 for the CNN
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_obs=10.0, gamma=0.99)
    return venv

def make_eval_vecenv():
    venv = make_vec_env(make_env_thunk(render_mode="human"), n_envs=1,
                        vec_env_cls=SubprocVecEnv,
                        vec_env_kwargs={"start_method": "forkserver"})
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=4, channels_order="first")
    return venv

# ---------------- Main ----------------
def main():
    now = datetime.now().strftime("%Y%m%d_%H_%M")
    log_dir = f"/app/src/car_racing/data/logs/PPO/car_racing_{now}"
    run_dir = f"/app/src/car_racing/data/model/PPO/car_racing_{now}"
    os.makedirs(log_dir, exist_ok=True); os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "model")

    print("Training CarRacing-v2 (VecEnv + reward norm + tuned PPO)…")
    venv = make_train_vecenv(n_envs=6)
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    model = PPO(
        "CnnPolicy",
        venv,
        verbose=1,
        n_steps=2048,                 # per env horizon ↑
        batch_size=2048,              # multiple of n_envs*n_steps
        learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.0,                 # start low
        vf_coef=0.5, n_epochs=10,
        use_sde=False,
        tensorboard_log=log_dir,
        policy_kwargs=dict(normalize_images=True),
    )
    model.set_logger(logger)

    # Build ROS AFTER VecEnv to avoid copying handles into workers
    rclpy.init()
    node = CarRacingObsPublisher(topic="/car_racing/obs_grid", frame_id="camera")

    callback = CallbackList([
        StableBaselinesVecPublisher(node, every=25),
        EpisodePrinter(print_env='auto', print_actions=True),
    ])

    try:
        model.learn(total_timesteps=1_200_000, progress_bar=True, log_interval=1, callback=callback)
        model.save(model_path)
        # Save VecNormalize statistics (reward scaler)
        venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
    finally:
        venv.close()
        node.destroy_node()
        rclpy.shutdown()

    # Optional pause before opening the window
    print("\n[PAUSE] Press Enter to start evaluation...", flush=True)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print("Evaluation canceled."); return

    print("Evaluating (VecEnv, render_mode=human)…")
    eval_env = make_eval_vecenv()
    # Load reward-normalization stats
    eval_env = VecNormalize.load(os.path.join(run_dir, "vecnormalize.pkl"), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        if dones[0]:
            obs = eval_env.reset()
    eval_env.close()
    print("Done.")

if __name__ == "__main__":
    main()
