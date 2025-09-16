#!/usr/bin/env python3
import os
from datetime import datetime
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper, Wrapper
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList

# ROS 2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image

# Anzahl Pixel unten wegschneiden (schwarze HUD-Leiste)
CROP_BOTTOM_PX = 12

# --------- Utils ----------
def _obs_to_hwc_uint8(obs, crop_bottom_px: int = 0) -> Optional[np.ndarray]:
    x = np.asarray(obs)
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        return None
    if x.shape[0] in (1, 3) and (x.shape[2] not in (1, 3)):
        x = np.transpose(x, (1, 2, 0))
    if crop_bottom_px > 0:
        x = x[:-crop_bottom_px, :, :]
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def _grass_ratio(img_hwc_uint8: np.ndarray) -> float:
    h = img_hwc_uint8.shape[0]
    roi = img_hwc_uint8[h//2:, :, :]
    r, g, b = roi[..., 0], roi[..., 1], roi[..., 2]
    grass = (g > 110) & (g > r + 15) & (g > b + 15)
    return float(np.mean(grass))

def is_offroad(obs, info=None, grass_thresh: float = 0.35) -> bool:
    for k in ("offroad", "is_off_road", "on_track"):
        if info is not None and k in info:
            if k == "on_track":
                return not bool(info["on_track"])
            return bool(info[k])
    img = _obs_to_hwc_uint8(obs, crop_bottom_px=CROP_BOTTOM_PX)
    if img is None:
        return False
    return _grass_ratio(img) > grass_thresh

# ---------------- Clear Episodenlogs ----------------

class EpisodePrinter(BaseCallback):
    """
    Klare Episodenlogs:
      [E{ep} S{step}] a=... r=... | done? ...
      >>> Episode {ep} finished: return=..., length=...
    - Unterstützt VecEnv (n_envs>=1).
    - print_env: 'all' oder int (Index). Default: 0 wenn n_envs>1.
    """
    def __init__(self, print_env='auto', print_actions=True):
        super().__init__()
        self.print_env = print_env
        self.print_actions = print_actions

    def _on_training_start(self) -> None:
        self.n_envs = getattr(self.training_env, "num_envs", 1)
        self.ep_idx   = np.zeros(self.n_envs, dtype=int)      # Episode-Zähler je Env (1-basiert in Logs)
        self.ep_len   = np.zeros(self.n_envs, dtype=int)
        self.ep_ret   = np.zeros(self.n_envs, dtype=float)
        # Auswahl, welche Envs pro Step gedruckt werden
        if self.print_env == 'all':
            self.env_ids = list(range(self.n_envs))
        elif self.print_env == 'auto':
            self.env_ids = [0] if self.n_envs > 1 else [0]
        else:
            self.env_ids = [int(self.print_env)]

        # Startbanner
        for i in self.env_ids:
            self.ep_idx[i] = 1
            print(f"--- Episode {self.ep_idx[i]} (env {i}) START ---")

    def _on_step(self) -> bool:
        # SB3 liefert Arrays für VecEnv, Skalar für SingleEnv -> in Arrays konvertieren
        actions = self.locals.get("clipped_actions", self.locals["actions"])
        rewards = self.locals["rewards"]
        dones   = self.locals["dones"]
        infos   = self.locals["infos"]

        a = np.atleast_2d(actions) if np.ndim(actions) <= 1 else actions
        r = np.atleast_1d(rewards)
        d = np.atleast_1d(dones)

        # fehlende Shapes angleichen
        if a.shape[0] != self.n_envs:
            a = np.tile(a, (self.n_envs, 1))

        for i in self.env_ids:
            self.ep_len[i] += 1
            self.ep_ret[i] += float(r[i])

            # per-step Log
            a_list = np.asarray(a[i]).reshape(-1).tolist() if self.print_actions else "hidden"
            print(f"[E{self.ep_idx[i]} S{self.ep_len[i]} | env {i}] a={a_list} r={float(r[i]):+.3f} done={bool(d[i])}")

            # Episodenende?
            if d[i]:
                # Falls Monitor Info vorhanden, nimm das
                ep_info = infos[i].get("episode") if isinstance(infos, (list, tuple)) and i < len(infos) else None
                ep_r = ep_info["r"] if ep_info else self.ep_ret[i]
                ep_l = ep_info["l"] if ep_info else self.ep_len[i]
                trunc = bool(infos[i].get("TimeLimit.truncated", False)) if isinstance(infos, (list, tuple)) else False

                print(f">>> Episode {self.ep_idx[i]} (env {i}) finished: return={ep_r:.2f}, length={ep_l}, truncated={trunc}")
                # auf nächste Episode vorbereiten
                self.ep_idx[i] += 1
                self.ep_len[i] = 0
                self.ep_ret[i] = 0.0
                print(f"--- Episode {self.ep_idx[i]} (env {i}) START ---")
        return True

# ---------------- Discrete Action Wrapper ----------------
class DiscretizeAction(ActionWrapper):
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

# -------------- Safe Reset: auf Straße spawnen --------------
class SafeResetOnRoad(Wrapper):
    def __init__(self, env, max_resets: int = 5, settle_noop_steps: int = 2, grass_thresh: float = 0.35):
        super().__init__(env)
        self.max_resets = max_resets
        self.settle_noop_steps = settle_noop_steps
        self.grass_thresh = grass_thresh

    def reset(self, **kwargs):
        for _ in range(self.max_resets):
            obs, info = self.env.reset(**kwargs)
            for _ in range(self.settle_noop_steps):
                obs, _, term, trunc, info = self.env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                if term or trunc:
                    obs, info = self.env.reset(**kwargs)
                    break
            if not is_offroad(obs, info, self.grass_thresh):
                return obs, info
        return obs, info

# -------------- Offroad-Timeout --------------
class OfftrackTimeout(Wrapper):
    def __init__(self, env, max_off_steps: int = 180, grass_thresh: float = 0.35):
        super().__init__(env)
        self.max_off_steps = int(max_off_steps)
        self.grass_thresh = grass_thresh
        self._off = 0

    def reset(self, **kwargs):
        self._off = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        off = is_offroad(obs, info, self.grass_thresh)
        self._off = 0 if not off else self._off + 1
        if self._off >= self.max_off_steps:
            trunc = True
            info = dict(info or {})
            info["offroad_timeout"] = True
            self._off = 0
        return obs, reward, term, trunc, info

# ---------------- ClipReward for Gym ----------------

class ClipReward(gym.RewardWrapper):
    def __init__(self, env, rmin=-1.0, rmax=1.0):
        super().__init__(env); self.rmin=rmin; self.rmax=rmax
    def reward(self, r):
        return float(np.clip(r, self.rmin, self.rmax))

# ---------------- ROS Publisher Node ----------------
class CarRacingObsPublisher(Node):
    def __init__(self, topic="/car_racing/obs", frame_id="camera"):
        super().__init__("car_racing_obs_publisher")
        self.pub = self.create_publisher(Image, topic, 1)
        self.frame_id = frame_id

    def publish_img(self, img_hwc_uint8: np.ndarray):
        h, w = img_hwc_uint8.shape[:2]
        ch = img_hwc_uint8.shape[2] if img_hwc_uint8.ndim == 3 else 1
        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = h
        msg.width = w
        msg.encoding = "rgb8" if ch == 3 else "mono8"
        msg.is_bigendian = 0
        msg.step = w * (3 if ch == 3 else 1)
        msg.data = img_hwc_uint8.tobytes()
        self.pub.publish(msg)

# ---------------- SB3 → ROS Passthrough ----------------
class StableBaselinesPassthrough(BaseCallback):
    def __init__(self, ros_node: CarRacingObsPublisher, every: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.node = ros_node
        self.every = every

    def _on_step(self) -> bool:
        action = self.locals.get("clipped_actions", self.locals["actions"])
        rewards = self.locals["rewards"]
        a_list = np.asarray(action).reshape(-1).tolist()
        r_val = float(np.asarray(rewards).mean())
        print(f"Step {self.num_timesteps} | Action={a_list} | Reward={r_val:.3f}")

        if self.every > 0 and (self.num_timesteps % self.every == 0):
            obs = self.locals["new_obs"]
            img = _obs_to_hwc_uint8(obs, crop_bottom_px=CROP_BOTTOM_PX)
            if img is not None:
                self.node.publish_img(img)
        return True

# ---------------- Env-Fabrik ----------------
def make_env(render_mode="rgb_array"):
    env = gym.make("CarRacing-v2", render_mode=render_mode)
    env = DiscretizeAction(env)
    env = gym.wrappers.TimeLimit(env.unwrapped, 5000)
    env = SafeResetOnRoad(env, max_resets=5, settle_noop_steps=2, grass_thresh=0.35)
    env = OfftrackTimeout(env, max_off_steps=200, grass_thresh=0.9)
    env = ClipReward(env, rmin=-1.0, rmax=1.0)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    return env

# ---------------- Main ----------------
def main():
    rclpy.init()
    node = CarRacingObsPublisher(topic="/car_racing/obs", frame_id="camera")

    now = datetime.now().strftime("%Y%m%d_%H_%M")
    log_dir = f"/app/src/car_racing/data/logs/PPO/car_racing_{now}"
    run_dir = f"/app/src/car_racing/data/model/PPO/car_racing_{now}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "model")

    print("Training CarRacing-v2 (discrete, safe road spawn, offroad timeout, cropped obs)...")

    env = make_env()
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        use_sde=False,
        tensorboard_log=log_dir,
    )
    model.set_logger(new_logger)

    callback = CallbackList([
        StableBaselinesPassthrough(ros_node=node, every=10),  # dein Bild-Publisher
        EpisodePrinter(print_env='auto', print_actions=True)  # klare Episodenlogs
    ])

    try:
        model.learn(total_timesteps=10_000, progress_bar=True, callback=callback, log_interval=1)
        model.save(model_path)
        env.close()

        # -------- Evaluation mit GUI --------
        print("Evaluating (render_mode=human) …")
        eval_env = make_env(render_mode="human")
        n_episodes = 5
        for ep in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            ep_ret = 0.0
            steps = 0
            while not done and steps < 3000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_ret += float(reward)
                steps += 1
                done = bool(terminated or truncated)
            print(f"Episode {ep+1}/{n_episodes}: return={ep_ret:.2f}, steps={steps}, info={info}")
        eval_env.close()

    finally:
        node.destroy_node()
        rclpy.shutdown()

    print("Done.")

if __name__ == "__main__":
    main()
