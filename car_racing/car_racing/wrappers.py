# wrappers_and_callbacks.py
from __future__ import annotations
import math
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics, ClipAction
from stable_baselines3.common.callbacks import BaseCallback

# ---------- utils ----------
def tile_gray_images(batch_hw: np.ndarray) -> np.ndarray:
    """Tile a batch of mono images (N,H,W) into a single grid image."""
    n, h, w = batch_hw.shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for idx in range(n):
        r = idx // cols
        c = idx % cols
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = batch_hw[idx]
    return canvas

# ---------- action wrapper ----------
class DiscretizeAction(gym.ActionWrapper):
    """Map a discrete action id to a 3-dim continuous action vector via LUT."""
    def __init__(self, env: gym.Env):
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
        if isinstance(a, np.ndarray):
            a = int(a.item())
        return self._lut[int(a)]

# ---------- observation wrappers ----------
class ToGray(gym.ObservationWrapper):
    """Convert RGB (H,W,3) to mono8 (H,W,1)."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, shape=(h, w, 1), dtype=np.uint8)

    def observation(self, obs):
        gray = (0.299*obs[...,0] + 0.587*obs[...,1] + 0.114*obs[...,2]).astype(np.uint8)
        return gray[..., None]

class CropBottom(gym.ObservationWrapper):
    """Crop bottom N pixels from mono or RGB observation."""
    def __init__(self, env: gym.Env, crop_pixels: int = 12):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.crop = int(crop_pixels)
        self.observation_space = gym.spaces.Box(0, 255, shape=(h-self.crop, w, c), dtype=np.uint8)

    def observation(self, obs):
        return obs[:-self.crop, :, :]

# ---------- reward wrappers ----------
class ClipReward(gym.Wrapper):
    """Clip rewards to [rmin, rmax] and expose original reward in info."""
    def __init__(self, env: gym.Env, rmin: float = -1.0, rmax: float = 1.0):
        super().__init__(env)
        self.rmin = float(rmin)
        self.rmax = float(rmax)

    def step(self, action):
        obs, env_r, term, trunc, info = self.env.step(action)
        r = float(np.clip(env_r, self.rmin, self.rmax))
        info = dict(info)
        info["env_reward"] = float(env_r)
        info["reward_clipped"] = r
        return obs, r, term, trunc, info

class OfftrackTimeout(gym.Wrapper):
    """Terminate episode early if off-track for too many consecutive steps."""
    def __init__(self, env: gym.Env, max_off_steps: int = 200, offroad_threshold: float = 1e-3):
        super().__init__(env)
        self.max_off = int(max_off_steps)
        self.th = float(offroad_threshold)
        self._off = 0

    def reset(self, **kw):
        self._off = 0
        return self.env.reset(**kw)

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
            trunc = True
            info["offtrack_timeout"] = True
            self._off = 0
        return obs, r, term, trunc, info

# ---------- training console callbacks ----------
class EpisodePrinter(BaseCallback):
    """Pretty-print per-step info for selected env(s) in a VecEnv."""
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

# ---------- ROS publishing callback ----------
class StableBaselinesVecPublisher(BaseCallback):
    """
    Publish a tiled mono8 grid from the last frame of each env in the VecEnv stack.
    Expects a ROS node with method `publish_mono_grid(mono_img: np.ndarray)`.
    """
    def __init__(self, ros_node, every: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.node = ros_node
        self.every = int(every)
        self.n_envs = None

    def _on_training_start(self) -> None:
        self.n_envs = getattr(self.training_env, "num_envs", 1)

    def _on_step(self) -> bool:
        # quick scalar status
        rewards = np.atleast_1d(self.locals["rewards"])
        print(f"t={self.num_timesteps} | mean_r={float(np.mean(rewards)):+.3f}")

        if self.every > 0 and (self.num_timesteps % self.every == 0):
            obs = self.locals.get("new_obs", None)  # expected (N, C_stack, H, W)
            if obs is None:
                return True
            x = np.asarray(obs)
            if x.ndim != 4 or x.shape[1] < 1:
                return True
            last = x[:, -1, :, :]  # (N,H,W)
            if last.dtype != np.uint8:
                last = np.clip(last, 0, 255).astype(np.uint8)
            grid = tile_gray_images(last)
            try:
                self.node.publish_mono_grid(grid)
            except Exception as e:
                print(f"[VecPublisher] publish failed: {e}")
        return True
