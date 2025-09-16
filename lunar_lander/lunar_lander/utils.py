#!/usr/bin/env python3
import math
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
# ROS 2
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image


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

    s = np.tanh(states)  # scaled to [-1,1]
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
                img[n, y1:y2, x0:x1] = 255  # positive values → white bars above center
            else:
                y1, y2 = y0, y0 + h
                img[n, y1:y2, x0:x1] = 160  # negative values → gray bars below center
            x += bar_w + gap
    return img


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
