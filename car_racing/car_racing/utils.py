# car_racing_ros_utils.py
# Comments in English.

from __future__ import annotations
import math
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# ROS 2
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image


class CarRacingObsPublisher(Node):
    """ROS2 node that publishes mono8 images on a topic."""
    def __init__(self, topic: str = "/car_racing/obs_grid", frame_id: str = "camera"):
        super().__init__("car_racing_obs_publisher")
        self.pub = self.create_publisher(Image, topic, 1)
        self.frame_id = frame_id

    def publish_mono_grid(self, img_mono8: np.ndarray):
        img = np.asarray(img_mono8)
        if img.ndim != 2: return
        if img.dtype != np.uint8: img = np.clip(img,0,255).astype(np.uint8)
        h, w = img.shape
        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height = h; msg.width = w
        msg.encoding = "mono8"; msg.is_bigendian = 0; msg.step = w
        msg.data = img.tobytes()
        self.pub.publish(msg)

class EpisodePrinter(BaseCallback):
    """Print per-step action/reward and episode summaries for selected env(s)."""
    def __init__(self, print_env: str | int = 'auto', print_actions: bool = True):
        super().__init__()
        self.print_env = print_env
        self.print_actions = print_actions

    def _init_callback(self):
        super()._init_callback()
        print("[CB] attached:", self.__class__.__name__, flush=True)

    def _on_training_start(self):
        print("[CB] training_start:", self.__class__.__name__, flush=True)

    def _on_training_start(self) -> None:
        self.n_envs = getattr(self.training_env, "num_envs", 1)
        self.ep_idx = np.zeros(self.n_envs, dtype=int)
        self.ep_len = np.zeros(self.n_envs, dtype=int)
        self.ep_ret = np.zeros(self.n_envs, dtype=float)

        if self.print_env == 'all':
            self.env_ids = list(range(self.n_envs))
        elif self.print_env == 'auto':
            self.env_ids = [0]
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
                ep_r = float(ep_info["r"]) if ep_info else self.ep_ret[i]
                ep_l = int(ep_info["l"]) if ep_info else self.ep_len[i]
                trunc = bool(infos[i].get("TimeLimit.truncated", False)) if isinstance(infos, (list, tuple)) else False
                print(f">>> Episode {self.ep_idx[i]} (env {i}) finished: return={ep_r:.2f}, length={ep_l}, truncated={trunc}")
                self.ep_idx[i] += 1
                self.ep_len[i] = 0
                self.ep_ret[i] = 0.0
                print(f"--- Episode {self.ep_idx[i]} (env {i}) START ---")
        return True


class StableBaselinesVecPublisher(BaseCallback):
    """
    Publishes last grayscale frame grid from (Vec)Env observations.
    Works with:
      - VecEnv: (N,C,H,W) or (N,H,W,C)
      - Single env: (C,H,W) or (H,W,C)  -> auto-batched to N=1
      - With/without frame stack (takes last channel if stacked)
    """
    def __init__(self, ros_node, every: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.node = ros_node
        self.every = int(every)

    def _init_callback(self):
        super()._init_callback()
        print("[CB] attached:", self.__class__.__name__, flush=True)

    def _on_training_start(self):
        print("[CB] training_start:", self.__class__.__name__, flush=True)
        # (bei EpisodePrinter ggf. bisherige Logik + flush=True in allen print(...))

    def _on_step(self) -> bool:
        if self.every <= 0 or (self.num_timesteps % self.every) != 0:
            return True

        x = self.locals.get("new_obs", None)
        if x is None:
            return True
        x = np.asarray(x)

        # Normalize shapes to (N,C,H,W)
        if x.ndim == 2:
            # (H,W) -> (1,1,H,W)
            x = x[None, None, ...]
        elif x.ndim == 3:
            # (C,H,W) or (H,W,C)
            if x.shape[0] in (1, 3, 4):  # likely channels_first
                x = x[None, ...]  # (1,C,H,W)
            else:
                # assume channels_last (H,W,C)
                x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,C,H,W)
        elif x.ndim == 4:
            # (N,H,W,C) -> (N,C,H,W)
            if x.shape[-1] in (1, 3, 4):
                x = np.transpose(x, (0, 3, 1, 2))
        else:
            return True  # unsupported

        N, C, H, W = x.shape

        # Take last frame if stacked, else use the only channel
        last = x[:, -1, :, :] if C >= 1 else x[:, 0, :, :]

        # To uint8 mono
        if last.dtype != np.uint8:
            last = np.clip(last, 0, 255).astype(np.uint8)

        # Tile into grid
        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))
        grid = np.zeros((rows * H, cols * W), dtype=np.uint8)
        for i in range(N):
            r, c = divmod(i, cols)
            grid[r*H:(r+1)*H, c*W:(c+1)*W] = last[i]

        # Publish
        self.node.publish_mono_grid(grid)
        return True