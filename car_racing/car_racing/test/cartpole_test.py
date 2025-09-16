#!/usr/bin/env python3
# coding: utf-8
"""
ROS2 node that trains PPO on CartPole and runs a short evaluation rollout.
- Uses Gymnasium + Stable-Baselines3
- Logs via ROS2 logger only (no prints)
"""
import rclpy
from rclpy.node import Node

import gymnasium as gym
from stable_baselines3 import PPO
import torch


class CartPolePPONode(Node):
    def __init__(self):
        super().__init__("cartpole_ppo_node")

        # -------- parameters --------
        self.env_id      = str(self.declare_parameter("env_id", "CartPole-v1").value)
        self.timesteps   = int(self.declare_parameter("timesteps", 10_000).value)
        self.n_steps     = int(self.declare_parameter("n_steps", 2048).value)
        self.batch_size  = int(self.declare_parameter("batch_size", 256).value)
        self.lr          = float(self.declare_parameter("learning_rate", 3e-4).value)
        self.seed        = int(self.declare_parameter("seed", 0).value)
        self.progress    = bool(self.declare_parameter("progress_bar", True).value)
        self.eval_steps  = int(self.declare_parameter("eval_steps", 1000).value)
        self.render_mode = str(self.declare_parameter("render_mode", "human").value)  # "rgb_array" or "human"
        self.use_cuda    = bool(self.declare_parameter("gpu", True).value)

        # -------- device --------
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        if device == "cuda":
            i = torch.cuda.current_device()
            name = torch.cuda.get_device_name(i)
            self.get_logger().info(f"[device] cuda:{i} {name}")
        else:
            self.get_logger().info("[device] cpu")

        # -------- env --------
        self.get_logger().info(f"[env] make '{self.env_id}' render_mode={self.render_mode}")
        self.env = gym.make(self.env_id, render_mode=self.render_mode)
        self.env.reset(seed=self.seed)

        # -------- model --------
        self.get_logger().info("[train] building PPO(MlpPolicy)")
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,                 # SB3 internal logging; TB compatible
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            learning_rate=self.lr,
            device=device,
        )

    def run(self):
        # -------- train --------
        self.get_logger().info(f"[train] timesteps={self.timesteps}")
        use_tty = False  # progress bar often noisy under ROS; set True if desired
        self.model.learn(total_timesteps=self.timesteps, progress_bar=self.progress and use_tty)
        self.get_logger().info("[train] done")

        # -------- evaluate --------
        self.get_logger().info(f"[eval] steps={self.eval_steps} deterministic=True")
        vec_env = self.model.get_env()
        obs = vec_env.reset()
        total = 0.0
        for t in range(self.eval_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total += float(reward[0]) if hasattr(reward, "__len__") else float(reward)
            # optional render (works if env supports window with chosen render_mode)
            try:
                vec_env.render()
            except Exception:
                pass
            if (hasattr(done, "__len__") and done[0]) or (not hasattr(done, "__len__") and done):
                self.get_logger().info(f"[eval] episode end at t={t+1} return={total:.3f}")
                break
        self.get_logger().info(f"[eval] total_return={total:.3f}")

        # cleanup
        try:
            self.env.close()
        except Exception:
            pass


def main():
    rclpy.init()
    node = CartPolePPONode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
