#!/usr/bin/env python3
from dataclasses import dataclass
import numpy as np
from rclpy.node import Node
from rl_core.plugins import load_task_spec
from rl_trainers.configs import Config

class EnvManager:
    """Creates single or vectorized envs via task plugin."""
    def __init__(self, node: Node, cfg: Config):
        self.node = node
        self.cfg = cfg
        self.env_build = load_task_spec(cfg.meta.task).build(node=node)
        self.train_env = None
        self.eval_env = None

    def make_train(self):
        n = int(self.cfg.train.n_envs)
        if n > 1 and getattr(self.env_build, "make_train_vecenv", None):
            self.train_env = self.env_build.make_train_vecenv(n)
            self.node.get_logger().info(f"Created custom VecEnv with n_envs={n}")
        elif n > 1:
            from stable_baselines3.common.env_util import make_vec_env
            self.train_env = make_vec_env(self.env_build.make_train_env, n_envs=n)
            self.node.get_logger().info(f"Created generic VecEnv with n_envs={n}")
        else:
            self.train_env = self.env_build.make_train_env()
            self.node.get_logger().info("Created single train env")
        return self.train_env

    def make_eval(self):
        # Always prefer vec eval pipeline (even when n_envs == 1)
        if getattr(self.env_build, "make_eval_vecenv", None):
            self.eval_env = self.env_build.make_eval_vecenv()
            self.node.get_logger().info(
                f"Created eval VecEnv with n_envs={getattr(self.eval_env, 'num_envs', 1)}"
            )
        else:
            self.eval_env = self.env_build.make_eval_env()
            self.node.get_logger().info("Created single eval env")
        return self.eval_env

    def close(self):
        for attr in ("train_env", "eval_env"):
            env = getattr(self, attr, None)
            if env is not None:
                try: env.close()
                except Exception: pass
                setattr(self, attr, None)
