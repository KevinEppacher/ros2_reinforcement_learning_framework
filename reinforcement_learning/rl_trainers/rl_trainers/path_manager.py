#!/usr/bin/env python3
import os, sys, glob, json, inspect
from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np
import rclpy
from rclpy.node import Node
from datetime import datetime

from rl_core.plugins import load_task_spec
from rl_core.algos import get_algo_defaults, make_algo
from rl_trainers import BLUE, GREEN, YELLOW, RED, BOLD, RESET
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rl_trainers.configs import TrainCfg, EvalCfg, Config

class PathManager:
    """Builds log/model paths from config and timestamp."""
    def __init__(self, cfg: Config):
        ts = datetime.now().strftime("%Y%m%d_%H_%M")
        self.timestamp = ts
        self.log_dir = os.path.join(cfg.meta.artifacts_dir, "logs", cfg.algo.name.upper(), f"{cfg.meta.task}_{ts}")
        self.run_dir = os.path.join(cfg.meta.artifacts_dir, "model", cfg.algo.name.upper(), f"{cfg.meta.task}_{ts}") \
                       if cfg.meta.mode == "train" else None
        self.model_path = os.path.join(self.run_dir, "model") if self.run_dir else None

    def ensure(self):
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
