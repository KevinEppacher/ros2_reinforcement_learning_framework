#!/usr/bin/env python3
import json, inspect
from typing import Optional
from rclpy.node import Node
from rl_core.algos import make_algo
from rl_trainers.configs import Config
from rl_trainers.path_manager import PathManager

class ModelFactory:
    """Builds or loads SB3 models with safe kwargs filtering."""
    def __init__(self, node: Node, cfg: Config, paths: PathManager):
        self.n = node; self.cfg = cfg; self.paths = paths

    def _filtered_kwargs(self, Algo):
        raw = dict(self.cfg.algo_params) if self.cfg.meta.mode == "train" else {}
        raw.update({"verbose": 1, "tensorboard_log": self.paths.log_dir, "device": self.cfg.algo.device})
        raw.pop("tensorboard", None)
        pk = raw.get("policy_kwargs", {})
        if isinstance(pk, str):
            try: pk = json.loads(pk) if pk.strip() else {}
            except Exception: pk = {}
        raw["policy_kwargs"] = pk
        allowed = set(inspect.signature(Algo.__init__).parameters.keys()); allowed.discard("self")
        return {k: v for k, v in raw.items() if k in allowed}

    def build(self, env=None, for_eval=False, load_path: Optional[str]=None):
        Algo = make_algo(self.cfg.algo.name)
        # Sanity checks for action spaces
        if env is not None:
            from gymnasium.spaces import Discrete, Box
            if self.cfg.algo.name == "dqn" and not isinstance(env.action_space, Discrete):
                raise RuntimeError("DQN requires a Discrete action space.")
            if self.cfg.algo.name in ("sac", "td3") and not isinstance(env.action_space, Box):
                raise RuntimeError(f"{self.cfg.algo.name.upper()} requires a continuous (Box) action space.")
        if for_eval:
            if not load_path: raise ValueError("load_path is required when for_eval=True.")
            return Algo.load(load_path, device=self.cfg.algo.device, print_system_info=False)
        kwargs = self._filtered_kwargs(Algo)
        return Algo(self.cfg.algo.policy, env, **kwargs)
