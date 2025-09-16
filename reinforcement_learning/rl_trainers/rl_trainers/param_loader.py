#!/usr/bin/env python3
import json
from typing import Any, Dict
from rclpy.node import Node
from datetime import datetime

from rl_core.algos import get_algo_defaults
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rl_trainers.configs import TrainCfg, EvalCfg, Config, AlgoCfg, MetaCfg

# ---------------------------- Helpers ----------------------------
class ParamLoader:
    """Reads ROS params into typed config. Declares defaults first."""
    def __init__(self, node: Node):
        self.n = node

    def load(self) -> Config:
        n = self.n
        # Declare defaults
        n.declare_parameter("metadata.namespace", "default")
        n.declare_parameter("metadata.mode", "train")
        n.declare_parameter("metadata.task", "default")
        n.declare_parameter("metadata.artifacts_dir", "default")

        n.declare_parameter("algorithm.name", "default")
        n.declare_parameter("algorithm.device", "auto")
        n.declare_parameter("algorithm.policy", "MlpPolicy")

        n.declare_parameter("train.total_timesteps", 500_000)
        n.declare_parameter("train.save_freq", 100_000)
        n.declare_parameter("train.progress", True)
        n.declare_parameter("train.n_envs", 1)

        n.declare_parameter("eval.episodes", 20)
        n.declare_parameter("eval.deterministic", True)
        n.declare_parameter("eval.render", False)
        n.declare_parameter("eval.video", False)
        n.declare_parameter("eval.model_path", "")
        n.declare_parameter("eval.n_envs", 1)

        gp = n.get_parameter
        meta = MetaCfg(
            namespace=gp("metadata.namespace").value,
            mode=gp("metadata.mode").value,
            task=gp("metadata.task").value,
            artifacts_dir=gp("metadata.artifacts_dir").value,
        )
        algo = AlgoCfg(
            name=(gp("algorithm.name").value or "default").lower(),
            device=gp("algorithm.device").value,
            policy=gp("algorithm.policy").value,
        )
        train = TrainCfg(
            total_timesteps=gp("train.total_timesteps").value,
            save_freq=gp("train.save_freq").value,
            progress=gp("train.progress").value,
            n_envs=gp("train.n_envs").value,
        )
        evalc = EvalCfg(
            episodes=gp("eval.episodes").value,
            deterministic=gp("eval.deterministic").value,
            render=gp("eval.render").value,
            video=gp("eval.video").value,
            model_path=gp("eval.model_path").value,
            n_envs=gp("eval.n_envs").value,
        )

        # Algo defaults + YAML overrides (only declare once)
        algo_params: Dict[str, Any] = {}
        if algo.name != "default":
            try:
                defaults = get_algo_defaults(algo.name)
            except Exception as e:
                n.get_logger().error(f"Cannot load defaults for algo='{algo.name}': {e}")
                defaults = {}
            for k, v in defaults.items():
                n.declare_parameter(f"{algo.name}.{k}", v)
            # Populate only when training; for eval SB3 loads from checkpoint
            if meta.mode == "train":
                for k in defaults.keys():
                    algo_params[k] = gp(f"{algo.name}.{k}").value
                # Post-process target_kl
                tk = algo_params.get("target_kl", None)
                if isinstance(tk, (int, float)):
                    algo_params["target_kl"] = None if tk is None or tk <= 0 else float(tk)
                elif isinstance(tk, str):
                    tks = tk.strip().lower()
                    if tks in ("", "none", "null"):
                        algo_params["target_kl"] = None
                    else:
                        try: algo_params["target_kl"] = float(tk)
                        except ValueError: algo_params["target_kl"] = None
                else:
                    algo_params["target_kl"] = None
                # Parse policy_kwargs
                pk = algo_params.get("policy_kwargs", "{}")
                if isinstance(pk, str):
                    try: algo_params["policy_kwargs"] = json.loads(pk) if pk.strip() else {}
                    except Exception:
                        n.get_logger().warning("policy_kwargs JSON parse failed. Using {}.")
                        algo_params["policy_kwargs"] = {}
                elif not isinstance(pk, dict):
                    algo_params["policy_kwargs"] = {}

        return Config(meta=meta, algo=algo, train=train, eval=evalc, algo_params=algo_params)
